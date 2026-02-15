"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from train_oasis.model.rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from train_oasis.model.attention import SpatialAxialAttention, TemporalAxialAttention
from timm.models.vision_transformer import Mlp
from .blocks import (
    PatchEmbed, 
    modulate, 
    gate,
    FinalLayer,
    TimestepEmbedder,
)
from torch.utils.checkpoint import checkpoint

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor, total_buffer_len=124):
        # assert T >= total_buffer_len, "TemporalAxialAttention: length of data must be longer than length of buffer"
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)
    
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        max_frames=32,
        gradient_checkpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, external_cond=None, buffer_size=124, num_samples=10):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        buffer_size: size of buffer
        num_samples: number of samples from the buffer and recent window each (can further differentiate these two)
        """

        B, T, C, H, W = x.shape

        x_out = x.clone()
        available_buffer_len = max(0, T - num_samples)
        buffer_len = min(buffer_size, available_buffer_len)
        window_len = T - available_buffer_len
        assert window_len == num_samples, "DiT Training Error: length of sequence is shorter than the current window size"

        indices = self.sample_exp_buffer(x, buffer_size, num_samples, 1, 2, 5)

        x = x[:, indices]
        t = t[:, indices]
        B, T, C, H, W = x.shape
    
        # add spatial embeddings
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model
        # restore shape
        x = rearrange(x, "(b t) h w d -> b t h w d", t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)  # (N, D)
        c = rearrange(c, "(b t) d -> b t d", t=T)
        if torch.is_tensor(external_cond):
            # use zero arrays for actions associated with buffer/context frames
            e_total = external_cond[:, indices]
            e_total[:, :-window_len] = 0
            c += self.external_cond(e_total.contiguous())
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)  # (N, T, H, W, D)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
        
        # return a tensor that has the same shape as input tensor, but with the current window modified
        x_out[:, -window_len:] = x[:, -window_len:] # first num_samples are from buffer which serve as context
        return x_out 

    def sample_exp_buffer(self, x, total_buffer_len=124, num_samples=10, s=1, r=2, num_segments=5):
        """
        x: [B, T, C, H, W] tensor of latent frames
        total_buffer_len: number of preceding frames to consider
        num_samples: total number of frames to sample from the buffer
        s: stride
        r: expand ratio
        num_segments: number of segments or brackets
        """
        B, T, C, H, W = x.shape
        assert T >= num_samples, "DiT error: length of input x is not equal to buffer size + window size"

        # Determine buffer start
        available_buffer_len = max(0, T - num_samples)
        buffer_len = min(total_buffer_len, available_buffer_len)
        buffer_start = T - num_samples - buffer_len
        window_start = max(0, T - num_samples)
        window_indices = torch.arange(window_start, T, device=x.device)
        if buffer_len <= 0:
            buffer_indices = torch.zeros(num_samples, dtype=torch.long, device=x.device)
            return torch.cat([buffer_indices, window_indices], dim=0)
            # return x[:, :1].expand(B, num_samples, C, H, W), t[:, :1].expand(B, num_samples)
            
        # x_buffer = x[:, buffer_start:buffer_start+buffer_len]
        # t_buffer = t[:, buffer_start:buffer_start+buffer_len]
        segments = []
        seg_total = 0
        samples_per_segment = max(1, num_samples // num_segments)
        for k in range(num_segments):
            seg_len = s * (r ** k)
            seg_total += seg_len
            segments.append(seg_len)

        if seg_total > buffer_len:
            scale = buffer_len / seg_total
            segments = [max(1, int(seg_len * scale)) for seg_len in segments]  # rescale
        
        indices = []
        start = 0
        for seg_len in segments:
            end = min(start + seg_len, buffer_len)
            if start == end == buffer_len:
                seg_indices = [end - 1] * samples_per_segment
            elif end - start >= samples_per_segment:
                seg_indices = torch.randint(start, end, (samples_per_segment,), device=x.device)
                seg_indices = sorted(seg_indices.tolist())
            else:
                pad_val = end - 1 if end > start else start
                seg_indices = list(range(start, end)) + [pad_val] * (samples_per_segment - (end - start))
            indices.extend(seg_indices)
            start = end
        
        buffer_indices = torch.tensor(indices[:num_samples], device=x.device) + buffer_start
        assert len(buffer_indices) == num_samples, "number of samples from buffer != num_samples"
        # x_sampled = x_buffer[:, indices].contiguous()  # [B, num_samples, C, H, W]
        # t_sampled = t_buffer[:, indices].contiguous()
        return torch.cat([buffer_indices, window_indices], dim=0)


def DiT_S_2():
    return DiT(
        patch_size=2,
        hidden_size=1024,
        depth=16,
        num_heads=16,
    )

def flappy_bird_dit():
    return DiT(
        input_h=64,
        input_w=36,
        in_channels=4,
        patch_size=2,
        hidden_size=512,
        depth=8,
        num_heads=16,
        external_cond_dim=2,
        max_frames=10
    )


def self_train():
    return DiT(
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=4,
        max_frames=10,
        gradient_checkpointing=True,
        dtype=torch.bfloat16
    )

DiT_models = {
    "DiT-S/2": DiT_S_2,
    "flappy_bird_dit": flappy_bird_dit,
    "self_train_mc": self_train,
}