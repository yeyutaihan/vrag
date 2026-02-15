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
from train_oasis.model.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from train_oasis.utils import sigmoid_beta_schedule
from einops import rearrange
from torch import einsum
from timm.models.vision_transformer import Mlp
from tqdm import tqdm
from .blocks import (
    PatchEmbed, 
    modulate, 
    gate,
    FinalLayer,
    TimestepEmbedder,
)
from torch.utils.checkpoint import checkpoint

def sigma_act(x):
    return F.elu(x) + 1

'''
0 0 0 0 0 0 0 0
        1 1 1 1 1 1 1 1
'''
def updateMemory(kv_mem, k, v, z, stride, delta=False):
    k = k[:, :, :stride]
    v = v[:, :, :stride]
    if kv_mem is not None:
        assert kv_mem.shape[1] == v.shape[1], "Memory length must be the same as the number of new keys"
    if delta and (kv_mem is not None):
        sigma_k = sigma_act(k)
        numerator = einsum(
            "b h N k, b h k v -> b h N v",
            sigma_k,
            kv_mem,
        )
        denominator = einsum(
            "b h N k, b h k -> b h N",
            sigma_k,
            z,
        )
        denominator = rearrange(
            denominator,
            "b h N -> b h N 1",
        )
        prev_v = numerator / denominator
        new_value_states = v - prev_v
        new_kv_mem = torch.matmul(sigma_k.transpose(-2, -1), new_value_states)
    else:
        k_T = rearrange(sigma_act(k), "B h N d -> B h d N")
        new_kv_mem = sigma_act(k_T) @ v
    if kv_mem is not None:
        new_kv_mem = kv_mem + new_kv_mem
    return new_kv_mem

def getAttnMem(kv_mem, z, q):
    assert kv_mem is not None, "Attention memory must be provided"
    sigma_q = sigma_act(q)
    retrieved_memory = einsum(
        "b h N k, b h k v -> b h N v",
        sigma_q,
        kv_mem,
    )

    denominator = einsum(
        "b h N d, b h d -> b h N",
        sigma_q,
        z,
    )
    denominator = rearrange(
        denominator,
        "b h N -> b h N 1",
    )
    retrieved_memory = retrieved_memory / denominator
    return retrieved_memory

def updateZ(z, k):
    # k: (B, h, N, d)
    k = sigma_act(k)
    new_z = torch.sum(k, dim=2, keepdim=False)
    if z is not None:
        new_z = z + new_z
    return new_z # (B, h, d)

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        stride: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        attn_drop: float = 0.0,
        delta_update: bool = False,
        bptt: bool = False,
    ):
        super().__init__()
        self.delta_update = delta_update
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.stride = stride
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.bptt = bptt

        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5

        self.mem_beta = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(self, x: torch.Tensor, kv_t):
        B, T, H, W, D = x.shape

        if kv_t is None:
            mem, z = None, None
        else:
            mem, z = kv_t

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        if self.bptt:
            # new_mem = updateMemory(mem, k, v, z, self.stride, self.delta_update)
            new_mem = updateMemory(mem, k, v, z, 8, self.delta_update)
            new_z = updateZ(z, k)
        else:
            with torch.no_grad():
                new_mem = updateMemory(mem, k, v, z, self.stride, self.delta_update)
                new_z = updateZ(z, k)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        if mem is not None:
            mem_x = getAttnMem(mem, z, q)
            x = F.sigmoid(self.mem_beta) * mem_x + (1 - F.sigmoid(self.mem_beta)) * x

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)
        # linear proj
        x = self.to_out(x)
        return x, (new_mem, new_z)


class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        stride: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        attn_drop: float = 0.0,
        delta_update: bool = False,
        bptt: bool = False,
    ):
        super().__init__()
        self.delta_update = delta_update
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.stride = stride
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5
        self.bptt = bptt

        self.mem_beta = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(self, x: torch.Tensor, kv_s):
        B, T, H, W, D = x.shape
        if kv_s is None:
            mem, z = None, None
        else:
            mem, z = kv_s

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        if self.bptt:
            new_mem = updateMemory(mem, rearrange(k, "BT h H W d -> BT h (H W) d"), rearrange(v, "BT h H W d -> BT h (H W) d"), z, self.stride, self.delta_update)
            new_z = updateZ(z, rearrange(k, "BT h H W d -> BT h (H W) d"))
        else:
            with torch.no_grad():
                new_mem = updateMemory(mem, rearrange(k, "BT h H W d -> BT h (H W) d"), rearrange(v, "BT h H W d -> BT h (H W) d"), z, self.stride, self.delta_update)
                new_z = updateZ(z, rearrange(k, "BT h H W d -> BT h (H W) d"))
        
        freqs = self.rotary_emb.get_axial_freqs(H, W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        if mem is not None:
            mem_x = getAttnMem(mem, z, q)
            x = F.sigmoid(self.mem_beta) * mem_x + (1 - F.sigmoid(self.mem_beta)) * x

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x, (new_mem, new_z)

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        stride,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
        delta_update: bool = False,
        bptt: bool = False,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            stride=stride,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
            delta_update=delta_update,
            bptt=bptt,
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
            stride=stride,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
            delta_update=delta_update,
            bptt=bptt,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, kv_mem):
        B, T, H, W, D = x.shape
        if kv_mem is None:
            kv_s, kv_t = None, None
        else:
            kv_s, kv_t = kv_mem
        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        # x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x, new_kv_s = self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa), kv_s)
        x = x + gate(x, s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        # x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x, new_kv_t = self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa), kv_t)
        x = x + gate(x, t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x, (new_kv_s, new_kv_t)


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
        stride=5,
        stabilization_level=15,
        clip_noise=6,
        timesteps=1000,
        delta_update=False,
        bptt=False,
        gradient_ckeckpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.stride = stride
        self.stabilization_level = stabilization_level
        self.clip_noise = clip_noise
        self.timesteps = timesteps
        self.delta_update = delta_update
        self.bptt = bptt
        self.gradient_ckeckpointing = gradient_ckeckpointing
        self.dtype = dtype

        assert stride*2 == max_frames, "Stride must be half of max_frames"

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
                    stride=stride,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                    delta_update=delta_update,
                    bptt=bptt,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self._build_buffer()

    def _build_buffer(self):
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # diffusion forcing
        betas = sigmoid_beta_schedule(self.timesteps).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
        register_buffer("alphas_cumprod", alphas_cumprod)

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

    def forward(self, x, t, external_cond=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """

        B, T, C, H, W = x.shape

        kv_mem = None
        last_end = 0
        for start, end in zip(range(0, T - self.max_frames + self.stride, self.stride), range(self.max_frames, T + self.stride, self.stride)):
            if end > T:
                end = T
            x_slice = x[:, start:end]
            t_slice = t[:, start:end]
            # if last_end != 0:
            #     t_slice[:, :last_end - end] = self.stabilization_level
            if torch.is_tensor(external_cond):
                external_cond_slice = external_cond[:, start:end]
            else:
                external_cond_slice = None
            x_slice, kv_mem = self.forward_one_step(x_slice, t_slice, kv_mem, external_cond_slice)
            # if start != last_end:
            #     # These frames have been denoised
            #     x_output[:, start:last_end] = x_slice[:, :last_end - end]
            x[:, last_end:end] = x_slice[:, last_end - end:]
            last_end = end
        
        return x

    def forward_one_step(self, x, t, kv_mem, external_cond):
        B, T = x.shape[:2]
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
            c += self.external_cond(external_cond)

        new_kv_mem = []
        if kv_mem is None:
            # first step
            kv_mem = [None] * len(self.blocks)
        for block, one_kv_mem in zip(self.blocks, kv_mem):
            if self.gradient_ckeckpointing and self.training:
                x, new_one_kv_mem = checkpoint(block, x, c, one_kv_mem, use_reentrant=False)
            else:
                x, new_one_kv_mem = block(x, c, one_kv_mem)  # (N, T, H, W, D)
            new_kv_mem.append(new_one_kv_mem)
        if self.gradient_ckeckpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x, new_kv_mem

    @torch.no_grad()
    def sample(self, x, n_context_frames, n_frames, sampling_timesteps, external_cond=None):
        B = x.shape[0]
        kv_mem = None
        last_end = n_context_frames
        pbar = tqdm(total=len(range(n_context_frames - self.stride, n_frames - self.max_frames + self.stride, self.stride)), desc="Inference")
        for start, end in zip(range(n_context_frames - self.stride, n_frames - self.max_frames + self.stride, self.stride), range(n_context_frames - self.stride + self.max_frames, n_frames + self.stride, self.stride)):
            if end > n_frames:
                end = n_frames
            pbar.set_postfix(
                {
                    "start": start,
                    "end": end,
                }
            )
            chunk = torch.randn((B, end - last_end, *x.shape[-3:]), device=x.device) # (B, T, C, H, W)
            chunk = torch.clamp(chunk, -self.clip_noise, +self.clip_noise)
            chunk = chunk.type_as(x)
            x_slice = x[:, start:]
            x_slice = torch.cat([x_slice, chunk], dim=1)
            if torch.is_tensor(external_cond):
                external_cond_slice = external_cond[:, start:end]
            else:
                external_cond_slice = None
            x_slice, kv_mem = self.sample_one_step(x=x_slice, kv_mem=kv_mem, external_cond=external_cond_slice, sampling_timesteps=sampling_timesteps, n_context_frames=last_end-start)
            # if start != last_end:
            #     # These frames have been denoised
            #     x_output[:, start:last_end] = x_slice[:, :last_end - end]
            x = torch.cat([x, x_slice[:, last_end - end:]], dim=1)
            last_end = end
            pbar.update(1)
        pbar.close()

        return x

    @torch.no_grad()
    def sample_one_step(self, x, kv_mem, external_cond, sampling_timesteps, n_context_frames):
        B, T = x.shape[:2]
        noise_range = torch.linspace(-1, self.timesteps - 1, sampling_timesteps + 1)
        for noise_idx in reversed(range(1, sampling_timesteps + 1)):
            # set up noise values
            t_ctx = torch.full((B, n_context_frames), self.stabilization_level - 1, dtype=torch.long, device=x.device)
            t = torch.full((B, T - n_context_frames), noise_range[noise_idx], dtype=torch.long, device=x.device)
            t_next = torch.full((B, T - n_context_frames), noise_range[noise_idx - 1], dtype=torch.long, device=x.device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1) # (B, T)
            t_next = torch.cat([t_ctx, t_next], dim=1) # (B, T)

            # get model predictions
            x_start, new_kv_mem = self.forward_one_step(x, t, kv_mem, external_cond)
            x_noise = ((1 / self.alphas_cumprod[t]).sqrt() * x - x_start) / (1 / self.alphas_cumprod[t] - 1).sqrt()
            # get frame prediction
            alpha_next = self.alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, n_context_frames:] = x_pred[:, n_context_frames:]

        return x, new_kv_mem