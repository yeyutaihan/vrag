"""
Based on https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/attention.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from train_oasis.model.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from train_oasis.model.yarn import LlamaYaRNScaledRotaryEmbedding, apply_rotary_pos_emb
from typing import Union


class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: Union[RotaryEmbedding, LlamaYaRNScaledRotaryEmbedding],
        is_causal: bool = True,
        attn_drop: float = 0.0,
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

        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        if isinstance(self.rotary_emb, RotaryEmbedding):
            q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
            k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)
        elif isinstance(self.rotary_emb, LlamaYaRNScaledRotaryEmbedding): 
            cos, sin = self.rotary_emb(v, T) # shape of cos/sin: (T, d)
            position_ids = torch.arange(T, device=q.device).expand(B * H * W, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)
        else:
            raise ValueError(f"TemporalAxialAttention: {self.rotary_emb} is not supported")

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x


class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: Union[RotaryEmbedding, LlamaYaRNScaledRotaryEmbedding],
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if isinstance(self.rotary_emb, RotaryEmbedding):
            q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
            k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
            v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        
            freqs = self.rotary_emb.get_axial_freqs(H, W)
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

            # prepare for attn
            q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
            k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
            v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        elif isinstance(self.rotary_emb, LlamaYaRNScaledRotaryEmbedding):
            q = rearrange(q, "B T H W (h d) -> (B T) h (H W) d", h=self.heads)
            k = rearrange(k, "B T H W (h d) -> (B T) h (H W) d", h=self.heads)
            v = rearrange(v, "B T H W (h d) -> (B T) h (H W) d", h=self.heads)

            cos, sin = self.rotary_emb(v, H*W) # shape of cos/sin: (HW, d)
            position_ids = torch.arange(H*W).expand(B*T, H*W)
            # print(f"SpatialAxialAttention: q.shape: {q.shape}, cos.shape: {cos.shape}, sin.shape: {sin.shape}")
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)

        else:
            raise ValueError(f"SpatialAxialAttention: {self.rotary_emb} is not supported")

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x

approx_gelu = lambda: nn.GELU(approximate="tanh")

def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        rope=None,
        qk_norm_legacy: bool = False,
        use_causal_mask: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.use_causal_mask = use_causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = rearrange(x, "b t h w c -> b (t h w) c")
        N = x.shape[1]
        # flash attn is not memory efficient for small sequences, this is empirical
        qkv = self.qkv(x)
        # qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        # qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        qkv = rearrange(qkv, "b n (u h d) -> u b h n d", u=3, b=B, n=N, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if self.use_causal_mask and self.training:
            # query (B, ..., heads, N, dim)
            # attn_mask: bool (B,..., N, N)
            causal_mask = torch.tril(torch.ones(T, T))
            causal_mask = causal_mask.repeat_interleave(H * W, dim=0)  # Expand rows
            causal_mask = causal_mask.repeat_interleave(H * W, dim=1)  # Expand columns
            causal_mask = causal_mask.bool().to(q.device)
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=causal_mask, is_causal=False) # (B, H, N, D)
        else:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        # x_output_shape = (B, N, C)
        # x = x.reshape(x_output_shape)
        x = rearrange(x, "b h n d -> b n (h d)", b=B, n=N, h=self.num_heads, d=self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)
        return x

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(
        self, 
        dim, 
        n_heads, 
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # query projection
        # self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        # self.q_norm = RMSNorm(self.q_lora_rank)
        # self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, rope_func=None):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        # q = self.wq_b(self.q_norm(self.wq_a(x))) # (bsz, seqlen, n_heads * qk_head_dim)
        q = self.wq(x)
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim) # (bsz, seqlen, n_heads, qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # (bsz, seqlen, n_heads, qk_nope_head_dim), (bsz, seqlen, n_heads, qk_rope_head_dim)
        q_pe = rearrange(q_pe, "b s h d -> b s (h d)")
        q_pe = rope_func(q_pe)
        q_pe = rearrange(q_pe, "b s (h d) -> b s h d", h=self.n_heads)
        q = torch.cat([q_nope, q_pe], dim=-1) # (bsz, seqlen, n_heads, qk_head_dim)
        kv = self.wkv_a(x) # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # (bsz, seqlen, kv_lora_rank), (bsz, seqlen, qk_rope_head_dim)
        k_pe = rope_func(k_pe)

        kv = self.kv_norm(kv) # (bsz, seqlen, kv_lora_rank)
        # kv = self.wkv_b(kv) # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        # kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        # k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1) # (bsz, seqlen, n_heads, qk_head_dim)
        
        wkv_b = self.wkv_b.weight
        wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank) # (n_heads, self.qk_nope_head_dim + self.v_head_dim, kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        scores = (torch.einsum("bshc,btc->bsht", q_nope, kv) +
                    torch.einsum("bshr,btr->bsht", q_pe, k_pe)) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,btc->bshc", scores, kv)
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = rearrange(x, "b s h d -> b s (h d)")
        x = self.wo(x)
        return x

class TemporalAxialMLA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        lora_rank: int,
        is_causal: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.rotary_emb = rotary_emb
        rope_func = lambda x: self.rotary_emb.rotate_queries_or_keys(x, self.rotary_emb.freqs)
        self.rope_func = rope_func
        self.attn = MLA(
            dim=dim,
            n_heads=heads,
            kv_lora_rank=lora_rank,
            qk_nope_head_dim=dim_head,
            qk_rope_head_dim=dim_head // 2,
            v_head_dim=dim_head,
        )
        self.is_causal = is_causal
        self.attn_drop = attn_drop

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape
        x = rearrange(x, "B T H W D -> (B H W) T D")
        if self.is_causal and self.training:
            mask = torch.tril(torch.ones(T, T)).to(x.device)
        else:
            mask = None
        x = self.attn(x=x, mask=mask, rope_func=self.rope_func)
        x = rearrange(x, "(B H W) T D -> B T H W D", B=B, H=H, W=W, T=T, D=D)
        return x


class SpatialAxialMLA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        lora_rank: int,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.attn = MLA(
            dim=dim,
            n_heads=heads,
            kv_lora_rank=lora_rank,
            qk_nope_head_dim=dim_head,
            qk_rope_head_dim=dim_head // 2,
            v_head_dim=dim_head,
        )
        self.attn_drop = attn_drop

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape
        x = rearrange(x, "B T H W D -> (B T) (H W) D")
        def rope_func(x):
            freqs = self.rotary_emb.get_axial_freqs(H, W)
            x = rearrange(x, "(B T) (H W) D -> (B T) H W D", B=B, T=T, H=H, W=W)
            x = apply_rotary_emb(freqs, x)
            x = rearrange(x, "(B T) H W D -> (B T) (H W) D", B=B, T=T, H=H, W=W)
            return x

        x = self.attn(x=x, mask=None, rope_func=rope_func)
        x = rearrange(x, "(B T) (H W) D -> B T H W D", B=B, T=T, H=H, W=W, D=D)
        return x