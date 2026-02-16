"""Standard Multi-Head Attention — the original Transformer mechanism.

Included as a baseline for comparing against MLA, GQA, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mosaic.position.rope import RotaryEmbedding, apply_rotary_emb


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention with RoPE support and KV caching."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.n_heads = cfg.attention.n_heads
        self.d_head = d_model // self.n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.attention.dropout)
        self.rope = RotaryEmbedding(self.d_head, max_seq_len=cfg.position.max_seq_len)

    def forward(self, x: torch.Tensor, kv_cache=None):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None),
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attn_out), new_kv_cache
