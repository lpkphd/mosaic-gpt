"""Grouped Query Attention (GQA) — used by LLaMA 2/3, Mistral, etc.

Groups multiple query heads to share a single KV head, reducing
KV cache size without significant quality loss. When n_kv_heads=1,
this becomes Multi-Query Attention (MQA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mosaic.position.rope import RotaryEmbedding, apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """GQA: n_heads query heads share n_kv_heads key-value heads."""

    def __init__(self, cfg, d_model: int, n_kv_heads: int = None):
        super().__init__()
        self.n_heads = cfg.attention.n_heads
        self.n_kv_heads = n_kv_heads or cfg.attention.n_kv_heads or self.n_heads
        self.d_head = d_model // self.n_heads
        self.d_model = d_model
        self.n_rep = self.n_heads // self.n_kv_heads  # Heads per KV group

        self.W_q = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.attention.dropout)
        self.rope = RotaryEmbedding(self.d_head, max_seq_len=cfg.position.max_seq_len)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query head count."""
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return x.unsqueeze(2).expand(B, n_kv, self.n_rep, T, D).reshape(B, self.n_heads, T, D)

    def forward(self, x: torch.Tensor, kv_cache=None):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV to match query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None),
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attn_out), new_kv_cache
