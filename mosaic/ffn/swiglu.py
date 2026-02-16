"""SwiGLU Feed-Forward Network — used by LLaMA, DeepSeek, Mistral, etc.

SwiGLU replaces the standard GELU FFN with a gated linear unit using
SiLU (Swish) activation. It uses 8/3 * d_model hidden dim instead of
4 * d_model to keep parameter count similar.

Reference: Shazeer, "GLU Variants Improve Transformer" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU: gate * swish(W1 x) * (W3 x), then project back."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        hidden_dim = cfg.ffn_hidden_dim

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(cfg.ffn.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
