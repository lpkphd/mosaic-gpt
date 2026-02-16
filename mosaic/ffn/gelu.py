"""Standard GELU Feed-Forward Network — the original GPT-2 / BERT FFN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeluFFN(nn.Module):
    """Standard FFN: Linear -> GELU -> Linear."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        hidden_dim = d_model * 4  # Standard 4x expansion

        self.w1 = nn.Linear(d_model, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=True)
        self.dropout = nn.Dropout(cfg.ffn.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))
