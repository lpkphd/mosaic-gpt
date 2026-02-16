"""Standard LayerNorm — included for comparison with RMSNorm."""

import torch.nn as nn


class LayerNorm(nn.Module):
    """Standard Layer Normalization (centers + rescales)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.ln(x)
