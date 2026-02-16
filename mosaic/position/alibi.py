"""ALiBi (Attention with Linear Biases) — alternative to RoPE.

Adds a linear bias to attention scores based on distance between tokens.
No learned parameters, naturally supports length extrapolation.
"""

import torch
import math


def build_alibi_bias(n_heads: int, max_seq_len: int) -> torch.Tensor:
    """Build ALiBi attention bias matrix.

    Returns:
        Tensor of shape (n_heads, max_seq_len, max_seq_len)
    """

    def _get_slopes(n: int):
        ratio = 2 ** (-8 / n)
        return [ratio ** i for i in range(1, n + 1)]

    slopes = torch.tensor(_get_slopes(n_heads), dtype=torch.float32)
    # Distance matrix: positions[i] - positions[j]
    positions = torch.arange(max_seq_len)
    dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
    # Bias: slope * -|distance|, only for causal (j <= i)
    bias = slopes.unsqueeze(1).unsqueeze(2) * dist.unsqueeze(0).abs().neg()
    return bias
