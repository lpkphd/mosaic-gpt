"""Mixture of Experts FFN — DeepSeek-style with shared expert.

Routes each token to top-k experts out of n_experts total. Includes
an optional shared expert (always active) as used in DeepSeek-V3.
Load balancing loss encourages even expert utilization.

References:
    - Switch Transformers (Fedus et al., 2021)
    - DeepSeek-V2 MoE: https://arxiv.org/abs/2405.04434
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mosaic.ffn.swiglu import SwiGLUFFN


class MoEFFN(nn.Module):
    """Mixture of Experts with SwiGLU experts and optional shared expert."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.n_experts = cfg.ffn.n_experts
        self.top_k = cfg.ffn.top_k
        self.n_shared = cfg.ffn.shared_experts
        self.load_balance_weight = cfg.ffn.load_balance_weight

        # Router: projects input to expert scores
        self.router = nn.Linear(d_model, self.n_experts, bias=False)

        # Expert networks (each is a SwiGLU FFN)
        self.experts = nn.ModuleList([
            SwiGLUFFN(cfg, d_model) for _ in range(self.n_experts)
        ])

        # Shared expert(s) — always active, not routed
        self.shared_experts = nn.ModuleList([
            SwiGLUFFN(cfg, d_model) for _ in range(self.n_shared)
        ]) if self.n_shared > 0 else None

        self._aux_loss = 0.0

    @property
    def aux_loss(self) -> float:
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        n_tokens = x_flat.shape[0]

        # Router logits and top-k selection
        router_logits = self.router(x_flat)  # (B*T, n_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B*T, top_k)

        # Compute load balancing auxiliary loss
        if self.training:
            self._compute_aux_loss(router_logits, top_k_indices, n_tokens)

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*T,)
            weights = top_k_weights[:, k]  # (B*T,)

            for e_idx in range(self.n_experts):
                mask = (expert_indices == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output

        # Add shared expert output
        if self.shared_experts is not None:
            for shared in self.shared_experts:
                output = output + shared(x_flat)

        return output.view(B, T, C)

    def _compute_aux_loss(self, router_logits, top_k_indices, n_tokens):
        """Load balancing loss to encourage even expert utilization."""
        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(top_k_indices, self.n_experts).float()
        tokens_per_expert = one_hot.sum(dim=0).sum(dim=0) / n_tokens  # (n_experts,)

        # Average router probability for each expert
        router_probs = F.softmax(router_logits, dim=-1)
        avg_prob = router_probs.mean(dim=0)  # (n_experts,)

        # Auxiliary loss: dot product of fraction * probability
        self._aux_loss = self.load_balance_weight * self.n_experts * (tokens_per_expert * avg_prob).sum()
