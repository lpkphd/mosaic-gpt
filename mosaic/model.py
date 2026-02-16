"""
MOSAIC-GPT: Top-level model that assembles all swappable components.

This is a decoder-only transformer where every component (attention, FFN,
normalization, positional encoding, output head) is configured via the
MosaicConfig and can be swapped without changing this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mosaic.config import MosaicConfig
from mosaic.attention import build_attention
from mosaic.ffn import build_ffn
from mosaic.norm import build_norm


class TransformerBlock(nn.Module):
    """Single transformer block with Pre-LN architecture."""

    def __init__(self, cfg: MosaicConfig):
        super().__init__()
        d = cfg.d_model

        # Pre-attention norm
        self.attn_norm = build_norm(cfg.norm.type, d, cfg.norm.eps)
        # Attention (swappable: MHA, GQA, MQA, MLA)
        self.attn = build_attention(cfg, d)
        # Pre-FFN norm
        self.ffn_norm = build_norm(cfg.norm.type, d, cfg.norm.eps)
        # FFN (swappable: SwiGLU, GELU, MoE)
        self.ffn = build_ffn(cfg, d)

        self.residual_dropout = nn.Dropout(cfg.residual_dropout)

    def forward(self, x: torch.Tensor, kv_cache=None):
        # Attention with residual
        attn_out, new_kv_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = x + self.residual_dropout(attn_out)

        # FFN with residual
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))

        return x, new_kv_cache


class MosaicGPT(nn.Module):
    """MOSAIC-GPT: Modular Open-Source Architecture for Intelligent Computation.

    A decoder-only transformer where every component is swappable via config.

    Args:
        cfg: MosaicConfig specifying all component types and hyperparameters.
    """

    def __init__(self, cfg: MosaicConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.embed_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        # Final norm
        self.final_norm = build_norm(cfg.norm.type, cfg.d_model, cfg.norm.eps)

        # Output head
        if cfg.output.type == "tied":
            self.output_head = None  # Will use tok_emb.weight
        else:
            self.output_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) per GPT-2
        for block in self.blocks:
            if hasattr(block.attn, 'W_o'):
                block.attn.W_o.weight.data *= (2 * cfg.n_layers) ** -0.5
            if hasattr(block.ffn, 'w2'):
                block.ffn.w2.weight.data *= (2 * cfg.n_layers) ** -0.5

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, kv_caches=None):
        """Forward pass.

        Args:
            idx: Token indices, shape (batch, seq_len)
            kv_caches: Optional list of KV caches for each layer (inference)

        Returns:
            logits: Shape (batch, seq_len, vocab_size)
            new_kv_caches: Updated KV caches for each layer
        """
        B, T = idx.shape

        # Token embeddings (no separate positional encoding — RoPE is in attention)
        x = self.embed_dropout(self.tok_emb(idx))

        # Transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache)
            new_kv_caches.append(new_cache)

        # Final norm + output projection
        x = self.final_norm(x)

        if self.output_head is not None:
            logits = self.output_head(x)
        else:
            # Tied embeddings
            logits = F.linear(x, self.tok_emb.weight)

        return logits, new_kv_caches

    def get_aux_loss(self) -> torch.Tensor:
        """Collect auxiliary losses from MoE layers."""
        aux = 0.0
        for block in self.blocks:
            if hasattr(block.ffn, 'aux_loss'):
                aux = aux + block.ffn.aux_loss
        return aux

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_active_params(self) -> int:
        """Approximate active parameters per token (excludes inactive MoE experts)."""
        total = self.num_params
        for block in self.blocks:
            if hasattr(block.ffn, 'experts') and hasattr(block.ffn, 'n_experts'):
                # Each expert has the same size; only top_k are active
                n_experts = block.ffn.n_experts
                top_k = block.ffn.top_k
                expert_params = sum(p.numel() for p in block.ffn.experts[0].parameters())
                inactive = expert_params * (n_experts - top_k)
                total -= inactive
        return total

    def summary(self) -> str:
        lines = [
            f"MOSAIC-GPT Configuration:",
            f"  Layers: {self.cfg.n_layers}",
            f"  d_model: {self.cfg.d_model}",
            f"  Attention: {self.cfg.attention.type} ({self.cfg.attention.n_heads} heads)",
            f"  FFN: {self.cfg.ffn.type}",
            f"  Norm: {self.cfg.norm.type}",
            f"  Position: {self.cfg.position.type}",
            f"  Output: {self.cfg.output.type}",
            f"  Total params: {self.num_params:,}",
            f"  Active params/token: {self.num_active_params:,}",
            f"  Vocab size: {self.cfg.vocab_size:,}",
        ]
        if self.cfg.ffn.type == "moe_swiglu":
            lines.insert(6, f"    Experts: {self.cfg.ffn.n_experts} (top-{self.cfg.ffn.top_k}, {self.cfg.ffn.shared_experts} shared)")
        return "\n".join(lines)
