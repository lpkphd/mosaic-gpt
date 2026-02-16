from mosaic.attention.mha import MultiHeadAttention
from mosaic.attention.gqa import GroupedQueryAttention
from mosaic.attention.mla import MultiHeadLatentAttention

ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "gqa": GroupedQueryAttention,
    "mqa": lambda cfg, d_model: GroupedQueryAttention(cfg, d_model, n_kv_heads=1),
    "mla": MultiHeadLatentAttention,
}


def build_attention(cfg, d_model: int):
    attn_type = cfg.attention.type
    if attn_type not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type: {attn_type}. Available: {list(ATTENTION_REGISTRY.keys())}")
    return ATTENTION_REGISTRY[attn_type](cfg, d_model)
