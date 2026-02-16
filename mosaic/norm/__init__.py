from mosaic.norm.rmsnorm import RMSNorm
from mosaic.norm.layernorm import LayerNorm

NORM_REGISTRY = {
    "rmsnorm": RMSNorm,
    "layernorm": LayerNorm,
}


def build_norm(norm_type: str, d_model: int, eps: float = 1e-6):
    if norm_type not in NORM_REGISTRY:
        raise ValueError(f"Unknown norm type: {norm_type}. Available: {list(NORM_REGISTRY.keys())}")
    return NORM_REGISTRY[norm_type](d_model, eps=eps)
