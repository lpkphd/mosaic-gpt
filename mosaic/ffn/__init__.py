from mosaic.ffn.swiglu import SwiGLUFFN
from mosaic.ffn.gelu import GeluFFN
from mosaic.ffn.moe import MoEFFN

FFN_REGISTRY = {
    "swiglu": SwiGLUFFN,
    "gelu": GeluFFN,
    "moe_swiglu": MoEFFN,
}


def build_ffn(cfg, d_model: int):
    ffn_type = cfg.ffn.type
    if ffn_type not in FFN_REGISTRY:
        raise ValueError(f"Unknown FFN type: {ffn_type}. Available: {list(FFN_REGISTRY.keys())}")
    return FFN_REGISTRY[ffn_type](cfg, d_model)
