"""
Configuration system for MOSAIC-GPT.

Every component is configured via a dataclass hierarchy. Swap components
by changing the `type` field in the config or YAML file.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class AttentionConfig:
    type: str = "mla"  # "mla", "mha", "gqa", "mqa"
    n_heads: int = 12
    n_kv_heads: Optional[int] = None  # For GQA; None = same as n_heads
    # MLA-specific
    kv_compression_dim: int = 256  # Latent dimension for KV compression
    q_compression_dim: int = 384  # Latent dimension for Q compression
    rope_dim: int = 64  # Decoupled RoPE dimension
    dropout: float = 0.0


@dataclass
class FFNConfig:
    type: str = "moe_swiglu"  # "moe_swiglu", "swiglu", "gelu", "bitlinear"
    hidden_mult: float = 8 / 3  # SwiGLU uses 8/3 instead of 4x
    # MoE-specific
    n_experts: int = 8
    top_k: int = 2
    shared_experts: int = 1  # DeepSeek-style shared expert
    load_balance_weight: float = 0.01
    dropout: float = 0.0


@dataclass
class PositionConfig:
    type: str = "decoupled_rope"  # "decoupled_rope", "rope", "alibi", "sinusoidal"
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    rope_dim: Optional[int] = None  # None = d_head


@dataclass
class NormConfig:
    type: str = "rmsnorm"  # "rmsnorm", "layernorm"
    eps: float = 1e-6


@dataclass
class OutputConfig:
    type: str = "tied"  # "tied", "linear"


@dataclass
class TrainingConfig:
    batch_size: int = 64
    seq_len: int = 1024
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 50000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    # Data
    dataset: str = "HuggingFaceFW/fineweb-edu"
    tokenizer: str = "gpt2"  # Or path to custom tokenizer
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2500
    eval_steps: int = 100


@dataclass
class MosaicConfig:
    # Model dimensions
    d_model: int = 768
    n_layers: int = 12
    vocab_size: int = 50257  # GPT-2 tokenizer default
    # Component configs
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Global
    embed_dropout: float = 0.0
    residual_dropout: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "MosaicConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "MosaicConfig":
        config = cls()
        for key, val in d.items():
            if key == "attention" and isinstance(val, dict):
                config.attention = AttentionConfig(**val)
            elif key == "ffn" and isinstance(val, dict):
                config.ffn = FFNConfig(**val)
            elif key == "position" and isinstance(val, dict):
                config.position = PositionConfig(**val)
            elif key == "norm" and isinstance(val, dict):
                config.norm = NormConfig(**val)
            elif key == "output" and isinstance(val, dict):
                config.output = OutputConfig(**val)
            elif key == "training" and isinstance(val, dict):
                config.training = TrainingConfig(**val)
            elif hasattr(config, key):
                setattr(config, key, val)
        return config

    def to_yaml(self, path: str):
        import dataclasses
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @property
    def d_head(self) -> int:
        return self.d_model // self.attention.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """Hidden dimension for FFN, rounded to nearest multiple of 256."""
        raw = int(self.d_model * self.ffn.hidden_mult)
        return ((raw + 255) // 256) * 256
