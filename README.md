# MOSAIC-GPT

**Modular Open-Source Architecture for Intelligent Computation**

A sub-500M parameter language model that combines every state-of-the-art technique in a fully modular, swappable architecture designed for architecture research.

## Why MOSAIC?

No small-scale model exists that combines **all** modern LLM techniques in one place:

| Technique | SmolLM | Pythia | OLMo | DeepSeek-V3 | **MOSAIC** |
|-----------|--------|--------|------|-------------|------------|
| MLA (Multi-head Latent Attention) | - | - | - | Yes (236B+) | **Yes (150M)** |
| MoE (Mixture of Experts) | - | - | - | Yes | **Yes** |
| SwiGLU | Yes | - | Yes | Yes | **Yes** |
| RoPE (Decoupled) | Yes | Yes | Yes | Yes | **Yes** |
| RMSNorm (Pre-LN) | Yes | - | Yes | Yes | **Yes** |
| Modular/Swappable | - | - | - | - | **Yes** |

MOSAIC brings MLA and MoE down to research-friendly scale (~150M active / ~400M total params) where a single GPU can train the full model.

## Architecture

```
MOSAIC-GPT (~150M active params / ~400M total with MoE)

Embeddings (768d, 32K vocab, tied output head)
  |
  v
12x Transformer Blocks:
  +-- [ATTENTION]  -- swappable: MLA, MHA, GQA, MQA, Linear
  +-- [FFN]        -- swappable: MoE+SwiGLU, Dense SwiGLU, GELU, BitLinear
  +-- [POSITION]   -- swappable: Decoupled RoPE, RoPE, ALiBi, Sinusoidal
  +-- [NORM]       -- swappable: RMSNorm, LayerNorm
  +-- [OUTPUT]     -- swappable: Tied Embeddings, Linear, Factored
  |
  v
Output Head -> Logits
```

Every component follows a standard interface and can be swapped via config:

```yaml
model:
  d_model: 768
  n_layers: 12
  attention:
    type: "mla"           # or "mha", "gqa", "mqa"
    n_heads: 12
    kv_compression_dim: 256
  ffn:
    type: "moe_swiglu"    # or "swiglu", "gelu", "bitlinear"
    n_experts: 8
    top_k: 2
    shared_experts: 1
  position:
    type: "decoupled_rope" # or "rope", "alibi", "sinusoidal"
  norm:
    type: "rmsnorm"        # or "layernorm"
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train MOSAIC-GPT small (single GPU)
python experiments/train.py --config configs/mosaic_small.yaml --device cuda

# Train with 2 GPUs
torchrun --nproc_per_node=2 experiments/train.py --config configs/mosaic_small.yaml

# Evaluate
python experiments/eval.py --checkpoint runs/mosaic_small/best.pt --eval wikitext2
```

## Swapping Components

MOSAIC is designed for architecture research. Swap any component and measure the impact:

```python
from mosaic import MosaicGPT, MosaicConfig

# Load default config (MLA + MoE + SwiGLU)
config = MosaicConfig.from_yaml("configs/mosaic_small.yaml")

# Swap attention to standard MHA
config.attention.type = "mha"

# Swap FFN to dense SwiGLU (no MoE)
config.ffn.type = "swiglu"

# Build and train
model = MosaicGPT(config)
```

## Model Variants

| Config | Active Params | Total Params | Attention | FFN | Notes |
|--------|--------------|-------------|-----------|-----|-------|
| `mosaic_small` | 150M | 400M | MLA | MoE+SwiGLU | Default, full SOTA |
| `mosaic_dense` | 150M | 150M | MLA | Dense SwiGLU | No MoE |
| `gpt2_reference` | 124M | 124M | MHA | GELU | GPT-2 equivalent |
| `mosaic_ternary` | 150M | 400M | MLA | MoE+BitLinear | 1.58-bit weights |

## Pretrained Weights

Available on HuggingFace: [lpkphd/mosaic-gpt](https://huggingface.co/lpkphd/mosaic-gpt)

## Citation

```bibtex
@article{krause2026mosaic,
  title={MOSAIC-GPT: A Modular Open-Source Architecture for Intelligent Computation at Small Scale},
  author={Krause, Lucas},
  journal={arXiv preprint},
  year={2026}
}
```

## License

Apache 2.0
