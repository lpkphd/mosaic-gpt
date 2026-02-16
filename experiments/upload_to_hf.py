"""Upload a trained MOSAIC-GPT checkpoint to HuggingFace Hub.

Usage:
    python experiments/upload_to_hf.py \
        --checkpoint runs/mosaic_small/best.pt \
        --repo lpkphd/mosaic-gpt-small \
        --token $HF_TOKEN
"""

import argparse
import os
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Upload MOSAIC-GPT to HuggingFace")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True, help="HF repo (e.g. lpkphd/mosaic-gpt-small)")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: provide --token or set HF_TOKEN env var")
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(args.repo, token=token, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # Save model weights as safetensors-compatible state dict
    import dataclasses
    tmp_dir = Path("/tmp/mosaic_hf_upload")
    tmp_dir.mkdir(exist_ok=True)

    torch.save(ckpt["model_state"], tmp_dir / "model.pt")

    config_dict = dataclasses.asdict(cfg)
    with open(tmp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Model card
    card = f"""---
license: apache-2.0
tags:
  - mosaic-gpt
  - modular-transformer
  - mla
  - moe
---

# MOSAIC-GPT

Modular Open-Source Architecture for Intelligent Computation.

## Architecture
- **Attention**: {cfg.attention.type}
- **FFN**: {cfg.ffn.type}
- **Norm**: {cfg.norm.type}
- **Position**: {cfg.position.type}
- **d_model**: {cfg.d_model}
- **Layers**: {cfg.n_layers}
- **Total params**: {sum(p.numel() for p in torch.nn.Module().parameters())}

## Training
- Step: {ckpt.get('step', 'N/A')}
- Metrics: {ckpt.get('metrics', {})}

## Usage
```python
from mosaic.config import MosaicConfig
from mosaic.model import MosaicGPT
import torch

ckpt = torch.load("model.pt", map_location="cpu")
config = MosaicConfig.from_yaml("config.yaml")
model = MosaicGPT(config)
model.load_state_dict(ckpt)
```
"""
    with open(tmp_dir / "README.md", "w") as f:
        f.write(card)

    # Upload
    api.upload_folder(
        folder_path=str(tmp_dir),
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
