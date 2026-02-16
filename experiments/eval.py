"""Evaluate a trained MOSAIC-GPT checkpoint.

Usage:
    python experiments/eval.py --checkpoint runs/mosaic_small/best.pt --device cuda
"""

import argparse
import os
import sys
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic.config import MosaicConfig
from mosaic.model import MosaicGPT
from training.data import build_eval_loader
from training.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOSAIC-GPT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-steps", type=int, default=0, help="Max eval steps (0 = all)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device(args.device)

    model = MosaicGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    print(model.summary())
    print(f"\nCheckpoint from step {ckpt.get('step', '?')}")
    if ckpt.get("metrics"):
        print(f"Checkpoint metrics: {ckpt['metrics']}")

    eval_loader = build_eval_loader(cfg, split="test")
    max_steps = args.eval_steps if args.eval_steps > 0 else 10000

    print(f"\nEvaluating on WikiText-2 test set...")
    metrics = evaluate(model, eval_loader, device, max_steps=max_steps)
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Tokens evaluated: {metrics['tokens']:,}")


if __name__ == "__main__":
    main()
