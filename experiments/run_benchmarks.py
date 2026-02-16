"""Run all evaluation benchmarks on a trained MOSAIC-GPT checkpoint.

Usage:
    python experiments/run_benchmarks.py --checkpoint runs/mosaic_small/best.pt --device cuda
    python experiments/run_benchmarks.py --checkpoint runs/mosaic_small/best.pt --device mps --max-examples 100
    python experiments/run_benchmarks.py --checkpoint runs/mosaic_small/best.pt --benchmarks wikitext2 lambada
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic.config import MosaicConfig
from mosaic.model import MosaicGPT
from eval.benchmarks import run_all


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MosaicGPT, MosaicConfig, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = MosaicGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    meta = {
        "step": ckpt.get("step", "unknown"),
        "checkpoint_metrics": ckpt.get("metrics", {}),
    }
    return model, cfg, meta


def print_results_table(results: dict, meta: dict):
    print("\n" + "=" * 60)
    print("MOSAIC-GPT Benchmark Results")
    print(f"Checkpoint step: {meta['step']}")
    print("=" * 60)
    print(f"{'Benchmark':<15} {'Metric':<12} {'Value':>12}")
    print("-" * 60)

    for name, res in results.items():
        metric = res["metric"]
        if metric == "perplexity":
            val_str = f"{res['perplexity']:.2f}"
            detail = f"({res['num_tokens']:,} tokens)"
        else:
            val_str = f"{res['accuracy'] * 100:.2f}%"
            detail = f"({res['num_correct']}/{res['num_total']})"
        print(f"{name:<15} {metric:<12} {val_str:>12}  {detail}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run MOSAIC-GPT benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu, mps)")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Max examples per benchmark (0 = full eval)")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Which benchmarks to run (default: all). "
                             "Options: wikitext2 lambada hellaswag arc_easy")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: alongside checkpoint)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg, meta = load_model(args.checkpoint, device)
    print(model.summary())
    print(f"Checkpoint from step {meta['step']}")
    print(f"Device: {device}")
    if args.max_examples > 0:
        print(f"Max examples per benchmark: {args.max_examples}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(cfg.training.tokenizer)

    t0 = time.time()
    results = run_all(
        model=model,
        device=device,
        tokenizer=tokenizer,
        max_examples=args.max_examples,
        benchmarks=args.benchmarks,
    )
    elapsed = time.time() - t0

    print_results_table(results, meta)
    print(f"Total evaluation time: {elapsed:.1f}s")

    output_path = args.output
    if output_path is None:
        ckpt_dir = Path(args.checkpoint).parent
        output_path = str(ckpt_dir / "benchmark_results.json")

    serializable = {
        "step": meta["step"],
        "device": str(device),
        "max_examples": args.max_examples,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
