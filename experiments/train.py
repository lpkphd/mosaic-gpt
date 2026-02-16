"""MOSAIC-GPT training entry point.

Usage:
    # Single GPU
    python experiments/train.py --config configs/mosaic_small.yaml --device cuda

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 experiments/train.py --config configs/mosaic_small.yaml

    # CPU (testing)
    python experiments/train.py --config configs/gpt2_reference.yaml --device cpu --max-steps 100
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic.config import MosaicConfig
from mosaic.model import MosaicGPT
from training.data import build_train_loader, build_eval_loader
from training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train MOSAIC-GPT")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu, mps)")
    parser.add_argument("--run-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--workers", type=int, default=2, help="Data loader workers")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (or 'auto')")
    args = parser.parse_args()

    cfg = MosaicConfig.from_yaml(args.config)

    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = rank == 0
    else:
        device = torch.device(args.device)
        is_main = True

    if is_main:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        run_dir = args.run_dir or f"runs/{config_name}"
        os.makedirs(run_dir, exist_ok=True)
        cfg.to_yaml(os.path.join(run_dir, "config.yaml"))

    model = MosaicGPT(cfg)

    if is_main:
        print(model.summary())

    if ddp:
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        model = model.to(device)
        raw_model = model

    train_loader = build_train_loader(cfg, num_workers=args.workers)
    eval_loader = build_eval_loader(cfg)

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = None  # trainer auto-detects latest.pt

    train(
        model=raw_model,
        cfg=cfg,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        run_dir=run_dir if is_main else "/tmp/mosaic_worker",
        use_wandb=args.wandb and is_main,
        grad_accum_steps=args.grad_accum,
        resume_from=resume_path,
    )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
