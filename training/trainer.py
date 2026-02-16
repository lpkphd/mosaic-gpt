"""Training loop for MOSAIC-GPT.

Handles:
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Cosine LR schedule with warmup
- MoE auxiliary loss
- Periodic evaluation and checkpointing
- Wandb logging (optional)
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from mosaic.config import MosaicConfig
from mosaic.model import MosaicGPT


def get_lr(step: int, cfg: MosaicConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < cfg.training.warmup_steps:
        return cfg.training.lr * step / cfg.training.warmup_steps
    if step >= cfg.training.max_steps:
        return cfg.training.min_lr

    progress = (step - cfg.training.warmup_steps) / (cfg.training.max_steps - cfg.training.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.training.min_lr + (cfg.training.lr - cfg.training.min_lr) * cosine


@torch.no_grad()
def evaluate(model: MosaicGPT, eval_loader, device: torch.device, max_steps: int = 100) -> dict:
    """Compute perplexity on evaluation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    steps = 0

    for batch in eval_loader:
        if steps >= max_steps:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        steps += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 20.0))
    model.train()
    return {"loss": avg_loss, "perplexity": ppl, "tokens": total_tokens}


def train(
    model: MosaicGPT,
    cfg: MosaicConfig,
    train_loader,
    eval_loader,
    device: torch.device,
    run_dir: str = "runs/mosaic",
    use_wandb: bool = False,
    grad_accum_steps: int = 1,
):
    """Main training loop."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(cfg.training.beta1, cfg.training.beta2),
        weight_decay=cfg.training.weight_decay,
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    if use_wandb:
        import wandb
        wandb.init(project="mosaic-gpt", config=vars(cfg))

    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    tokens_processed = 0

    print(f"\nTraining MOSAIC-GPT")
    print(model.summary())
    print(f"Device: {device}")
    print(f"Grad accumulation: {grad_accum_steps}")
    print(f"Effective batch size: {cfg.training.batch_size * grad_accum_steps}")
    print(f"Max steps: {cfg.training.max_steps}\n")

    while step < cfg.training.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_aux = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                aux_loss = model.get_aux_loss()
                total_loss = (loss + aux_loss) / grad_accum_steps

            scaler.scale(total_loss).backward()
            accum_loss += loss.item() / grad_accum_steps
            if isinstance(aux_loss, torch.Tensor):
                accum_aux += aux_loss.item() / grad_accum_steps
            tokens_processed += y.numel()

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # Logging
        if step % cfg.training.log_interval == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed
            ppl = math.exp(min(accum_loss, 20.0))
            msg = (
                f"step {step:>6d}/{cfg.training.max_steps} | "
                f"loss {accum_loss:.4f} | ppl {ppl:.1f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"{tok_per_sec:.0f} tok/s"
            )
            if accum_aux > 0:
                msg += f" | aux {accum_aux:.4f}"
            print(msg)

            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": accum_loss,
                    "train/ppl": ppl,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/aux_loss": accum_aux,
                }, step=step)

        # Evaluation
        if step % cfg.training.eval_interval == 0:
            metrics = evaluate(model, eval_loader, device, max_steps=cfg.training.eval_steps)
            print(f"  EVAL step {step}: loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.1f}")

            if use_wandb:
                import wandb
                wandb.log({
                    "eval/loss": metrics["loss"],
                    "eval/ppl": metrics["perplexity"],
                }, step=step)

            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
                save_checkpoint(model, optimizer, cfg, step, metrics, run_path / "best.pt")
                print(f"  New best! Saved to {run_path / 'best.pt'}")

        # Periodic save
        if step % cfg.training.save_interval == 0:
            save_checkpoint(model, optimizer, cfg, step, {}, run_path / f"step_{step}.pt")

    # Final save
    save_checkpoint(model, optimizer, cfg, step, {}, run_path / "final.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    if use_wandb:
        import wandb
        wandb.finish()

    return best_val_loss


def save_checkpoint(model, optimizer, cfg, step, metrics, path):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "metrics": metrics,
    }, path)
