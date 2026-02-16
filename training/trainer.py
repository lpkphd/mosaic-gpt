"""Training loop for MOSAIC-GPT.

Handles:
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Cosine LR schedule with warmup
- MoE auxiliary loss
- Periodic evaluation and checkpointing
- Wandb logging (optional)
- Checkpoint backup to HuggingFace Hub (for spot instance safety)
"""

import os
import math
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path

from mosaic.config import MosaicConfig
from mosaic.model import MosaicGPT


def _upload_to_hf_async(local_path, repo_id, remote_path):
    """Upload a file to HuggingFace Hub in a background thread."""
    def _upload():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Backed up {remote_path} to hf://{repo_id}")
        except Exception as e:
            print(f"  Warning: backup failed for {remote_path}: {e}")
    thread = threading.Thread(target=_upload, daemon=True)
    thread.start()
    return thread


def _save_training_log(run_path, checkpoint_repo, step, metrics_history):
    """Save training metrics as JSON for easy plotting later."""
    import json
    log_path = run_path / "metrics.json"
    with open(log_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    if checkpoint_repo:
        _upload_to_hf_async(log_path, checkpoint_repo, f"{run_path.name}/metrics.json")


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
    resume_from: str = None,
    checkpoint_repo: str = None,
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

    best_val_loss = float("inf")
    step = 0

    # Resume from checkpoint
    if resume_from is None:
        latest = run_path / "latest.pt"
        if latest.exists():
            resume_from = str(latest)
            print(f"Auto-resuming from {resume_from}")
        elif checkpoint_repo:
            # Try downloading latest checkpoint from HuggingFace
            try:
                from huggingface_hub import hf_hub_download
                run_name = run_path.name
                dl_path = hf_hub_download(
                    repo_id=checkpoint_repo,
                    filename=f"{run_name}/latest.pt",
                    repo_type="model",
                )
                resume_from = dl_path
                print(f"Downloaded checkpoint from hf://{checkpoint_repo}/{run_name}/latest.pt")
            except Exception:
                print("No remote checkpoint found, starting fresh.")

    if resume_from is not None and os.path.exists(resume_from):
        step, best_val_loss = load_checkpoint(resume_from, model, optimizer, scaler, device)

    if use_wandb:
        import wandb
        wandb.init(project="mosaic-gpt", config=vars(cfg))

    train_iter = iter(train_loader)
    t0 = time.time()
    tokens_processed = 0
    metrics_history = {"train": [], "eval": []}

    print(f"\nTraining MOSAIC-GPT")
    print(model.summary())
    print(f"Device: {device}")
    print(f"Grad accumulation: {grad_accum_steps}")
    print(f"Effective batch size: {cfg.training.batch_size * grad_accum_steps}")
    print(f"Max steps: {cfg.training.max_steps}")
    if step > 0:
        print(f"Resuming from step: {step}")
    print()

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

            metrics_history["train"].append({
                "step": step, "loss": accum_loss, "ppl": ppl,
                "lr": lr, "grad_norm": float(grad_norm), "tok_s": tok_per_sec,
            })

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

            metrics_history["eval"].append({
                "step": step, "loss": metrics["loss"], "ppl": metrics["perplexity"],
            })

            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
                save_checkpoint(model, optimizer, scaler, cfg, step, metrics, best_val_loss, run_path / "best.pt")
                print(f"  New best! Saved to {run_path / 'best.pt'}")
                if checkpoint_repo:
                    _upload_to_hf_async(run_path / "best.pt", checkpoint_repo, f"{run_path.name}/best.pt")

        # Periodic save
        if step % cfg.training.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, cfg, step, {}, best_val_loss, run_path / f"step_{step}.pt")
            save_checkpoint(model, optimizer, scaler, cfg, step, {}, best_val_loss, run_path / "latest.pt")
            _save_training_log(run_path, checkpoint_repo, step, metrics_history)
            if checkpoint_repo:
                run_name = run_path.name
                _upload_to_hf_async(run_path / "latest.pt", checkpoint_repo, f"{run_name}/latest.pt")
                _upload_to_hf_async(run_path / f"step_{step}.pt", checkpoint_repo, f"{run_name}/step_{step}.pt")
                # Also back up the config
                cfg_path = run_path / "config.yaml"
                if cfg_path.exists():
                    _upload_to_hf_async(cfg_path, checkpoint_repo, f"{run_name}/config.yaml")

        # Auto-save latest every 250 steps (crash recovery)
        elif step % 250 == 0:
            save_checkpoint(model, optimizer, scaler, cfg, step, {}, best_val_loss, run_path / "latest.pt")
            if checkpoint_repo:
                run_name = run_path.name
                _upload_to_hf_async(run_path / "latest.pt", checkpoint_repo, f"{run_name}/latest.pt")

    # Final save
    save_checkpoint(model, optimizer, scaler, cfg, step, {}, best_val_loss, run_path / "final.pt")
    _save_training_log(run_path, None, step, metrics_history)  # Save locally
    if checkpoint_repo:
        run_name = run_path.name
        _upload_to_hf_async(run_path / "final.pt", checkpoint_repo, f"{run_name}/final.pt")
        _upload_to_hf_async(run_path / "best.pt", checkpoint_repo, f"{run_name}/best.pt")
        cfg_path = run_path / "config.yaml"
        if cfg_path.exists():
            _upload_to_hf_async(cfg_path, checkpoint_repo, f"{run_name}/config.yaml")
        t = _upload_to_hf_async(run_path / "metrics.json", checkpoint_repo, f"{run_name}/metrics.json")
        t.join()  # Wait for final uploads before exiting
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    if use_wandb:
        import wandb
        wandb.finish()

    return best_val_loss


def save_checkpoint(model, optimizer, scaler, cfg, step, metrics, best_val_loss, path):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step,
        "config": cfg,
        "metrics": metrics,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    step = ckpt["step"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"Resumed from checkpoint at step {step} (best val loss: {best_val_loss:.4f})")
    return step, best_val_loss
