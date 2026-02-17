"""MOSAIC-GPT training infrastructure manager.

Core orchestrator library for managing Vast.ai training instances.
Used by the monitor daemon and agent team. Importable as a module
or runnable as a CLI.

Usage (CLI):
    python scripts/train_manager.py status
    python scripts/train_manager.py health 12345
    python scripts/train_manager.py search --gpus 4 --gpu-ram 20 --max-price 2.0
    python scripts/train_manager.py provision mosaic_moe 98765
    python scripts/train_manager.py destroy 12345
    python scripts/train_manager.py label 12345 "my-label"
    python scripts/train_manager.py cleanup 12345 mosaic_moe
    python scripts/train_manager.py restart 12345 mosaic_moe
    python scripts/train_manager.py journal --last 20
    python scripts/train_manager.py log --type restart --model mosaic_moe --message "Restarted after stall"

Usage (library):
    from scripts.train_manager import TrainManager
    tm = TrainManager()
    instances = tm.list_instances()
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

VASTAI_API_BASE = "https://console.vast.ai/api/v0"
DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
HF_CHECKPOINT_REPO = "lpkphd/mosaic-gpt-checkpoints"
GIT_REPO = "https://github.com/lpkphd/mosaic-gpt.git"

_DEFAULT_VASTAI_KEY = ""  # Set VASTAI_API_KEY env var
_DEFAULT_HF_TOKEN = ""   # Set HF_TOKEN env var

STALL_THRESHOLD_S = 600  # 10 min without heartbeat update = stalled


def _normalize_gpu_name(name: str) -> str:
    """Normalize GPU name from Vast.ai (e.g. 'RTX 3090' -> 'RTX_3090')."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ssh_client(host, port, timeout=15):
    """Create a connected paramiko SSHClient. Caller must close it."""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=int(port), username="root",
                   timeout=timeout, allow_agent=True, look_for_keys=True)
    return client


class TrainManager:
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent
        self.project_root = Path(project_root)
        self.api_key = os.environ.get("VASTAI_API_KEY", _DEFAULT_VASTAI_KEY)
        self.hf_token = os.environ.get("HF_TOKEN", _DEFAULT_HF_TOKEN)

        profiles_path = self.project_root / "scripts" / "instance_profiles.yaml"
        with open(profiles_path) as f:
            self.profiles = yaml.safe_load(f)

        self.journal_path = self.project_root / "scripts" / "journal.jsonl"
        self.status_path = self.project_root / "scripts" / "status.json"

    # ------------------------------------------------------------------ #
    # Vast.ai API helpers
    # ------------------------------------------------------------------ #

    def _api_url(self, path: str) -> str:
        sep = "&" if "?" in path else "?"
        return f"{VASTAI_API_BASE}/{path.lstrip('/')}{sep}api_key={self.api_key}"

    def _get(self, path: str, **kwargs) -> requests.Response:
        return requests.get(self._api_url(path), timeout=30, **kwargs)

    def _put(self, path: str, json_body=None) -> requests.Response:
        return requests.put(self._api_url(path), json=json_body, timeout=30)

    def _delete(self, path: str) -> requests.Response:
        return requests.delete(self._api_url(path), timeout=30)

    # ------------------------------------------------------------------ #
    # Instance management
    # ------------------------------------------------------------------ #

    def list_instances(self) -> list[dict]:
        resp = self._get("instances/")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("instances", [])
        return data

    def get_instance(self, instance_id) -> dict:
        for inst in self.list_instances():
            if inst.get("id") == int(instance_id):
                return inst
        return {}

    def destroy_instance(self, instance_id) -> bool:
        resp = self._delete(f"instances/{instance_id}/")
        self.log_event("destroy", "", f"Destroyed instance {instance_id}",
                       {"instance_id": int(instance_id), "status_code": resp.status_code})
        return resp.status_code in (200, 204)

    def label_instance(self, instance_id, label: str) -> bool:
        resp = self._put(f"instances/{instance_id}/", json_body={"label": label})
        return resp.status_code in (200, 204)

    def search_offers(self, num_gpus=4, gpu_ram_min=20, max_price=2.0,
                      gpu_name=None) -> list[dict]:
        query = {
            "num_gpus": {"gte": num_gpus, "lte": num_gpus},
            "gpu_ram": {"gte": gpu_ram_min},
            "rentable": {"eq": True},
            "dph_total": {"lte": max_price},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
        }
        if gpu_name:
            query["gpu_name"] = {"eq": gpu_name}
        encoded = urllib.parse.quote(json.dumps(query))
        resp = self._get(f"bundles/?q={encoded}")
        resp.raise_for_status()
        data = resp.json()
        offers = data.get("offers", data) if isinstance(data, dict) else data
        return sorted(offers, key=lambda o: o.get("dph_total", 999))

    def provision_instance(self, offer_id, model_name: str,
                           onstart_script: str | None = None) -> dict:
        model_cfg = self.get_model_config(model_name)
        offer = None
        try:
            offers = self.search_offers(num_gpus=1, gpu_ram_min=1, max_price=99)
            for o in offers:
                if o.get("id") == int(offer_id):
                    offer = o
                    break
        except Exception:
            pass

        if offer:
            gpu_name_raw = offer.get("gpu_name", "unknown")
            num_gpus = offer.get("num_gpus", 1)
        else:
            gpu_name_raw = "unknown"
            num_gpus = 1

        if onstart_script is None:
            onstart_script = self.generate_onstart(model_name, gpu_name_raw, num_gpus)

        body = {
            "client_id": "me",
            "image": DOCKER_IMAGE,
            "disk": 50,
            "label": f"{model_name}-auto",
            "onstart": onstart_script,
        }
        resp = self._put(f"asks/{offer_id}/", json_body=body)
        resp.raise_for_status()
        result = resp.json()
        self.log_event("provision", model_name,
                       f"Provisioned offer {offer_id}",
                       {"offer_id": int(offer_id), "result": result})
        return result

    # ------------------------------------------------------------------ #
    # SSH helpers
    # ------------------------------------------------------------------ #

    def _get_ssh_info(self, instance: dict) -> tuple[str, int]:
        host = instance.get("ssh_host", "")
        port = instance.get("ssh_port", 22)
        return host, int(port)

    def ssh_exec(self, instance: dict, command: str,
                 timeout: int = 30) -> tuple[str, str, int]:
        host, port = self._get_ssh_info(instance)
        client = _ssh_client(host, port, timeout=timeout)
        try:
            _, stdout, stderr = client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            return stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace"), exit_code
        finally:
            client.close()

    def ssh_read_file(self, instance: dict, remote_path: str) -> str:
        stdout, _, code = self.ssh_exec(instance, f"cat {remote_path}")
        if code != 0:
            return ""
        return stdout

    # ------------------------------------------------------------------ #
    # Health checks
    # ------------------------------------------------------------------ #

    def check_health(self, instance: dict) -> dict:
        iid = instance.get("id")
        label = instance.get("label", "")
        status_str = instance.get("actual_status", "unknown")
        cost_per_hr = instance.get("dph_total", 0)
        start_ts = instance.get("start_date", 0)

        report = {
            "instance_id": iid,
            "model": label,
            "status": "unreachable",
            "step": None,
            "max_steps": None,
            "loss": None,
            "ppl": None,
            "tok_s": None,
            "gpu_util": None,
            "disk_pct": None,
            "process_alive": False,
            "heartbeat_age_s": None,
            "cost_per_hr": cost_per_hr,
            "uptime_hrs": 0.0,
            "cost_total": 0.0,
            "eta_hrs": None,
            "pct_complete": None,
        }

        if start_ts:
            uptime_s = time.time() - start_ts
            report["uptime_hrs"] = round(uptime_s / 3600, 2)
            report["cost_total"] = round(cost_per_hr * uptime_s / 3600, 2)

        if status_str != "running":
            report["status"] = status_str
            return report

        try:
            host, port = self._get_ssh_info(instance)
            client = _ssh_client(host, port)
        except Exception:
            report["status"] = "unreachable"
            return report

        try:
            # Determine run_dir from label
            run_dir_name = self._label_to_run_dir(label)

            # Read heartbeat
            hb_path = f"/workspace/mosaic-gpt/runs/{run_dir_name}/heartbeat.json"
            _, hb_out, _ = client.exec_command(f"cat {hb_path} 2>/dev/null")
            hb_raw = hb_out.read().decode(errors="replace").strip()
            heartbeat = {}
            if hb_raw:
                try:
                    heartbeat = json.loads(hb_raw)
                except json.JSONDecodeError:
                    pass

            if heartbeat:
                hb_time = heartbeat.get("timestamp", 0)
                age = time.time() - hb_time if hb_time else 9999
                report["heartbeat_age_s"] = round(age, 1)
                report["step"] = heartbeat.get("step")
                report["max_steps"] = heartbeat.get("max_steps")
                report["loss"] = heartbeat.get("loss")
                report["ppl"] = heartbeat.get("ppl")
                report["tok_s"] = heartbeat.get("tok_s")

                if report["step"] and report["max_steps"] and report["max_steps"] > 0:
                    pct = report["step"] / report["max_steps"] * 100
                    report["pct_complete"] = round(pct, 1)
                    tok_s = report["tok_s"]
                    if tok_s and tok_s > 0 and report["step"] > 0:
                        remaining_steps = report["max_steps"] - report["step"]
                        elapsed_s = report["uptime_hrs"] * 3600
                        if elapsed_s > 0:
                            step_s = elapsed_s / report["step"]
                            report["eta_hrs"] = round(remaining_steps * step_s / 3600, 1)

            # Check process
            _, proc_out, _ = client.exec_command(
                "pgrep -f 'experiments/train.py' > /dev/null 2>&1 && echo alive || echo dead")
            proc_status = proc_out.read().decode().strip()
            report["process_alive"] = proc_status == "alive"

            # GPU utilization
            _, gpu_out, _ = client.exec_command(
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null")
            gpu_lines = gpu_out.read().decode().strip().split("\n")
            gpu_vals = []
            for line in gpu_lines:
                line = line.strip()
                if line.isdigit():
                    gpu_vals.append(int(line))
            if gpu_vals:
                report["gpu_util"] = round(sum(gpu_vals) / len(gpu_vals), 1)

            # Disk usage
            _, disk_out, _ = client.exec_command(
                "df /workspace 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%'")
            disk_str = disk_out.read().decode().strip()
            if disk_str.isdigit():
                report["disk_pct"] = int(disk_str)

            # Determine overall status
            if not report["process_alive"]:
                report["status"] = "crashed"
            elif report["heartbeat_age_s"] is not None and report["heartbeat_age_s"] > STALL_THRESHOLD_S:
                report["status"] = "stalled"
            else:
                report["status"] = "healthy"

        except Exception as e:
            report["status"] = "unreachable"
            report["_error"] = str(e)
        finally:
            client.close()

        return report

    def _label_to_run_dir(self, label: str) -> str:
        """Best-effort mapping from instance label to run directory name."""
        for model_name, cfg in self.profiles.items():
            run_dir = cfg.get("run_dir", "")
            dir_name = run_dir.split("/")[-1] if "/" in run_dir else run_dir
            if model_name in label or dir_name in label:
                return dir_name
        # Fallback: strip -auto suffix and try as run dir
        base = label.replace("-auto", "").replace("_auto", "")
        for model_name, cfg in self.profiles.items():
            if base in model_name or model_name in base:
                run_dir = cfg.get("run_dir", "")
                return run_dir.split("/")[-1] if "/" in run_dir else run_dir
        return label

    # ------------------------------------------------------------------ #
    # Disk management
    # ------------------------------------------------------------------ #

    def cleanup_disk(self, instance: dict, model_name: str) -> dict:
        model_cfg = self.get_model_config(model_name)
        run_dir = model_cfg.get("run_dir", "")

        commands = [
            # Remove old step_*.pt checkpoints (keep latest.pt and best.pt)
            f"find /workspace/mosaic-gpt/{run_dir} -name 'step_*.pt' -mmin +60 -delete 2>/dev/null",
            # Clear pip cache
            "rm -rf /root/.cache/pip 2>/dev/null",
            # Clear HF download cache (not model cache)
            "rm -rf /tmp/hf_cache/downloads 2>/dev/null",
            # Clear torch compile cache
            "rm -rf /tmp/torch_compile_* 2>/dev/null",
        ]
        combined = " && ".join(commands)

        # Get disk before
        before_out, _, _ = self.ssh_exec(instance,
            "df /workspace 2>/dev/null | tail -1 | awk '{print $4}'")
        before_kb = int(before_out.strip()) if before_out.strip().isdigit() else 0

        self.ssh_exec(instance, combined, timeout=60)

        # Get disk after
        after_out, _, _ = self.ssh_exec(instance,
            "df /workspace 2>/dev/null | tail -1 | awk '{print $4}'")
        after_kb = int(after_out.strip()) if after_out.strip().isdigit() else 0

        freed_kb = max(0, after_kb - before_kb)
        result = {
            "freed_bytes": freed_kb * 1024,
            "freed_mb": round(freed_kb / 1024, 1),
            "disk_available_mb": round(after_kb / 1024, 1),
        }
        self.log_event("cleanup", model_name,
                       f"Freed {result['freed_mb']} MB on instance {instance.get('id')}",
                       result)
        return result

    # ------------------------------------------------------------------ #
    # Training control
    # ------------------------------------------------------------------ #

    def start_training(self, instance: dict, model_name: str) -> bool:
        model_cfg = self.get_model_config(model_name)
        gpu_name_raw = instance.get("gpu_name", "unknown")
        num_gpus = instance.get("num_gpus", 1)
        train_cfg = self.get_training_config(model_name, gpu_name_raw, num_gpus)

        config_file = model_cfg["config"]
        run_dir = model_cfg["run_dir"]
        batch_size = train_cfg["batch_size"]
        grad_accum = train_cfg["grad_accum"]
        nproc = train_cfg["nproc"]

        if nproc > 1:
            launch = f"torchrun --nproc_per_node={nproc}"
        else:
            launch = "python3 -u"

        cmd = (
            f"cd /workspace/mosaic-gpt && git pull -q 2>/dev/null; "
            f"export HF_TOKEN={self.hf_token} HF_HOME=/tmp/hf_cache; "
            f"rm -f /tmp/mosaic_worker/latest.pt /tmp/mosaic_worker/best.pt 2>/dev/null; "
            f"nohup {launch} experiments/train.py "
            f"--config {config_file} "
            f"--run-dir {run_dir} "
            f"--batch-size {batch_size} --grad-accum {grad_accum} "
            f"--workers 4 --checkpoint-repo {HF_CHECKPOINT_REPO} "
            f"--plateau-patience 10 --plateau-min-delta 0.01 "
            f"> /workspace/training.log 2>&1 &"
        )
        _, stderr, code = self.ssh_exec(instance, cmd, timeout=15)
        success = code == 0
        self.log_event("start", model_name,
                       f"Started training on instance {instance.get('id')}",
                       {"instance_id": instance.get("id"), "nproc": nproc,
                        "batch_size": batch_size, "grad_accum": grad_accum,
                        "success": success})
        return success

    def stop_training(self, instance: dict) -> bool:
        cmd = "killall -9 torchrun 2>/dev/null; sleep 1; killall -9 python3 2>/dev/null"
        _, _, code = self.ssh_exec(instance, cmd, timeout=10)
        self.log_event("stop", "", f"Stopped training on instance {instance.get('id')}",
                       {"instance_id": instance.get("id")})
        return True

    def restart_training(self, instance: dict, model_name: str) -> bool:
        self.stop_training(instance)
        time.sleep(2)
        return self.start_training(instance, model_name)

    # ------------------------------------------------------------------ #
    # Checkpoint management
    # ------------------------------------------------------------------ #

    def verify_checkpoint(self, instance: dict, model_name: str) -> dict:
        model_cfg = self.get_model_config(model_name)
        run_dir = model_cfg["run_dir"]
        ckpt_path = f"/workspace/mosaic-gpt/{run_dir}/latest.pt"

        stdout, _, code = self.ssh_exec(instance,
            f"python3 -c \""
            f"import torch, json; "
            f"ckpt = torch.load('{ckpt_path}', map_location='cpu', weights_only=False); "
            f"print(json.dumps({{'step': ckpt.get('step', -1), 'exists': True}}))\" 2>/dev/null")

        if code == 0 and stdout.strip():
            try:
                info = json.loads(stdout.strip())
                info["path"] = ckpt_path
                return info
            except json.JSONDecodeError:
                pass

        return {"exists": False, "step": None, "path": ckpt_path}

    # ------------------------------------------------------------------ #
    # Journal
    # ------------------------------------------------------------------ #

    def log_event(self, event_type: str, model: str, message: str,
                  details: dict | None = None):
        entry = {
            "timestamp": _now_iso(),
            "type": event_type,
            "model": model,
            "message": message,
        }
        if details:
            entry["details"] = details
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent_events(self, n: int = 20) -> list[dict]:
        if not self.journal_path.exists():
            return []
        lines = self.journal_path.read_text().strip().split("\n")
        events = []
        for line in lines[-n:]:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events

    def search_events(self, pattern: str) -> list[dict]:
        if not self.journal_path.exists():
            return []
        regex = re.compile(pattern, re.IGNORECASE)
        matches = []
        for line in self.journal_path.read_text().strip().split("\n"):
            if line.strip() and regex.search(line):
                try:
                    matches.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return matches

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def write_status(self, health_reports: list[dict]):
        status = {
            "updated": _now_iso(),
            "timestamp": _now_iso(),
            "instances": health_reports,
            "total_cost_hr": sum(
                r.get("cost_per_hr") or 0 for r in health_reports
            ),
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

    def read_status(self) -> dict:
        if not self.status_path.exists():
            return {"updated": None, "instances": []}
        with open(self.status_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    # Config lookups
    # ------------------------------------------------------------------ #

    def get_training_config(self, model_name: str, gpu_name: str,
                            num_gpus: int) -> dict:
        model_cfg = self.profiles.get(model_name)
        if not model_cfg:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(self.profiles.keys())}")

        normalized = _normalize_gpu_name(gpu_name)
        profile_key = f"{normalized}_{num_gpus}x"
        profiles = model_cfg.get("profiles", {})

        if profile_key in profiles:
            return profiles[profile_key]

        # Try to find a profile with matching GPU count
        for key, cfg in profiles.items():
            if key.endswith(f"_{num_gpus}x"):
                return cfg

        # Fallback: pick the first profile and adjust grad_accum
        if profiles:
            first_key = next(iter(profiles))
            base = dict(profiles[first_key])
            base_nproc = base.get("nproc", 1)
            if base_nproc != num_gpus:
                ratio = base_nproc / num_gpus if num_gpus > 0 else 1
                base["grad_accum"] = max(1, int(base["grad_accum"] * ratio))
                base["nproc"] = num_gpus
            return base

        return {"batch_size": 8, "grad_accum": 16, "nproc": num_gpus}

    def get_model_config(self, model_name: str) -> dict:
        model_cfg = self.profiles.get(model_name)
        if not model_cfg:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(self.profiles.keys())}")
        return {
            "config": model_cfg["config"],
            "run_dir": model_cfg["run_dir"],
            "checkpoint_size_gb": model_cfg.get("checkpoint_size_gb", 2.0),
            "hf_path": model_cfg.get("hf_path", ""),
            "n_experts": model_cfg.get("n_experts"),
        }

    def generate_onstart(self, model_name: str, gpu_name: str,
                         num_gpus: int) -> str:
        model_cfg = self.get_model_config(model_name)
        train_cfg = self.get_training_config(model_name, gpu_name, num_gpus)

        config_file = model_cfg["config"]
        run_dir = model_cfg["run_dir"]
        hf_path = model_cfg["hf_path"]
        batch_size = train_cfg["batch_size"]
        grad_accum = train_cfg["grad_accum"]
        nproc = train_cfg["nproc"]
        run_dir_name = run_dir.split("/")[-1] if "/" in run_dir else run_dir

        if nproc > 1:
            launch = f"torchrun --nproc_per_node={nproc}"
        else:
            launch = "python3 -u"

        script = f"""#!/bin/bash
set -e
export HF_TOKEN={self.hf_token}
export HF_HOME=/tmp/hf_cache
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
pip install -q transformers datasets tiktoken huggingface_hub pyyaml

cd /workspace
if [ ! -d mosaic-gpt ]; then
    git clone {GIT_REPO}
fi
cd mosaic-gpt && git pull -q
mkdir -p {run_dir}

python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('{run_dir}', exist_ok=True)
for f in ['latest.pt', 'config.yaml']:
    try:
        p = hf_hub_download(
            repo_id='{HF_CHECKPOINT_REPO}',
            filename='{run_dir_name}/' + f,
            repo_type='model',
            token=os.environ.get('HF_TOKEN'))
        shutil.copy(p, '{run_dir}/' + f)
        print(f'Downloaded {{f}}')
    except Exception as e:
        print(f'Skip {{f}}: {{e}}')
"

rm -f /tmp/mosaic_worker/latest.pt /tmp/mosaic_worker/best.pt 2>/dev/null

nohup {launch} experiments/train.py \\
  --config {config_file} \\
  --run-dir {run_dir} \\
  --batch-size {batch_size} --grad-accum {grad_accum} \\
  --workers 4 --checkpoint-repo {HF_CHECKPOINT_REPO} \\
  --plateau-patience 10 --plateau-min-delta 0.01 \\
  > /workspace/training.log 2>&1 &
"""
        return script


# ====================================================================== #
# CLI
# ====================================================================== #

def _fmt_cost(v):
    return f"${v:.2f}" if v is not None else "—"


def _fmt_pct(v):
    return f"{v:.1f}%" if v is not None else "—"


def _fmt_val(v, suffix=""):
    if v is None:
        return "—"
    return f"{v}{suffix}"


def _print_health(report: dict):
    iid = report["instance_id"]
    model = report["model"]
    status = report["status"]

    status_colors = {
        "healthy": "\033[92m",   # green
        "stalled": "\033[93m",   # yellow
        "crashed": "\033[91m",   # red
        "unreachable": "\033[91m",
    }
    color = status_colors.get(status, "\033[0m")
    reset = "\033[0m"

    print(f"  Instance {iid} ({model})")
    print(f"    Status:     {color}{status}{reset}")
    print(f"    Step:       {_fmt_val(report['step'])}"
          f" / {_fmt_val(report['max_steps'])}"
          f"  ({_fmt_pct(report['pct_complete'])})")
    print(f"    Loss:       {_fmt_val(report['loss'])}"
          f"    PPL: {_fmt_val(report['ppl'])}"
          f"    tok/s: {_fmt_val(report['tok_s'])}")
    print(f"    GPU util:   {_fmt_val(report['gpu_util'], '%')}"
          f"    Disk: {_fmt_val(report['disk_pct'], '%')}"
          f"    Process: {'alive' if report['process_alive'] else 'dead'}")
    print(f"    Heartbeat:  {_fmt_val(report['heartbeat_age_s'], 's ago')}")
    print(f"    Cost:       {_fmt_cost(report['cost_per_hr'])}/hr"
          f"    Total: {_fmt_cost(report['cost_total'])}"
          f"    Uptime: {_fmt_val(report['uptime_hrs'], 'h')}"
          f"    ETA: {_fmt_val(report['eta_hrs'], 'h')}")
    print()


def cmd_status(tm: TrainManager, args):
    instances = tm.list_instances()
    if not instances:
        print("No active instances.")
        return

    reports = []
    print(f"\n  MOSAIC-GPT Training Status — {_now_iso()}\n")
    for inst in instances:
        report = tm.check_health(inst)
        reports.append(report)
        _print_health(report)

    tm.write_status(reports)


def cmd_health(tm: TrainManager, args):
    instance = tm.get_instance(args.instance_id)
    if not instance:
        print(f"Instance {args.instance_id} not found.")
        return
    report = tm.check_health(instance)
    _print_health(report)


def cmd_search(tm: TrainManager, args):
    offers = tm.search_offers(
        num_gpus=args.gpus,
        gpu_ram_min=args.gpu_ram,
        max_price=args.max_price,
        gpu_name=args.gpu_name,
    )
    if not offers:
        print("No offers found matching criteria.")
        return

    print(f"\n  {'ID':>8}  {'GPUs':>5}  {'GPU':>16}  {'RAM':>6}  {'$/hr':>6}  {'DLPerf':>7}  {'Reliability':>11}")
    print(f"  {'—'*8}  {'—'*5}  {'—'*16}  {'—'*6}  {'—'*6}  {'—'*7}  {'—'*11}")
    for o in offers[:20]:
        print(f"  {o.get('id', '?'):>8}"
              f"  {o.get('num_gpus', '?'):>5}"
              f"  {o.get('gpu_name', '?'):>16}"
              f"  {o.get('gpu_ram', 0):>5.0f}G"
              f"  {o.get('dph_total', 0):>6.3f}"
              f"  {o.get('dlperf', 0):>7.1f}"
              f"  {o.get('reliability2', 0):>10.3f}")
    print(f"\n  {len(offers)} total offers found.")


def cmd_provision(tm: TrainManager, args):
    result = tm.provision_instance(args.offer_id, args.model_name)
    new_id = result.get("new_contract", result.get("instance_id"))
    if new_id:
        print(f"Instance created: {new_id}")
        tm.label_instance(new_id, f"{args.model_name}-auto")
    else:
        print(f"Provision result: {json.dumps(result, indent=2)}")


def cmd_destroy(tm: TrainManager, args):
    ok = tm.destroy_instance(args.instance_id)
    print(f"Destroy {'succeeded' if ok else 'failed'} for instance {args.instance_id}")


def cmd_label(tm: TrainManager, args):
    ok = tm.label_instance(args.instance_id, args.label)
    print(f"Label {'set' if ok else 'failed'}: {args.label}")


def cmd_cleanup(tm: TrainManager, args):
    instance = tm.get_instance(args.instance_id)
    if not instance:
        print(f"Instance {args.instance_id} not found.")
        return
    result = tm.cleanup_disk(instance, args.model_name)
    print(f"Freed {result['freed_mb']} MB, {result['disk_available_mb']} MB available")


def cmd_restart(tm: TrainManager, args):
    instance = tm.get_instance(args.instance_id)
    if not instance:
        print(f"Instance {args.instance_id} not found.")
        return
    ok = tm.restart_training(instance, args.model_name)
    print(f"Restart {'succeeded' if ok else 'failed'} for {args.model_name} "
          f"on instance {args.instance_id}")


def cmd_journal(tm: TrainManager, args):
    events = tm.get_recent_events(n=args.last)
    if not events:
        print("No journal entries.")
        return
    for ev in events:
        ts = ev.get("timestamp", "?")[:19]
        etype = ev.get("type", "?")
        model = ev.get("model", "")
        msg = ev.get("message", "")
        print(f"  [{ts}] {etype:>10}  {model:>16}  {msg}")


def cmd_log(tm: TrainManager, args):
    tm.log_event(args.type, args.model, args.message)
    print(f"Logged: [{args.type}] {args.model}: {args.message}")


def main():
    parser = argparse.ArgumentParser(
        description="MOSAIC-GPT Training Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show health of all instances")

    # health
    p = sub.add_parser("health", help="Health check single instance")
    p.add_argument("instance_id", type=int)

    # search
    p = sub.add_parser("search", help="Search for available GPU offers")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--gpu-ram", type=int, default=20)
    p.add_argument("--max-price", type=float, default=2.0)
    p.add_argument("--gpu-name", type=str, default=None)

    # provision
    p = sub.add_parser("provision", help="Provision a new instance")
    p.add_argument("model_name", type=str)
    p.add_argument("offer_id", type=int)

    # destroy
    p = sub.add_parser("destroy", help="Destroy an instance")
    p.add_argument("instance_id", type=int)

    # label
    p = sub.add_parser("label", help="Label an instance")
    p.add_argument("instance_id", type=int)
    p.add_argument("label", type=str)

    # cleanup
    p = sub.add_parser("cleanup", help="Clean disk on instance")
    p.add_argument("instance_id", type=int)
    p.add_argument("model_name", type=str)

    # restart
    p = sub.add_parser("restart", help="Restart training on instance")
    p.add_argument("instance_id", type=int)
    p.add_argument("model_name", type=str)

    # journal
    p = sub.add_parser("journal", help="View recent journal events")
    p.add_argument("--last", type=int, default=20)

    # log
    p = sub.add_parser("log", help="Log an event to journal")
    p.add_argument("--type", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--message", type=str, required=True)

    args = parser.parse_args()
    tm = TrainManager()

    dispatch = {
        "status": cmd_status,
        "health": cmd_health,
        "search": cmd_search,
        "provision": cmd_provision,
        "destroy": cmd_destroy,
        "label": cmd_label,
        "cleanup": cmd_cleanup,
        "restart": cmd_restart,
        "journal": cmd_journal,
        "log": cmd_log,
    }
    dispatch[args.command](tm, args)


if __name__ == "__main__":
    main()
