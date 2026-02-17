#!/usr/bin/env python3
"""MOSAIC-GPT training monitor daemon.

Runs every 60 seconds, checking health of all training instances.
Writes status.json for the Streamlit dashboard.
Logs issues to journal.jsonl.
Sends macOS notifications for alerts.
"""

import sys
import os
import time
import json
import subprocess
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.train_manager import TrainManager

KNOWN_INSTANCES = {
    31515444: "gpt2_reference",
    31494488: "mosaic_dense",
    31523382: "mosaic_moe",
}

GPU_SIGNATURES = {
    (1, "RTX 4090"): "gpt2_reference",
    (4, "A100"):     "mosaic_dense",
    (8, "RTX 3090"): "mosaic_moe",
}

STALL_THRESHOLD = 5      # consecutive checks before alerting
DISK_WARN_PCT = 75
DISK_CRIT_PCT = 90
RECURRING_THRESHOLD = 3  # issue count before Ollama escalation


class Monitor:
    def __init__(self, interval=60):
        self.interval = interval
        self.manager = TrainManager()
        self.last_steps = {}
        self.stall_counts = {}
        self.issue_counts = {}

    def run(self):
        """Main loop."""
        print(f"[Monitor] Starting (interval={self.interval}s)")
        while True:
            try:
                self.check_cycle()
            except KeyboardInterrupt:
                print("\n[Monitor] Stopped.")
                break
            except Exception as e:
                print(f"[Monitor] Error in check cycle: {e}")
                self.manager.log_event("MONITOR_ERROR", "all", str(e))
            time.sleep(self.interval)

    def check_cycle(self):
        """One monitoring cycle: check instances, write status, print summary."""
        instances = self.manager.list_instances()

        reports = []
        for inst in instances:
            model_name = self._detect_model(inst)
            if not model_name:
                continue

            instance_id = inst.get("id")
            label = inst.get("label", "")
            status = inst.get("actual_status") or inst.get("status_msg", "")

            if "running" in str(status).lower():
                report = self.manager.check_health(inst)
                report["model_name"] = model_name
                report["label"] = label
                reports.append(report)
                self._handle_report(report, model_name)
            else:
                report = {
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "status": "down",
                    "label": label,
                    "raw_status": status,
                }
                reports.append(report)
                self._handle_down_instance(report, model_name)

        self.manager.write_status(reports)

        self._print_summary(reports)

    def _detect_model(self, instance):
        """Detect which model an instance is training.

        Checks, in order:
        1. Known instance ID mapping
        2. GPU count + type signature (disambiguates identically-labeled instances)
        3. Label keywords as fallback
        """
        iid = instance.get("id")
        if iid in KNOWN_INSTANCES:
            return KNOWN_INSTANCES[iid]

        num_gpus = instance.get("num_gpus", 0)
        gpu_name = instance.get("gpu_name", "")
        for (count, gpu_substr), model in GPU_SIGNATURES.items():
            if num_gpus == count and gpu_substr in gpu_name:
                return model

        label = (instance.get("label") or "").lower()
        if "gpt2" in label or "gpt-2" in label:
            return "gpt2_reference"
        if "dense" in label:
            return "mosaic_dense"
        if "moe" in label:
            return "mosaic_moe"

        return None

    def _handle_report(self, report, model_name):
        """Handle a running instance health report: stalls, disk, crashes."""
        status = report.get("status", "unknown")
        step = report.get("step") or 0
        disk_pct = report.get("disk_pct") or 0

        # -- Stall detection --
        last_step = self.last_steps.get(model_name, -1)
        if step > 0 and step == last_step:
            self.stall_counts[model_name] = self.stall_counts.get(model_name, 0) + 1
            if self.stall_counts[model_name] >= STALL_THRESHOLD:
                mins = self.stall_counts[model_name] * (self.interval // 60)
                self._alert(
                    f"{model_name} STALLED at step {step} for ~{mins}min",
                    "warning",
                )
                self.manager.log_event("STALL", model_name, f"Stalled at step {step}")
        else:
            self.stall_counts[model_name] = 0
        self.last_steps[model_name] = step

        # -- Disk pressure --
        if disk_pct > DISK_CRIT_PCT:
            self._alert(f"{model_name} DISK CRITICAL: {disk_pct:.0f}%", "critical")
            self.manager.log_event("DISK_CRITICAL", model_name, f"Disk at {disk_pct:.0f}%")
            try:
                self.manager.cleanup_disk(report, model_name)
                self.manager.log_event("AUTO_CLEANUP", model_name, "Triggered disk cleanup")
            except Exception as e:
                self.manager.log_event("CLEANUP_FAILED", model_name, str(e))
        elif disk_pct > DISK_WARN_PCT:
            self._alert(f"{model_name} disk warning: {disk_pct:.0f}%", "warning")

        # -- Crash detection --
        if status == "crashed":
            self._alert(f"{model_name} CRASHED", "critical")
            self.manager.log_event("CRASH", model_name, "Training process not found")
            self._track_recurring("crash:" + model_name)
            try:
                self.manager.restart_training(report, model_name)
                self.manager.log_event("AUTO_RESTART", model_name, "Restarted after crash")
            except Exception as e:
                self.manager.log_event("RESTART_FAILED", model_name, str(e))
                self._escalate(model_name, f"Restart failed: {e}")

    def _handle_down_instance(self, report, model_name):
        """Handle an instance that is not running."""
        raw_status = report.get("raw_status", "")
        self._alert(f"{model_name} instance DOWN: {raw_status}", "critical")
        self.manager.log_event("INSTANCE_DOWN", model_name, f"Status: {raw_status}")
        self._track_recurring("down:" + model_name)

    def _track_recurring(self, issue_key):
        """Track recurring issues; escalate to Ollama after threshold."""
        self.issue_counts[issue_key] = self.issue_counts.get(issue_key, 0) + 1
        if self.issue_counts[issue_key] >= RECURRING_THRESHOLD:
            self._escalate_with_ollama(issue_key)

    def _escalate(self, model_name, reason):
        """Escalate via Claude Code for complex fixes."""
        prompt = (
            f"MOSAIC-GPT training issue for {model_name}: {reason}. "
            "Check journal for history. Fix the issue."
        )
        self.manager.log_event("ESCALATION", model_name, f"Escalating: {reason}")
        try:
            subprocess.Popen(
                ["claude", "-p", prompt],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("[Monitor] Claude Code not available for escalation")

    def _escalate_with_ollama(self, issue_key):
        """Query local Ollama for advice on a recurring issue."""
        recent = self.manager.get_recent_events(20)
        context = json.dumps(recent, indent=2)
        prompt = (
            f"This training issue has occurred {self.issue_counts[issue_key]}+ times:\n"
            f"{issue_key}\n\nRecent journal:\n{context}\n\n"
            'What is the likely root cause and fix? Reply as JSON: '
            '{"action": "restart"|"escalate"|"ignore", "reason": "..."}'
        )
        try:
            result = subprocess.run(
                ["ollama", "run", "llama3.2", prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                advice = result.stdout.strip()
                self.manager.log_event(
                    "OLLAMA_ADVICE", "all", advice, {"issue_key": issue_key}
                )
                print(f"[Ollama] Advice for {issue_key}: {advice[:200]}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _alert(self, message, severity="info"):
        """Print alert and send macOS notification for warnings/criticals."""
        print(f"[Monitor] [{severity.upper()}] {message}")
        if severity in ("warning", "critical"):
            try:
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        f'display notification "{message}" '
                        f'with title "MOSAIC-GPT Monitor" sound name "Basso"',
                    ],
                    timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    def _print_summary(self, reports):
        """Print a clean one-line-per-model status table."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        total_cost = sum(r.get("cost_per_hr", 0) for r in reports)
        n_running = sum(1 for r in reports if r.get("status") != "down")

        print(f"\n[{ts}] --- Monitor Check --- "
              f"({n_running}/{len(reports)} running, ${total_cost:.2f}/hr)")
        print(f"  {'MODEL':<20s} {'STATUS':<12s} {'STEP':>10s} "
              f"{'PPL':>8s} {'TOK/S':>8s} {'DISK':>6s} {'PROGRESS':>8s}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

        for r in reports:
            model = r.get("model_name", "?")
            status = r.get("status", "?")
            step = r.get("step", "-")
            max_steps = r.get("max_steps", "-")
            ppl = r.get("ppl", "-")
            tok_s = r.get("tok_s", "-")
            disk = r.get("disk_pct", "-")
            pct = r.get("pct_complete")

            step_str = f"{step}/{max_steps}" if max_steps != "-" else str(step)
            ppl_str = f"{ppl:.1f}" if isinstance(ppl, (int, float)) else str(ppl)
            tok_str = f"{tok_s:.0f}" if isinstance(tok_s, (int, float)) else str(tok_s)
            disk_str = f"{disk:.0f}%" if isinstance(disk, (int, float)) else str(disk)
            pct_str = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "-"

            print(f"  {model:<20s} {status:<12s} {step_str:>10s} "
                  f"{ppl_str:>8s} {tok_str:>8s} {disk_str:>6s} {pct_str:>8s}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MOSAIC-GPT Training Monitor")
    parser.add_argument(
        "--interval", type=int, default=60, help="Check interval in seconds"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run one check cycle and exit"
    )
    args = parser.parse_args()

    monitor = Monitor(interval=args.interval)
    if args.once:
        monitor.check_cycle()
    else:
        monitor.run()


if __name__ == "__main__":
    main()
