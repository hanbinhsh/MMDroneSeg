"""Sequential single-GPU queue; no U-Net or DeepLabV3+ training jobs."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .data import ROOT
from .report import generate
from .train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["ours", "afenet", "logcan", "d2ls", "ours_rgb"])
    p.add_argument("--stages", nargs="+", type=int, default=[30, 50])
    p.add_argument("--max-epochs", type=int, default=50, help="Total epoch cap per model, including completed epochs")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--horizon", type=int, default=150, help="Learning-rate schedule horizon; use 50 for new 50-epoch studies")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--wait-for-queue", action="store_true", help="Wait for the current experiment queue to release its lock")
    args = p.parse_args()
    if args.max_epochs < 1 or any(stage < 1 for stage in args.stages):
        p.error("Epoch budgets must be positive")
    if args.checkpoint_every < 1:
        p.error("checkpoint-every must be positive")
    args.stages = sorted({min(stage, args.max_epochs) for stage in args.stages})
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    lock = runs / "queue.lock"
    if args.wait_for_queue and lock.exists():
        print("Waiting for the active queue to finish; no competing GPU training is started.", flush=True)
        while lock.exists():
            if (runs / "STOP_QUEUE").exists():
                return
            time.sleep(5)
    try:
        with lock.open("x") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError("Queue lock exists. Check its PID before removing a stale lock.")
    failures = []
    try:
        generate()
        for stage in args.stages:
            for model in args.models:
                if (runs / "STOP_QUEUE").exists():
                    atomic_json(runs/"queue_status.json", {"state": "stopped", "failures": failures})
                    return
                folder = runs / f"{model}_seed42"
                state_file = folder / "status.json"
                state = json.loads(state_file.read_text()) if state_file.exists() else {}
                if state.get("epoch", 0) >= stage or state.get("state") == "stopped":
                    continue
                command = [sys.executable, "-B", "-u", "-m", "experiments.train", "--model", model,
                           "--epochs", str(stage), "--max-epochs", str(args.max_epochs),
                           "--checkpoint-every", str(args.checkpoint_every),
                           "--horizon", str(args.horizon),
                           "--workers", str(args.workers), "--seed", "42"]
                if (folder / "config.json").exists():
                    command.append("--resume")
                atomic_json(runs/"queue_status.json", {"state": "running", "model": model, "target_epoch": stage,
                                                     "pid": os.getpid(), "failures": failures})
                with (runs / f"{model}_seed42.console.log").open("a", encoding="utf-8") as log:
                    log.write(f"\nCOMMAND {' '.join(command)}\n")
                    log.flush()
                    print(f"Starting {model} -> epoch {stage}", flush=True)
                    code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if code:
                    failures.append({"model": model, "stage": stage, "exit_code": code})
                    print(f"FAILED {model}, see console log", flush=True)
                generate()
        atomic_json(runs/"queue_status.json", {"state": "complete_with_failures" if failures else "complete", "failures": failures})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
