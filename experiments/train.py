import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ROOT, DATA_ROOT, CLASSES, COLORS, DroneData, dataset_spec
from .models import MODEL_NAMES, build_model, batch_needs, Objective
from .losses import Confusion
from .inference import sliding_logits


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    # Retain the previous complete checkpoint as an independent recovery option.
    # A power failure can defeat atomic rename alone through filesystem caching.
    if path.exists():
        path.replace(path.with_name(path.stem + ".prev" + path.suffix))
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Official grid_sample backward is not bitwise deterministic on CUDA.
    # Record this limitation instead of claiming strict deterministic LOGCAN training.


def worker_init(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def make_optimizer(model, name, horizon):
    if name == "afenet":
        lr, backbone_lr, decay = 6e-4, 6e-5, 0.01
    elif name == "logcan":
        lr, backbone_lr, decay = 1e-4, 1e-4, 1e-4
    elif name == "d2ls":
        lr, backbone_lr, decay = 1e-4, 1e-4, 0.01
    else:
        lr, backbone_lr, decay = 3e-4, 3e-5, 0.01
    backbone, head = [], []
    for key, p in model.named_parameters():
        if p.requires_grad:
            (backbone if "backbone" in key else head).append(p)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": lr}], weight_decay=decay)
    base = optimizer
    if name in ("d2ls", "afenet"):
        from timm.optim import Lookahead
        optimizer = Lookahead(base, alpha=0.5, k=5)
    if name == "afenet":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(base, T_0=15, T_mult=2)
    elif name == "d2ls":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=horizon, eta_min=1e-6)
    elif name == "logcan":
        scheduler = torch.optim.lr_scheduler.LambdaLR(base, lambda e: max(0, 1-e/horizon)**0.9)
    else:
        def schedule(e):
            return (e+1)/5 if e < 5 else 0.5 * (1+math.cos(math.pi * min(1, (e-5)/max(1,horizon-5))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(base, schedule)
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, device, amp=True, crop=512, output=None):
    model.eval()
    metric = Confusion(loader.dataset.num_classes)
    records = []
    started = time.perf_counter()
    for idx, batch in enumerate(loader):
        batch = to_device(batch, device)
        logits = sliding_logits(model, batch, crop=crop, stride=crop*3//4, amp=amp)
        pred = logits.argmax(1)
        metric.update(pred, batch["mask"])
        if output:
            per_image = Confusion(loader.dataset.num_classes)
            per_image.update(pred, batch["mask"])
            records.append({"name": batch["name"][0], **per_image.compute()})
            if idx < 6:
                from PIL import Image
                path = Path(output) / "predictions"
                path.mkdir(exist_ok=True)
                Image.fromarray(loader.dataset.colors[pred[0].cpu().numpy()]).save(path / batch["name"][0])
        if (idx+1) % 20 == 0:
            print(f"  validation {idx+1}/{len(loader)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = metric.compute()
    result["seconds"] = time.perf_counter() - started
    if output:
        atomic_json(Path(output) / "per_image.json", records)
    return result


def provenance():
    repositories = {}
    for name in ["D2LS", "AFENet", "rssegmentation"]:
        try:
            repositories[name] = subprocess.check_output(["git", "-C", str(ROOT/"external"/name), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            repositories[name] = "unavailable"
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "repositories": repositories,
            "adapter_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
            "determinism": "seeded; CUDA grid_sample backward in official LOGCAN may be nondeterministic"}


def run(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output or ROOT / "runs" / f"{args.model}_seed{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    spec = dataset_spec(args.data_root)
    config.update({"output": str(output), "data_root": str(args.data_root), "classes": spec["classes"],
                   "num_classes": len(spec["classes"]), "dataset_manifest_sha256": spec.get("manifest_sha256"),
                   "protocol": spec["protocol"], "provenance": provenance(),
                   "historical_baselines": "external results; not retrained; different training/evaluation protocol"})
    old_config_path = output / "config.json"
    if old_config_path.exists():
        previous = json.loads(old_config_path.read_text(encoding="utf-8"))
        for key in ["model", "seed", "crop", "horizon", "batch_size", "effective_batch", "no_pretrained", "no_amp", "data_root"]:
            if previous[key] != config[key]:
                raise ValueError(f"Resume config mismatch: {key}: {previous[key]} -> {config[key]}")
        if previous.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest changed since this experiment began")
        for key, default in (("class_aware_crop_prob", 0.0), ("rare_class_ids", [])):
            if previous.get(key, default) != config[key]:
                raise ValueError(f"Resume crop policy mismatch: {key}")
        if not args.resume and not args.evaluate:
            raise ValueError(f"Existing experiment {output}; use --resume or a new --output")
        atomic_json(output / "latest_invocation.json", config)
    else:
        atomic_json(old_config_path, config)
    needs = batch_needs(args.model)
    train_set = DroneData(args.data_root, "train", crop=args.crop, seed=args.seed, limit=args.train_limit,
                          class_aware_crop_prob=args.class_aware_crop_prob, rare_class_ids=args.rare_class_ids, **needs)
    val_set = DroneData(args.data_root, "val", crop=args.crop, seed=args.seed, limit=args.val_limit, **needs)
    generator = torch.Generator()
    kwargs = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=worker_init,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
    model = build_model(args.model, pretrained=not args.no_pretrained, num_classes=len(spec["classes"])).to(device)
    objective = Objective(args.model).to(device)
    optimizer, scheduler = make_optimizer(model, args.model, args.horizon)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch, best, bad = 0, -1.0, 0
    last_saved_epoch = 0
    last = output / "last.pt"
    if args.resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, bad = state["epoch"], state["best"], state["bad"]
        last_saved_epoch = start_epoch
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        # Remove log rows beyond the durable checkpoint if interrupted between saves.
        history = output / "metrics.jsonl"
        if history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            history.write_text("".join(json.dumps(r)+"\n" for r in rows if r["epoch"] <= start_epoch), encoding="utf-8")
        print(f"Resuming {args.model} from epoch {start_epoch}", flush=True)
    elif args.resume and any(output.glob("metrics.jsonl")):
        raise ValueError("Metrics exist but resumable last.pt is missing")
    if args.evaluate:
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        result = evaluate(model, val_loader, device, amp, args.crop, output)
        result.update({"epoch": checkpoint["epoch"], "model": args.model, "seed": args.seed, "protocol": config["protocol"],
                       "classes": spec["classes"], "num_classes": len(spec["classes"])})
        atomic_json(output / "evaluation.json", result)
        return result
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Training {args.model}, parameters={parameters:,}, device={device}, samples={len(train_set)}/{len(val_set)}", flush=True)
    accumulation = args.effective_batch // args.batch_size
    stopped = False
    for epoch in range(start_epoch, args.epochs):
        if epoch >= args.min_epochs and bad >= args.patience:
            stopped = True
            break
        model.train()
        train_set.epoch = epoch
        generator.manual_seed(args.seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        loss_sum, samples = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            window_start = (step // accumulation) * accumulation
            window_end = min(window_start + accumulation, len(train_loader))
            window_samples = min(len(train_set), window_end * args.batch_size) - window_start * args.batch_size
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(batch)
                loss = objective(outputs, batch["mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch+1}, step={step}: {loss.item()}")
            n = batch["image"].shape[0]
            scaler.scale(loss * (n/window_samples)).backward()
            if (step+1) % accumulation == 0 or step+1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * n
            samples += n
            if (step+1) % 20 == 0:
                print(f"  epoch {epoch+1} batch {step+1}/{len(train_loader)} loss={loss_sum/samples:.4f}", flush=True)
        train_seconds = time.perf_counter() - epoch_start
        result = evaluate(model, val_loader, device, amp, args.crop)
        scheduler.step()
        improved = result["miou"] > best
        if improved:
            best, bad = result["miou"], 0
            atomic_save(output / "best.pt", {"model": model.state_dict(), "epoch": epoch+1, "metrics": result, "config": config})
        else:
            bad += 1
        record = {"epoch": epoch+1, "train_loss": loss_sum/samples, "train_seconds": train_seconds,
                  **result, "best_miou": best, "lr": optimizer.param_groups[-1]["lr"]}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
            f.flush()
            os.fsync(f.fileno())
        stop_requested = (output / "STOP").exists()
        early_stop_due = epoch+1 >= args.min_epochs and bad >= args.patience
        # Optimizer checkpoints are large; always save at stage/stop boundaries,
        # but limit routine disk writes to the configured interval.
        if should_save_checkpoint(epoch+1, args.epochs, args.checkpoint_every, stop_requested or early_stop_due):
            atomic_save(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                              "epoch": epoch+1, "best": best, "bad": bad, "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            last_saved_epoch = epoch+1
        status = {"model": args.model, "seed": args.seed, "epoch": epoch+1, "target_epochs": args.epochs,
                  "best_miou": best, "last_miou": result["miou"], "parameters": parameters,
                  "peak_gpu_mb": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0,
                  "train_seconds": train_seconds, "validation_seconds": result["seconds"], "state": "running",
                  "checkpoint_epoch": last_saved_epoch, "checkpoint_every": args.checkpoint_every}
        atomic_json(output / "status.json", status)
        print(f"EPOCH {epoch+1}/{args.epochs} loss={loss_sum/samples:.4f} mIoU={result['miou']:.5f} best={best:.5f} train={train_seconds:.1f}s val={result['seconds']:.1f}s", flush=True)
        if stop_requested:
            stopped = True
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = evaluate(model, val_loader, device, amp, args.crop, output)
    result.update({"epoch": checkpoint["epoch"], "model": args.model, "seed": args.seed, "parameters": parameters,
                   "protocol": config["protocol"], "pretrained": not args.no_pretrained,
                   "classes": spec["classes"], "num_classes": len(spec["classes"]),
                   "train_limit": args.train_limit, "val_limit": args.val_limit})
    atomic_json(output / "evaluation.json", result)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"model": args.model, "epoch": start_epoch}
    status.update({"state": "stopped" if stopped else "stage_complete", "best_miou": result["miou"]})
    atomic_json(status_path, status)
    return result


def should_save_checkpoint(epoch, target, interval, stopping=False):
    return epoch % interval == 0 or epoch >= target or stopping


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=50, help="Hard cap on total epochs, including resumed epochs")
    p.add_argument("--checkpoint-every", type=int, default=5, help="Save full optimizer state every N epochs, plus final/stop boundaries")
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--effective-batch", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--class-aware-crop-prob", type=float, default=0.0)
    p.add_argument("--rare-class-ids", type=int, nargs="*", default=[])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.batch_size < 1 or args.effective_batch % args.batch_size:
        raise ValueError("effective-batch must be divisible by batch-size")
    if not 1 <= args.epochs <= min(args.horizon, args.max_epochs):
        raise ValueError("Require 1 <= epochs <= min(horizon, max-epochs)")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if args.crop % 128:
        raise ValueError("crop must be a multiple of 128 for official LOGCAN window partitions")
    run(args)
