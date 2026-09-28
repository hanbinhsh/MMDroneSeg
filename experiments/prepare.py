"""Download published backbone weights and audit/cache the existing dataset."""
import argparse
import hashlib
import json
import socket
import urllib.request
import time
import subprocess
from pathlib import Path

import torch
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
from .data import ROOT, DATA_ROOT, audit

WEIGHTS = ROOT / "weights"
URLS = {
    "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url,
    "d2ls": ConvNeXt_Base_Weights.IMAGENET1K_V1.url,
    "afenet": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth",
    "logcan": "https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_distill_450e.pth",
}


def weight_path(name):
    return WEIGHTS / URLS[name].rsplit("/", 1)[-1]


def download(names, proxy=None):
    socket.setdefaulttimeout(60)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in names:
        path = weight_path(name)
        if not path.exists():
            print(f"Downloading {name}: {URLS[name]}", flush=True)
            part = path.with_suffix(path.suffix + ".download")
            for attempt in range(3):
                try:
                    url = URLS[name] + (f"?attempt={attempt+1}" if attempt else "")
                    command = ["curl.exe", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "20",
                               "--max-time", "180", "--output", str(part), url]
                    if proxy:
                        command += ["--proxy", proxy]
                    subprocess.run(command, check=True, timeout=190)
                    part.replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    print(f"Retry {name} ({attempt+1}/3)", flush=True)
                    time.sleep(2)
        # Detect incomplete/unreadable files; never silently run pretrained experiments from scratch.
        torch.load(path, map_location="cpu", weights_only=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in ("convnext_tiny", "d2ls", "afenet"):
            expected = path.stem.rsplit("-", 1)[-1]
            if not digest.startswith(expected):
                raise ValueError(f"Official checkpoint SHA256 prefix mismatch: {path}")
        manifest[name] = {"url": URLS[name], "sha256": digest, "file": str(path)}
        print(f"Verified {name}: {digest}", flush=True)
        existing = WEIGHTS / "manifest.json"
        merged = json.loads(existing.read_text()) if existing.exists() else {}
        merged.update(manifest)
        existing.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="*", choices=list(URLS), default=[])
    p.add_argument("--audit", action="store_true")
    p.add_argument("--proxy", help="Explicit user/system HTTP proxy, e.g. http://127.0.0.1:7890")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = p.parse_args()
    if args.proxy:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})))
    if args.audit:
        result = audit(args.data_root)
        out = ROOT / "runs/data_audit.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "splits"}), flush=True)
    download(args.weights, proxy=args.proxy)


if __name__ == "__main__":
    main()
