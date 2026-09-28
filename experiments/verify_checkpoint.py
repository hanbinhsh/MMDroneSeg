"""Check durable checkpoints and pretrained coverage without starting another experiment."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not all(torch.isfinite(v).all() for v in state["model"].values() if torch.is_floating_point(v)):
        raise ValueError("Nonfinite model state")
    optimizer = state.get("optimizer", {})
    steps = [float(v["step"]) for v in optimizer.get("state", {}).values() if "step" in v]
    # timm Lookahead wraps fast-optimizer state in an 'optimizer' field on some versions.
    if not steps:
        steps = [float(v["step"]) for v in optimizer.get("optimizer", {}).get("state", {}).values() if "step" in v]
    if optimizer and (not steps or max(steps) == 0):
        raise ValueError("No optimizer updates found; check AMP overflows")
    print(json.dumps({"checkpoint": str(args.checkpoint), "epoch": state["epoch"],
                      "all_parameters_finite": True, "max_optimizer_step": max(steps) if steps else None,
                      "scaler": state.get("scaler", {})}, indent=2))


if __name__ == "__main__":
    main()
