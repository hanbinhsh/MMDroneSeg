"""Read-only error analysis of our saved model; never fits on validation masks."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ROOT, CLASSES, DroneData
from .models import build_model, batch_needs
from .inference import sliding_logits
from .losses import boundary_target
from .train import atomic_json, to_device


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    path = ROOT / "runs/ours_no_text_seed42/best.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model("ours_no_text", pretrained=False).cuda().eval()
    model.load_state_dict(state["model"])
    data = DroneData(split="val", **batch_needs("ours_no_text"))
    out = ROOT / "runs/ours_diagnostics"
    out.mkdir(exist_ok=True)
    errors, near_errors, pixels, edge_pixels = 0, 0, 0, 0
    records = []
    for batch in DataLoader(data, batch_size=1):
        batch = to_device(batch, torch.device("cuda"))
        with torch.no_grad():
            pred = sliding_logits(model, batch).argmax(1)
            edges, _ = boundary_target(batch["mask"])
            near = torch.nn.functional.max_pool2d(edges.unsqueeze(1), 11, 1, 5).squeeze(1).bool()
            wrong = pred != batch["mask"]
            errors += wrong.sum().item()
            near_errors += (wrong & near).sum().item()
            pixels += wrong.numel()
            edge_pixels += near.sum().item()
        label = batch["mask"][0].cpu().numpy()
        prediction = pred[0].cpu().numpy()
        obstacle_union = ((prediction == 0) | (label == 0)).sum()
        records.append({"name": batch["name"][0], "error_pixels": int(wrong.sum()),
                        "obstacle_pixels": int((label == 0).sum()),
                        "obstacle_iou": float(((prediction == 0) & (label == 0)).sum() / max(1, obstacle_union))})
    result = {"checkpoint": str(path), "epoch": state["epoch"], "errors": errors,
              "errors_within_5px_of_label_boundary": near_errors,
              "error_fraction_near_boundary": near_errors / max(1, errors),
              "pixel_fraction_near_boundary": edge_pixels / pixels,
              "difficult_obstacle_images": sorted([r for r in records if r["obstacle_pixels"] >= 5000],
                                                   key=lambda r: r["obstacle_iou"])[:8]}
    atomic_json(out / "error_analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
