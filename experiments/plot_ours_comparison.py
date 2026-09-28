"""Fixed qualitative examples selected before the refinement experiment finishes."""
import gc
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from .data import ROOT, CLASSES, COLORS, DroneData
from .models import build_model, batch_needs
from .inference import sliding_logits

NAMES = ["476.png", "478.png", "518.png", "569.png"]
MODELS = ["ours_no_text", "ours_sched50", "ours_refine"]


def main():
    torch.set_num_threads(4)
    predictions = {}
    data = DroneData(split="val", **batch_needs("ours_no_text"))
    indices = [data.names.index(name) for name in NAMES]
    for name in MODELS:
        path = ROOT / "runs" / f"{name}_seed42" / "best.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(name, pretrained=False).cuda().eval()
        model.load_state_dict(state["model"])
        predictions[name] = []
        for idx in indices:
            sample = data[idx]
            batch = {key: value.unsqueeze(0).cuda() for key, value in sample.items()
                     if key in ("image", "detail")}
            pred = sliding_logits(model, batch).argmax(1)[0].cpu().numpy()
            predictions[name].append(COLORS[pred])
        del model, state
        gc.collect()
        torch.cuda.empty_cache()
    fig, axes = plt.subplots(len(NAMES), 5, figsize=(17, 11))
    for i, idx in enumerate(indices):
        rgb, label = data.read(idx)
        images = [rgb, COLORS[label]] + [predictions[name][i] for name in MODELS]
        for ax, im in zip(axes[i], images):
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(NAMES[i])
    for ax, title in zip(axes[0], ["RGB", "Ground truth", "Previous no-text", "Faster LR decay", "Refinement + hard pixels"]):
        ax.set_title(title)
    fig.legend(handles=[Patch(color=c / 255, label=n) for c, n in zip(COLORS, CLASSES)],
               loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = ROOT / "runs/figures/ours_improvement_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
