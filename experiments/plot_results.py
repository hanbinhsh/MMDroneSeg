"""Export validation curves and fixed examples from completed experiment stages."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from .data import ROOT, DATA_ROOT, CLASSES, COLORS


def main():
    runs = ROOT / "runs"
    folders = sorted(p.parent for p in runs.glob("*_seed42/evaluation.json"))
    folders = [p for p in folders if not any(
        json.loads((p / "config.json").read_text()).get(k)
        for k in ("no_pretrained", "train_limit", "val_limit"))]
    if not folders:
        raise RuntimeError("No completed full-data experiment stage to plot")
    out = runs / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for folder in folders:
        rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        name = folder.name.removesuffix("_seed42")
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["miou"] * 100 for r in rows], label=name)
        axes[1].plot(epochs, [r["train_loss"] for r in rows], label=name)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Validation mIoU (%)")
    axes[1].set_ylabel("Training objective (method-specific)")
    fig.savefig(out / "training_curves.png", dpi=180)
    fig.savefig(out / "training_curves.pdf")
    plt.close(fig)
    for folder in folders:
        paths = sorted((folder / "predictions").glob("*.png"))[:4]
        if not paths:
            continue
        fig, axes = plt.subplots(len(paths), 3, figsize=(12, len(paths) * 3.1), squeeze=False)
        for row, prediction in zip(axes, paths):
            for ax, path in zip(row, [DATA_ROOT / "val_original" / prediction.name,
                                     DATA_ROOT / "val_label" / prediction.name, prediction]):
                with Image.open(path) as im:
                    ax.imshow(im.copy())
                ax.set_xticks([])
                ax.set_yticks([])
            row[0].set_ylabel(prediction.name)
        for ax, title in zip(axes[0], ["RGB", "Ground truth", "Prediction"]):
            ax.set_title(title)
        fig.legend(handles=[Patch(color=c / 255, label=n) for c, n in zip(COLORS, CLASSES)],
                   loc="lower center", ncol=5)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / f"{folder.name}_examples.png", dpi=160)
        plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
