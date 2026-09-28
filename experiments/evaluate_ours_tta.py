"""Optional four-flip inference study, kept separate from ordinary model metrics."""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from .data import ROOT, DroneData
from .models import build_model, batch_needs
from .inference import sliding_logits
from .losses import Confusion
from .train import atomic_json, seed_all, to_device


@torch.no_grad()
def four_flip_logits(model, batch, crop=512, amp=True):
    total, original = None, None
    for dims in [(), (-1,), (-2,), (-2, -1)]:
        flipped = {k: (v.flip(dims) if dims and k in ("image", "detail") else v)
                   for k, v in batch.items() if k in ("image", "detail", "text")}
        logits = sliding_logits(model, flipped, crop=crop, stride=crop*3//4, amp=amp)
        if dims:
            logits = logits.flip(dims)
        else:
            original = logits
        total = logits if total is None else total + logits
    return original, total / 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ours_no_text", choices=["ours_no_text", "ours_sched50", "ours_refine"])
    args = p.parse_args()
    torch.set_num_threads(4)
    seed_all(42)
    folder = ROOT / "runs" / f"{args.model}_seed42"
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_model(args.model, pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    data = DroneData(split="val", **batch_needs(args.model))
    baseline, ensemble = Confusion(), Confusion()
    started = time.perf_counter()
    for idx, batch in enumerate(DataLoader(data, batch_size=1)):
        batch = to_device(batch, torch.device("cuda"))
        original, averaged = four_flip_logits(model, batch, crop=config["crop"], amp=not config["no_amp"])
        baseline.update(original.argmax(1), batch["mask"])
        ensemble.update(averaged.argmax(1), batch["mask"])
        if (idx+1) % 20 == 0:
            print(f"TTA validation {idx+1}/{len(data)}", flush=True)
    torch.cuda.synchronize()
    result = {"model": args.model, "checkpoint_epoch": checkpoint["epoch"], "seed": 42,
              "ordinary_inference": baseline.compute(), "four_flip_tta": ensemble.compute(),
              "network_pass_multiplier": 4, "elapsed_seconds": time.perf_counter()-started,
              "protocol": "native_960x736_crop512_stride384_four_flip_logit_average",
              "note": "Additional inference only; not comparable to ordinary-inference results without disclosing TTA. Timing overlaps training and is not an FPS benchmark."}
    atomic_json(folder / "evaluation_tta4.json", result)
    print(json.dumps({"model": args.model, "ordinary_miou": result["ordinary_inference"]["miou"],
                      "tta_miou": result["four_flip_tta"]["miou"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
