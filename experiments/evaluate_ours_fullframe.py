"""A separate native-resolution, full-frame inference experiment for our model."""
import json
import time

import torch
from torch.utils.data import DataLoader

from .data import ROOT, DroneData
from .models import build_model, batch_needs
from .losses import Confusion
from .train import atomic_json, seed_all, to_device


def main():
    torch.set_num_threads(4)
    seed_all(42)
    folder = ROOT / "runs/ours_no_text_seed42"
    state = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model = build_model("ours_no_text", pretrained=False).cuda().eval()
    model.load_state_dict(state["model"])
    data = DroneData(split="val", **batch_needs("ours_no_text"))
    metric = Confusion()
    started = time.perf_counter()
    with torch.no_grad():
        for idx, batch in enumerate(DataLoader(data, batch_size=1)):
            batch = to_device(batch, torch.device("cuda"))
            with torch.autocast(device_type="cuda", enabled=True):
                output = model({k: batch[k] for k in ("image", "detail")})["logits"]
            assert output.shape[-2:] == batch["mask"].shape[-2:]
            metric.update(output.argmax(1), batch["mask"])
            if (idx+1) % 20 == 0:
                print(f"Full-frame validation {idx+1}/{len(data)}", flush=True)
    torch.cuda.synchronize()
    result = {**metric.compute(), "model": "ours_no_text", "checkpoint_epoch": state["epoch"],
              "protocol": "native_960x736_full_frame_no_tta", "seed": 42,
              "elapsed_seconds": time.perf_counter()-started,
              "note": "Separate inference protocol; no extra training. Timing overlaps training and is not an FPS benchmark."}
    atomic_json(folder / "evaluation_fullframe.json", result)
    print(json.dumps({"miou": result["miou"], "class_iou": result["class_iou"]}), flush=True)


if __name__ == "__main__":
    main()
