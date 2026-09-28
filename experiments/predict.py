"""Prediction using the exact preprocessing and sliding evaluator used in training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import ROOT, COLORS, image_tensor, detail_features, dataset_spec
from .models import build_model, batch_needs
from .inference import sliding_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tta4", action="store_true", help="Optional four-flip logit averaging, about four times the network passes")
    p.add_argument("--embeddings", type=Path, default=ROOT / "clip_embeddings/clip_text_embeddings_drone.json")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"], pretrained=False, num_classes=config.get("num_classes",5)).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    with Image.open(args.image) as image:
        rgb = np.array(image.convert("RGB"))
    batch = {"image": image_tensor(rgb).unsqueeze(0).to(device)}
    needs = batch_needs(config["model"])
    if needs["detail"]:
        batch["detail"] = torch.from_numpy(detail_features(rgb)).unsqueeze(0).to(device)
    if needs["text"]:
        embeddings = json.loads(args.embeddings.read_text(encoding="utf-8"))
        if args.image.name not in embeddings:
            raise ValueError("Image has no cached BLIP/CLIP embedding. Generate its embedding first or use an ours_no_text/ours_rgb checkpoint.")
        batch["text"] = torch.tensor(embeddings[args.image.name], device=device, dtype=torch.float32).reshape(1, 512)
    if args.tta4:
        from .evaluate_ours_tta import four_flip_logits
        _, logits = four_flip_logits(model, batch, crop=config["crop"], amp=not config["no_amp"])
    else:
        logits = sliding_logits(model, batch, crop=config["crop"], stride=config["crop"]*3//4,
                                amp=not config["no_amp"])
    labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(dataset_spec(config["data_root"])["colors"], dtype=np.uint8)
    Image.fromarray(palette[labels]).save(args.output)
    Image.fromarray(labels).save(args.output.with_name(args.output.stem + "_ids.png"))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
