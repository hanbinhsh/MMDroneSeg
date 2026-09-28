import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "Drone/classes_dataset/classes_dataset"
CLASSES = ["obstacles", "water", "soft-surfaces", "moving-objects", "landing-zones"]
COLORS = np.array([[155, 38, 182], [14, 135, 204], [124, 252, 0],
                   [255, 20, 147], [169, 169, 169]], dtype=np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE = 255


def dataset_spec(root):
    path = Path(root) / "dataset.json"
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return spec
    return {"classes": CLASSES, "colors": COLORS.tolist(), "mask_mode": "rgb",
            "protocol": "native_960x736_crop512_sliding_no_tta"}


def decode_mask(rgb):
    """Exact, vectorized color mapping; never silently turn unknown colors into class 0."""
    packed = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    palette = (COLORS[:, 0].astype(np.int32) << 16) | (COLORS[:, 1].astype(np.int32) << 8) | COLORS[:, 2]
    result = np.full(rgb.shape[:2], IGNORE, np.uint8)
    for idx, color in enumerate(palette):
        result[packed == color] = idx
    if (result == IGNORE).any():
        unknown = np.unique(rgb[result == IGNORE], axis=0)
        raise ValueError(f"Unknown label colors: {unknown[:12].tolist()}")
    return result


def detail_features(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Signed DoG retains both sides of an edge (uint8 subtraction clipped one side).
    dog = (cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
           - cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32)) / 255.0
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.stack([dog, threshold.astype(np.float32) / 255], axis=0)


def image_tensor(rgb):
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


def targeted_crop_origin(label, crop, rare_class_ids, rng):
    """Uniformly choose a present rare class and a pixel; keep it in the crop.

    Called only for training. Return None to retain the original uniform crop
    when this image has none of the selected classes.
    """
    present = np.intersect1d(np.unique(label), rare_class_ids)
    if not len(present):
        return None
    class_id = int(rng.choice(present))
    pixels = np.flatnonzero(label == class_id)
    y, x = divmod(int(rng.choice(pixels)), label.shape[1])
    top = int(rng.integers(max(0, y - crop + 1), min(y, label.shape[0] - crop) + 1))
    left = int(rng.integers(max(0, x - crop + 1), min(x, label.shape[1] - crop) + 1))
    return top, left


class DroneData(Dataset):
    def __init__(self, root=DATA_ROOT, split="train", crop=512, seed=42, text=False,
                 detail=False, limit=None, augment=True, legacy_size=False,
                 class_aware_crop_prob=0.0, rare_class_ids=None):
        self.root = Path(root)
        self.spec = dataset_spec(self.root)
        self.classes = self.spec["classes"]
        self.colors = np.asarray(self.spec["colors"], dtype=np.uint8)
        self.num_classes = len(self.classes)
        self.class_aware_crop_prob = float(class_aware_crop_prob)
        self.rare_class_ids = sorted(set(rare_class_ids or []))
        if not 0 <= self.class_aware_crop_prob <= 1:
            raise ValueError("class-aware-crop-prob must be in [0, 1]")
        if any(i < 0 or i >= self.num_classes for i in self.rare_class_ids):
            raise ValueError("rare-class-ids must be valid training IDs, excluding ignore")
        if self.class_aware_crop_prob and (split != "train" or legacy_size or not self.rare_class_ids):
            raise ValueError("Class-aware cropping requires training crops and nonempty rare-class-ids")
        self.split, self.crop, self.seed = split, crop, seed
        self.text, self.detail, self.augment = text, detail, augment
        self.legacy_size = legacy_size
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        image_dir, mask_dir = ("original_images", "label_images_semantic") if split == "train" else ("val_original", "val_label")
        self.images, self.masks = self.root / image_dir, self.root / mask_dir
        if "image_roots" in self.spec:
            self.images = Path(self.spec["image_roots"][split])
        self.names = sorted(p.name for p in self.images.glob("*.png"))
        masks = {p.name for p in self.masks.glob("*.png")}
        if not self.names or set(self.names) != masks:
            raise ValueError(f"Missing or unpaired images in {self.root}: {image_dir}")
        if limit:
            self.names = self.names[:limit]
        self.embeddings = {}
        if text:
            self.embeddings = json.loads((ROOT / "clip_embeddings/clip_text_embeddings_drone.json").read_text())
            missing = set(self.names) - self.embeddings.keys()
            if missing:
                raise ValueError(f"Missing text embeddings: {sorted(missing)[:8]}")
        # Palette decoding is cached on disk; RGB decoding remains lazy to keep worker RAM bounded.
        self.cache = ROOT / ".cache/drone_masks"
        self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.names)

    @property
    def epoch(self):
        return int(self._epoch.item())

    @epoch.setter
    def epoch(self, value):
        self._epoch.fill_(value)

    def read(self, idx):
        name = self.names[idx]
        with Image.open(self.images / name) as im:
            rgb = np.array(im.convert("RGB"))
        path = self.masks / name
        stat = path.stat()
        signature = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:palette-v1"
        if self.spec.get('manifest_sha256'):
            signature += ':' + self.spec['manifest_sha256']
        key = hashlib.sha256(signature.encode()).hexdigest()
        cache = self.cache / f"{key}.npy"
        if cache.exists():
            label = np.load(cache, allow_pickle=False)
        else:
            with Image.open(path) as im:
                if self.spec["mask_mode"] == "indexed":
                    label = np.array(im)
                    if label.ndim != 2 or not np.all((label < self.num_classes) | (label == IGNORE)):
                        raise ValueError(f"Invalid indexed labels: {path}")
                else:
                    label = decode_mask(np.array(im.convert("RGB")))
            # Preparation runs before DataLoader workers are started.
            np.save(cache, label, allow_pickle=False)
        if rgb.shape[:2] != label.shape:
            raise ValueError(f"Image/mask shape mismatch: {name}")
        return rgb, label

    def __getitem__(self, idx):
        rgb, label = self.read(idx)
        # Per-sample randomness makes resumed runs independent of worker scheduling.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, idx]))
        if self.legacy_size:
            rgb = cv2.resize(rgb, (480, 368), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (480, 368), interpolation=cv2.INTER_NEAREST)
        elif self.split == "train":
            h, w = label.shape
            pad_h, pad_w = max(0, self.crop - h), max(0, self.crop - w)
            rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE)
            h, w = label.shape
            top, left = rng.integers(h - self.crop + 1), rng.integers(w - self.crop + 1)
            if self.class_aware_crop_prob:
                # Separate stream leaves rotation/flip/photometric draws unchanged.
                crop_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, idx, 20260924]))
                if crop_rng.random() < self.class_aware_crop_prob:
                    origin = targeted_crop_origin(label, self.crop, self.rare_class_ids, crop_rng)
                    if origin is not None:
                        top, left = origin
            rgb = rgb[top:top+self.crop, left:left+self.crop].copy()
            label = label[top:top+self.crop, left:left+self.crop].copy()
        if self.split == "train" and self.augment:
            if not self.legacy_size:
                k = int(rng.integers(4))
                rgb, label = np.rot90(rgb, k), np.rot90(label, k)
            if rng.random() < 0.5:
                rgb, label = np.fliplr(rgb), np.fliplr(label)
            if rng.random() < 0.5:
                rgb, label = np.flipud(rgb), np.flipud(label)
            rgb = np.clip(rgb.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        rgb, label = np.ascontiguousarray(rgb), np.ascontiguousarray(label)
        result = {"image": image_tensor(rgb), "mask": torch.from_numpy(label.astype(np.int64)), "name": self.names[idx]}
        if self.detail:
            result["detail"] = torch.from_numpy(detail_features(rgb))
        if self.text:
            embed = np.array(self.embeddings[self.names[idx]], dtype=np.float32).reshape(-1)
            if embed.shape != (512,) or not np.isfinite(embed).all():
                raise ValueError(f"Invalid text embedding: {self.names[idx]}")
            result["text"] = torch.from_numpy(embed)
        return result


def audit(root=DATA_ROOT):
    counts = np.zeros(5, np.int64)
    hashes = {}
    split_names = {}
    shapes = set()
    for split in ["train", "val"]:
        data = DroneData(root, split, text=True)
        split_names[split] = data.names
        hashes[split] = set()
        for i in range(len(data)):
            rgb, label = data.read(i)
            shapes.add(tuple(label.shape))
            hashes[split].add(hashlib.sha256(rgb.tobytes()).hexdigest())
            if split == "train":
                counts += np.bincount(label.reshape(-1), minlength=5)
        print(f"Audited {split}: {len(data)} pairs", flush=True)
    overlap = hashes["train"] & hashes["val"]
    if overlap:
        raise ValueError(f"Train/val exact image duplication: {len(overlap)}")
    return {"splits": split_names, "shapes_hw": sorted(shapes), "train_pixels": counts.tolist(),
            "train_frequency": (counts / counts.sum()).tolist(), "exact_duplicates_across_splits": 0,
            "near_duplicate_check": "not performed; original split retained", "classes": CLASSES}
