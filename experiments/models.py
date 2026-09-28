"""Thin adapters around pinned, unmodified author repositories."""
import importlib.util
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_base

from .data import ROOT
from .ours import DroneSegV2
from .prepare import weight_path
from .losses import SoftDice, edge_bce, hard_pixel_ce

EXTERNAL = ROOT / "external"
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(0, str(EXTERNAL / "rssegmentation"))


def load_source(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def author_loss_package(name, directory):
    # Some author __init__.py files import missing, unrelated losses. Load the
    # actual published CE/Dice modules without rewriting their implementations.
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    ce = importlib.import_module(f"{name}.soft_ce").SoftCrossEntropyLoss
    dice = importlib.import_module(f"{name}.dice").DiceLoss
    return ce(smooth_factor=0.05, ignore_index=255), dice(smooth=0.05, ignore_index=255)


def require_weight(name, pretrained):
    if not pretrained:
        return None
    path = weight_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m experiments.prepare --weights {name}")
    return path


class OfficialModel(nn.Module):
    def __init__(self, name, pretrained=True, num_classes=5):
        super().__init__()
        self.name = name
        checkpoint = require_weight(name, pretrained)
        if name == "d2ls":
            mod = load_source("official_d2ls", EXTERNAL / "D2LS/network/models/d2ls.py")
            def local_convnext(*args, **kwargs):
                backbone = convnext_base(weights=None)
                if checkpoint:
                    backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                return backbone
            # Only replace weight retrieval, not backbone math or the author's model.
            with patch.object(mod, "convnext_base", local_convnext):
                self.net = mod.DynamicDictionaryLearning(model="convnext_base", token_length=num_classes, l=3)
        elif name == "afenet":
            from AFENet.model.AFENet import AFENet
            self.net = AFENet(num_classes=num_classes, pretrained=False, backbone_name="resnet18.fb_swsl_ig1b_ft_in1k")
            if checkpoint:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                result = self.net.backbone.load_state_dict(state, strict=False)
                if result.missing_keys or set(result.unexpected_keys) - {"fc.weight", "fc.bias"}:
                    raise RuntimeError(f"AFENet incompatible pretrained weights: {result}")
            # CUDA half FFT only supports power-of-two dimensions. Always compute this
            # published operation in FP32, keeping the rest of the author network in AMP.
            from AFENet.model.AFSIM import AFSIModule
            for module in self.net.modules():
                if isinstance(module, AFSIModule):
                    original = type(module).fft
                    def fft_fp32(instance, x, fn=original):
                        with torch.autocast(device_type=x.device.type, enabled=False):
                            return fn(instance, x.float())
                    module.fft = types.MethodType(fft_fp32, module)
        elif name == "logcan":
            from rsseg.models.backbones import repvit as rep
            from rsseg.models.segheads.logcanplus_head import LoGCANPlus_Head
            from rsseg.models.classifiers.base_classifier import Base_Classifier
            if checkpoint:
                self.backbone = rep.repvit_m2_3(init_cfg={"checkpoint": str(checkpoint)}, out_indices=[7, 15, 51, 54])
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                state = state.get("model", state.get("state_dict", state))
                result = self.backbone.load_state_dict(state, strict=False)
                if result.missing_keys:
                    raise RuntimeError(f"LOGCAN backbone not fully initialized: {result.missing_keys}")
            else:
                with patch.object(rep.RepViT, "init_weights", lambda *_: None):
                    self.backbone = rep.repvit_m2_3(init_cfg={}, out_indices=[7, 15, 51, 54])
            self.head = LoGCANPlus_Head(transform_channel=96, in_channel=[80, 160, 320, 640],
                                       num_class=num_classes, num_heads=8, patch_size=(4, 4))
            self.classifier = Base_Classifier(transform_channel=96, num_class=num_classes)
        else:
            raise ValueError(name)

    def forward(self, batch):
        x = batch["image"]
        if self.name == "logcan":
            outputs = self.classifier(self.head(self.backbone(x)))
            outputs = [F.interpolate(o, size=x.shape[-2:], mode="bilinear", align_corners=False) for o in outputs]
            return {"logits": outputs[0], "aux": outputs[1]}
        result = self.net(x)
        if isinstance(result, (tuple, list)):
            output = {"logits": result[0]}
            for item in result[1:]:
                output["contrastive" if item.ndim == 0 else "aux"] = item
            return output
        return {"logits": result}


ABLATION_FLAGS = {"ours_ablation_rgb": (False, False),
                  "ours_ablation_detail": (True, False),
                  "ours_ablation_boundary": (False, True)}

MODEL_NAMES = ["d2ls", "afenet", "logcan", "ours", "ours_rgb", "ours_no_detail", "ours_no_text", "ours_no_boundary",
               "ours_sched50", "ours_refine"]
MODEL_NAMES += list(ABLATION_FLAGS)

NO_TEXT = ("ours_rgb", "ours_no_text", "ours_sched50", "ours_refine")


def build_model(name, pretrained=True, num_classes=5):
    if name in ABLATION_FLAGS:
        # Initialize exactly like the full no-text reference, then remove modules.
        # Common weights and the post-construction RNG state stay matched at seed 42.
        model = DroneSegV2(require_weight("convnext_tiny", pretrained),
                           detail=True, text=False, boundary=True, num_classes=num_classes)
        model.use_detail, model.use_boundary = ABLATION_FLAGS[name]
        if not model.use_detail:
            for attr in ("detail_stem", "detail_down", "detail_proj", "detail_gate"):
                delattr(model, attr)
        if not model.use_boundary:
            del model.edge_head
        return model
    if name.startswith("ours"):
        return DroneSegV2(require_weight("convnext_tiny", pretrained),
                          detail=name not in ("ours_rgb", "ours_no_detail"),
                          text=name not in NO_TEXT,
                          boundary=name not in ("ours_rgb", "ours_no_boundary"),
                          refine=name == "ours_refine", num_classes=num_classes)
    return OfficialModel(name, pretrained, num_classes=num_classes)


def batch_needs(name):
    if name in ABLATION_FLAGS:
        return {"detail": ABLATION_FLAGS[name][0], "text": False}
    return {"detail": name.startswith("ours") and name not in ("ours_rgb", "ours_no_detail"),
            "text": name.startswith("ours") and name not in NO_TEXT}


class Objective(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.dice = SoftDice(smooth=0.05)
        if name in ("afenet", "d2ls"):
            directory = EXTERNAL / ("AFENet/losses" if name == "afenet" else "D2LS/network/losses")
            self.author_ce, self.author_dice = author_loss_package(f"official_{name}_losses", directory)

    def joint(self, logits, labels):
        return self.author_ce(logits.float(), labels) + self.author_dice(logits.float(), labels)

    def forward(self, outputs, labels):
        if not (labels != 255).any():
            # Entirely void crops have no supervised target; avoid mean-CE NaNs.
            return sum(value.float().sum() * 0 for value in outputs.values() if isinstance(value, torch.Tensor))
        if self.name == "logcan":
            return F.cross_entropy(outputs["logits"].float(), labels, ignore_index=255) + 0.8 * F.cross_entropy(outputs["aux"].float(), labels, ignore_index=255)
        if self.name.startswith("ours"):
            loss = F.cross_entropy(outputs["logits"].float(), labels, ignore_index=255) + self.dice(outputs["logits"], labels)
            if "boundary" in outputs:
                loss = loss + 0.1 * edge_bce(outputs["boundary"], labels)
            if self.name == "ours_refine":
                loss = loss + 0.25 * hard_pixel_ce(outputs["logits"], labels)
                loss = loss + 0.2 * F.cross_entropy(outputs["coarse_logits"].float(), labels, ignore_index=255)
            return loss
        if self.name == "afenet":
            return self.joint(outputs["logits"], labels)
        # Author UnetFormerLoss uses CE+Dice for the main head, CE only for aux.
        loss = self.joint(outputs["logits"], labels)
        if "aux" in outputs:
            loss = loss + 0.4 * self.author_ce(outputs["aux"].float(), labels)
        if "contrastive" in outputs:
            loss = loss + outputs["contrastive"]
        return loss
