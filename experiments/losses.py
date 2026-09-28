import torch
from torch import nn
import torch.nn.functional as F


class SoftDice(nn.Module):
    def __init__(self, smooth=0.05, ignore_index=255):
        super().__init__()
        self.smooth, self.ignore_index = smooth, ignore_index

    def forward(self, logits, target):
        valid = target != self.ignore_index
        probabilities = logits.float().softmax(1) * valid.unsqueeze(1)
        onehot = F.one_hot(target.masked_fill(~valid, 0), logits.shape[1]).permute(0, 3, 1, 2).float()
        onehot *= valid.unsqueeze(1)
        dims = (0, 2, 3)
        intersection = (probabilities * onehot).sum(dims)
        denominator = (probabilities + onehot).sum(dims)
        loss = 1 - (2 * intersection + self.smooth) / denominator.add(self.smooth).clamp_min(1e-7)
        # Same absent-class policy as the official GeoSeg Dice loss.
        return (loss * (onehot.sum(dims) > 0)).mean()


def boundary_target(target):
    valid = target != 255
    edges = torch.zeros_like(target, dtype=torch.bool)
    vertical = (target[:, 1:] != target[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    horizontal = (target[:, :, 1:] != target[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edges[:, 1:] |= vertical
    edges[:, :-1] |= vertical
    edges[:, :, 1:] |= horizontal
    edges[:, :, :-1] |= horizontal
    # Exclude an ignored pixel's neighborhood from boundary supervision.
    safe = F.max_pool2d((~valid).float().unsqueeze(1), 3, 1, 1).squeeze(1) == 0
    return edges.float(), valid & safe


def edge_bce(logits, target):
    edges, valid = boundary_target(target)
    losses = F.binary_cross_entropy_with_logits(logits.float().squeeze(1), edges, reduction="none")
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def hard_pixel_ce(logits, target, fraction=0.25):
    """Supplement ordinary CE with the hardest fraction of valid pixels.

    OHEM-style objective, independently implemented; no sampling of validation data.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Hard-pixel fraction must be in (0, 1]")
    losses = F.cross_entropy(logits.float(), target, ignore_index=255, reduction="none")
    valid_losses = losses[target != 255]
    if not valid_losses.numel():
        return logits.sum() * 0
    return valid_losses.topk(max(1, int(valid_losses.numel() * fraction)), sorted=False).values.mean()


class Confusion:
    def __init__(self, classes=5):
        self.classes = classes
        self.matrix = torch.zeros(classes, classes, dtype=torch.int64)

    def update(self, pred, target):
        valid = target != 255
        # CPU bincount avoids expensive CUDA atomic contention for only 25 bins.
        y = target[valid].detach().cpu().reshape(-1)
        p = pred[valid].detach().cpu().reshape(-1)
        self.matrix += torch.bincount(y * self.classes + p, minlength=self.classes**2).reshape(self.classes, self.classes)

    def compute(self):
        c = self.matrix.double()
        tp = c.diag()
        union = c.sum(0) + c.sum(1) - tp
        iou = tp / union.clamp_min(1)
        f1 = 2 * tp / (c.sum(0) + c.sum(1)).clamp_min(1)
        return {"miou": iou.mean().item(), "f1": f1.mean().item(),
                "pixel_accuracy": (tp.sum() / c.sum().clamp_min(1)).item(),
                "class_iou": iou.tolist(), "class_f1": f1.tolist(), "confusion_matrix": self.matrix.tolist()}
