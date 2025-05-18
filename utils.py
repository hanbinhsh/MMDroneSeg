import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred == target).float().sum()
        dice = (2. * intersection + self.smooth) / (pred.numel() + target.numel() + self.smooth)
        return 1 - dice

def compute_metrics(preds, labels, num_classes):
    """
    Compute F1 score, mean IoU and per-class IoU

    Args:
        preds: list of prediction tensors
        labels: list of ground truth tensors
        num_classes: number of classes

    Returns:
        f1: macro F1 score
        miou: mean IoU
        iou: per-class IoU
    """
    preds = torch.cat(preds).numpy().flatten()
    labels = torch.cat(labels).numpy().flatten()

    # Compute confusion matrix
    valid_mask = (labels != 255)  # Ignore invalid pixels (255)
    valid_preds = preds[valid_mask]
    valid_labels = labels[valid_mask]

    # Compute macro F1 score
    f1 = f1_score(valid_labels, valid_preds, average='macro', labels=range(num_classes), zero_division=0)

    # Compute confusion matrix
    cm = confusion_matrix(valid_labels, valid_preds, labels=range(num_classes))

    # Compute IoU for each class
    intersection = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        if union[i] > 0:
            iou[i] = intersection[i] / union[i]
        else:
            iou[i] = 0.0

    # Compute mean IoU
    miou = np.mean(iou)

    return f1, miou, iou