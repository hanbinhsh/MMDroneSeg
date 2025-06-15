import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F

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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=255):
        """
        Args:
            gamma (float): focusing parameter
            alpha (Tensor | None): class balance factor
            reduction (str): 'none' | 'mean' | 'sum'
            ignore_index (int): label to ignore
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: shape (N, C, H, W), logits (not softmaxed)
            target: shape (N, H, W), int64 with class indices
        """
        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
        else:
            alpha = None

        logpt = F.log_softmax(pred, dim=1)  # Log-probabilities
        pt = torch.exp(logpt)               # Probabilities

        # Gather the log-probability for the target class at each pixel
        target = target.long()
        loss = F.nll_loss(
            logpt, target,
            weight=alpha,
            reduction='none',
            ignore_index=self.ignore_index
        )
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)  # shape (N, H, W)

        # Compute the focal loss
        focal_loss = ((1 - pt) ** self.gamma) * loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss[target != self.ignore_index].mean()
        elif self.reduction == 'sum':
            return focal_loss[target != self.ignore_index].sum()
        else:
            return focal_loss

class EdgeLoss(nn.Module):
    def __init__(self, mode='l1'):
        super(EdgeLoss, self).__init__()
        if mode == 'bce':
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.L1Loss()

        # Sobel filter (3x3)
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def detect_edges(self, x):
        # x: [B, 1, H, W]
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edge

    def forward(self, pred, target):
        # pred: [B, C, H, W] logits -> get argmax
        pred = torch.argmax(pred, dim=1, keepdim=True).float()  # [B, 1, H, W]
        target = target.unsqueeze(1).float()  # [B, 1, H, W]

        edge_pred = self.detect_edges(pred)
        edge_target = self.detect_edges(target)

        return self.loss_fn(edge_pred, edge_target)

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