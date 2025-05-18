import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kf

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

class EdgeLoss(nn.Module):
    """
    Edge-aware loss function for semantic segmentation that focuses on boundaries.
    Uses Canny edge detection to extract edges from both predictions and target masks,
    then computes the loss between these edge maps.
    """

    def __init__(self, low_threshold=0.1, high_threshold=0.3, edge_weight=1.0, sigma=1.0):
        super(EdgeLoss, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.edge_weight = edge_weight
        self.sigma = sigma

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted segmentation logits of shape (B, C, H, W)
            target: Ground truth segmentation mask of shape (B, H, W)

        Returns:
            Edge loss between predictions and targets
        """
        B, C, H, W = pred.shape
        device = pred.device

        # 根据批量大小构造 sigma 的张量，形状为 (B, 2)
        sigma_tensor = torch.tensor([self.sigma, self.sigma], device=device).unsqueeze(0).expand(B, -1)

        # Convert predictions to probability maps using softmax
        pred_probs = F.softmax(pred, dim=1)  # (B, C, H, W)

        # For predictions, compute edge maps for each class channel and sum them
        pred_edge_maps = torch.zeros((B, H, W), device=device)
        for c in range(C):
            # Extract probability map for this class
            class_prob = pred_probs[:, c, :, :].unsqueeze(1)  # (B, 1, H, W)

            # Apply Canny edge detection using Kornia
            edges = kf.canny(
                class_prob,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                sigma=sigma_tensor  # 使用修改后的 sigma_tensor
            )[0]  # Get the first output which is the edge map

            # Add to the cumulative edge map
            pred_edge_maps += edges.squeeze(1)

        # Normalize pred edge maps to [0, 1]
        pred_edge_maps = torch.clamp(pred_edge_maps, 0, 1)

        # For targets, create one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute edge maps for each target class and sum them
        target_edge_maps = torch.zeros((B, H, W), device=device)
        for c in range(C):
            class_mask = target_one_hot[:, c, :, :].unsqueeze(1)  # (B, 1, H, W)

            # Apply Canny edge detection using the同样的 sigma_tensor
            edges = kf.canny(
                class_mask,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                sigma=sigma_tensor
            )[0]

            target_edge_maps += edges.squeeze(1)

        # Normalize target edge maps to [0, 1]
        target_edge_maps = torch.clamp(target_edge_maps, 0, 1)

        # Compute binary cross entropy loss between edge maps
        edge_loss = F.binary_cross_entropy(pred_edge_maps, target_edge_maps)

        return self.edge_weight * edge_loss


class BoundaryLoss(nn.Module):
    """
    Alternative implementation of boundary-aware loss without using Canny edge detection.
    Uses simple gradient operations to detect boundaries.
    """

    def __init__(self, weight=1.0):
        super(BoundaryLoss, self).__init__()
        self.weight = weight
        # Sobel filters for gradient computation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted segmentation logits of shape (B, C, H, W)
            target: Ground truth segmentation mask of shape (B, H, W)
        """
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        B, C, H, W = pred.shape

        # Convert predictions to probability maps using softmax
        pred_probs = F.softmax(pred, dim=1)  # (B, C, H, W)

        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute boundaries for predictions
        pred_boundaries = torch.zeros((B, H, W), device=device)
        target_boundaries = torch.zeros((B, H, W), device=device)

        for c in range(C):
            # For predictions
            pred_c = pred_probs[:, c:c + 1, :, :]  # Keep dimension for conv (B, 1, H, W)
            grad_x_pred = F.conv2d(pred_c, self.sobel_x, padding=1)
            grad_y_pred = F.conv2d(pred_c, self.sobel_y, padding=1)
            pred_grad_mag = torch.sqrt(grad_x_pred ** 2 + grad_y_pred ** 2).squeeze(1)
            pred_boundaries += pred_grad_mag

            # For targets
            target_c = target_one_hot[:, c:c + 1, :, :]
            grad_x_target = F.conv2d(target_c, self.sobel_x, padding=1)
            grad_y_target = F.conv2d(target_c, self.sobel_y, padding=1)
            target_grad_mag = torch.sqrt(grad_x_target ** 2 + grad_y_target ** 2).squeeze(1)
            target_boundaries += target_grad_mag

        # Normalize
        pred_boundaries = torch.sigmoid(pred_boundaries)
        target_boundaries = (target_boundaries > 0.1).float()  # Threshold to get binary boundaries

        # Compute loss
        boundary_loss = F.binary_cross_entropy(pred_boundaries, target_boundaries)

        return self.weight * boundary_loss