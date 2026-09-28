import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class BoundaryLoss(torch.nn.Module):
    """
    Boundary-aware loss using Sobel gradient filters instead of Canny edge detection.
    """

    def __init__(self, weight=1.0):
        super(BoundaryLoss, self).__init__()
        self.weight = weight
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)

    def compute_boundaries(self, tensor, C):
        B, _, H, W = tensor.shape
        boundaries = torch.zeros((B, H, W), device=tensor.device)

        for c in range(C):
            t = tensor[:, c:c + 1, :, :]
            grad_x = F.conv2d(t, self.sobel_x, padding=1)
            grad_y = F.conv2d(t, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(1)
            boundaries += grad_mag
        return boundaries

    def forward(self, pred, target):
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        B, C, H, W = pred.shape
        pred_probs = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        pred_boundaries = self.compute_boundaries(pred_probs, C)
        target_boundaries = self.compute_boundaries(target_one_hot, C)

        pred_boundaries = torch.sigmoid(pred_boundaries)
        target_boundaries = (target_boundaries > 0.1).float()

        loss = F.binary_cross_entropy(pred_boundaries, target_boundaries)
        return self.weight * loss, pred_boundaries, target_boundaries


def load_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert("L")  # 灰度图
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()  # [0,1]
    ])
    return transform(img).unsqueeze(0)  # (1, 1, H, W)


def visualize(original, pred_boundary, target_boundary):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original.squeeze().cpu(), cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(pred_boundary.squeeze().cpu(), cmap='hot')
    axs[1].set_title("Predicted Boundaries")
    axs[2].imshow(target_boundary.squeeze().cpu(), cmap='hot')
    axs[2].set_title("Target Boundaries")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_image(args.image).to(device)

    # 模拟二分类预测（伪 logits）
    pred_logits = torch.cat([img, 1 - img], dim=1) * 5  # (1, 2, H, W)
    target = (img > 0.5).long().squeeze(1)  # (1, H, W)

    loss_fn = BoundaryLoss()
    loss, pred_b, target_b = loss_fn(pred_logits, target)

    print(f"Boundary loss: {loss.item():.4f}")
    visualize(img, pred_b, target_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoundaryLoss Sobel Preview")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    main(args)
