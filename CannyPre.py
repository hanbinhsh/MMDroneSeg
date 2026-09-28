import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.filters as kf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def load_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert("L")  # 转为灰度图
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()  # [0,1]
    ])
    return transform(img).unsqueeze(0)  # (1, 1, H, W)

def canny_edge_map(tensor, low_threshold, high_threshold, sigma):
    B, C, H, W = tensor.shape
    sigma_tensor = torch.tensor([sigma, sigma], device=tensor.device).unsqueeze(0).expand(B, -1)
    edges = kf.canny(
        tensor,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        sigma=sigma_tensor
    )[0]
    return edges

def visualize_edges(original, pred_edges, target_edges):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original.squeeze().cpu(), cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(pred_edges.squeeze().cpu(), cmap='gray')
    axs[1].set_title("Predicted Edge Map")
    axs[2].imshow(target_edges.squeeze().cpu(), cmap='gray')
    axs[2].set_title("Target Edge Map")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main(args):
    # Load image
    img = load_image(args.image).to("cuda" if torch.cuda.is_available() else "cpu")  # (1, 1, H, W)

    # 模拟模型输出：假设是二分类 logits（直接复制图像作为一类的激活）
    pred_logits = torch.cat([img, 1 - img], dim=1) * 5  # (1, 2, H, W)

    # 伪标签：按 0.5 阈值分为两类
    target = (img > 0.5).long().squeeze(1)  # (1, H, W)

    # 预测边缘图
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_edge = torch.zeros_like(img)
    for c in range(2):
        class_prob = pred_probs[:, c, :, :].unsqueeze(1)
        edge = canny_edge_map(class_prob, args.canny_low, args.canny_high, args.canny_sigma)
        pred_edge += edge

    # 标签边缘图
    target_one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
    target_edge = torch.zeros_like(img)
    for c in range(2):
        class_mask = target_one_hot[:, c, :, :].unsqueeze(1)
        edge = canny_edge_map(class_mask, args.canny_low, args.canny_high, args.canny_sigma)
        target_edge += edge

    # 可视化
    visualize_edges(img, pred_edge, target_edge)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeLoss Canny Preview")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--canny_low", type=float, default=0.1, help="Canny low threshold")
    parser.add_argument("--canny_high", type=float, default=0.3, help="Canny high threshold")
    parser.add_argument("--canny_sigma", type=float, default=1.0, help="Canny Gaussian sigma")
    args = parser.parse_args()
    main(args)
