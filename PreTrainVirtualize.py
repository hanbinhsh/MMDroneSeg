import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from pretrainModel import ResNetVAE


def test_with_single_image(image_path, checkpoint_path, use_skip_connections=True,
                           latent_dim=256, height=384, width=512, device='cuda'):
    """
    对单个图像测试VAE重建效果的简化版本
    """
    # 检查图像路径
    if not os.path.exists(image_path):
        print(f"错误: 图像路径 {image_path} 不存在!")
        return

    # 检查检查点路径
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点路径 {checkpoint_path} 不存在!")
        return

    # 设备配置
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = ResNetVAE(
        latent_dim=latent_dim,
        use_skip_connections=use_skip_connections,
        input_height=height,
        input_width=width
    ).to(device)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"模型加载成功: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    # 设置为评估模式
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])

    # 加载和处理图像
    image = Image.open(image_path).convert('RGB')
    original = image.copy()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 重建图像
    with torch.no_grad():
        recon_batch, _, _ = model(image_tensor)

    # 转换为PIL图像
    recon_img = transforms.ToPILImage()(recon_batch.squeeze(0).cpu())

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(recon_img)
    plt.title("Reconstruction Result")
    plt.axis('off')

    plt.tight_layout()
    output_file = 'reconstruction_result.png'
    plt.savefig(output_file)
    plt.show()
    print(f"结果已保存为 {output_file}")


def main():
    parser = argparse.ArgumentParser(description='VAE重建快速测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--skip', action='store_true', default=True, help='使用跳跃连接')
    parser.add_argument('--latent_dim', type=int, default=256, help='潜在空间维度')
    parser.add_argument('--height', type=int, default=384, help='输入高度')
    parser.add_argument('--width', type=int, default=512, help='输入宽度')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')

    args = parser.parse_args()

    device = 'cpu' if args.cpu else 'cuda'

    test_with_single_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        use_skip_connections=args.skip,
        latent_dim=args.latent_dim,
        height=args.height,
        width=args.width,
        device=device
    )


if __name__ == '__main__':
    main()