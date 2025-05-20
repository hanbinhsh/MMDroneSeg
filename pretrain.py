import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pretrainModel import vae_loss, ResNetVAE
from torch.utils.data import Dataset
from PIL import Image
import os

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image  # 只返回图像，不返回 label


def train_resnet_vae(model, train_loader, optimizer, device, epoch, kld_weight=0.005):
    """
    训练ResNet VAE一个epoch
    """
    model.train()
    train_loss = 0
    recon_loss_sum = 0
    kld_loss_sum = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data)
        loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, log_var, kld_weight)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        kld_loss_sum += kld_loss.item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_sum / len(train_loader.dataset)
    avg_kld = kld_loss_sum / len(train_loader.dataset)

    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}, '
          f'Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}')

    return avg_loss, avg_recon, avg_kld


def pretrain_resnet():
    """
    ResNet VAE预训练主函数
    """
    # 参数设置
    batch_size = 16  # 减小批量大小以节省内存
    epochs = 50
    learning_rate = 1e-4
    latent_dim = 256
    use_skip_connections = True  # 是否使用跳跃连接
    input_height = 384
    input_width = 512

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor()
    ])

    # 替换为你的数据集路径
    data_path = '../dataset/Drone/classes_dataset/classes_dataset/original_images/'
    print(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        print(f"WARNING: Dataset path {data_path} does not exist!")
        print("Please check your dataset path and update accordingly.")
        return

    train_dataset = UnlabeledImageDataset(data_path, transform=transform)
    print(f"Dataset size: {len(train_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 初始化模型
    model = ResNetVAE(
        latent_dim=latent_dim,
        use_skip_connections=use_skip_connections,
        input_height=input_height,
        input_width=input_width
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 创建保存检查点的目录
    os.makedirs('checkpoints', exist_ok=True)

    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, recon_loss, kld_loss = train_resnet_vae(
            model, train_loader, optimizer, device, epoch)

        # 保存检查点
        if epoch % 5 == 0 or epoch == 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'recon_loss': recon_loss,
                'kld_loss': kld_loss,
            }, f'checkpoints/resnet_vae_epoch_{epoch}.pth')
            print(f"Checkpoint saved at epoch {epoch}")

    # 最终保存预训练的编码器权重
    torch.save(model.encoder.state_dict(), 'pretrained_resnet_encoder.pth')
    print("Pretrained encoder weights saved to 'pretrained_resnet_encoder.pth'")

    return model.encoder


if __name__ == '__main__':
    pretrain_resnet()