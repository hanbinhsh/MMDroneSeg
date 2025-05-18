import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim=256, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        # 存储中间特征图用于跳跃连接
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # 1/4 resolution, 64 channels
        self.layer1 = resnet.layer1  # 1/4 resolution, 64 channels
        self.layer2 = resnet.layer2  # 1/8 resolution, 128 channels
        self.layer3 = resnet.layer3  # 1/16 resolution, 256 channels
        self.layer4 = resnet.layer4  # 1/32 resolution, 512 channels

        # 添加VAE特定层以生成潜在空间分布参数
        self.mu = nn.Linear(512 * (512 // 32) * (384 // 32), latent_dim)  # 根据输入大小调整
        self.log_var = nn.Linear(512 * (512 // 32) * (384 // 32), latent_dim)

        self.out_channels = 512

    def forward(self, x):
        # 保存中间特征用于跳跃连接
        x0 = self.layer0(x)  # 1/4
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        # 扁平化用于VAE潜在空间
        batch_size = x4.size(0)
        x_flat = x4.view(batch_size, -1)

        # 获取潜在空间参数
        mu = self.mu(x_flat)
        log_var = self.log_var(x_flat)

        # 重参数化技巧
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 返回潜在向量和所有特征(用于跳跃连接)
        return z, mu, log_var, (x0, x1, x2, x3, x4)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # 从潜在空间投影到初始特征图
        self.latent_proj = nn.Linear(latent_dim, 512 * (512 // 32) * (384 // 32))

        # 上采样块 - 深度逐渐减小，分辨率增加
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 1/16 resolution

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 1/8 resolution

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 1/4 resolution

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 1/2 resolution

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )

    def forward(self, z):
        # 将潜在向量重塑为初始特征图
        x = self.latent_proj(z)
        x = x.view(-1, 512, 512 // 32, 384 // 32)  # 重塑为特征图

        # 上采样
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)

        return x


class VAEDecoderWithSkips(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # 从潜在空间投影到初始特征图
        self.latent_proj = nn.Linear(latent_dim, 512 * (512 // 32) * (384 // 32))

        # 上采样块 - 每个块都接收上一层输出和对应的跳跃连接
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 1/16 resolution

        self.skip1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 1/8 resolution

        self.skip2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 1/4 resolution

        self.skip3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 1/2 resolution

        self.skip4 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, skip_features):
        x0, x1, x2, x3, x4 = skip_features

        # 将潜在向量重塑为初始特征图
        x = self.latent_proj(z)
        x = x.view(-1, 512, 512 // 32, 384 // 32)  # 重塑为特征图

        # 上采样并融合跳跃连接
        x = self.up1(x)
        # 使用插值调整skip connection的大小以匹配上采样特征的大小
        x3_resized = F.interpolate(x3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x3_resized], dim=1)
        x = self.skip1(x)

        x = self.up2(x)
        x2_resized = F.interpolate(x2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x2_resized], dim=1)
        x = self.skip2(x)

        x = self.up3(x)
        x1_resized = F.interpolate(x1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x1_resized], dim=1)
        x = self.skip3(x)

        x = self.up4(x)
        x0_resized = F.interpolate(x0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x0_resized], dim=1)
        x = self.skip4(x)

        x = self.final(x)

        return x


class ResNetVAE(nn.Module):
    def __init__(self, latent_dim=256, use_skip_connections=True):
        super().__init__()
        self.encoder = ResNetEncoder(latent_dim=latent_dim)
        self.use_skip_connections = use_skip_connections

        if use_skip_connections:
            self.decoder = VAEDecoderWithSkips(latent_dim=latent_dim)
        else:
            self.decoder = VAEDecoder(latent_dim=latent_dim)

    def forward(self, x):
        # 编码
        z, mu, log_var, skip_features = self.encoder(x)

        # 解码
        if self.use_skip_connections:
            recon = self.decoder(z, skip_features)
        else:
            recon = self.decoder(z)

        return recon, mu, log_var


def vae_loss(recon_x, x, mu, log_var, kld_weight=0.005):
    """
    计算VAE损失: 重建损失 + KL散度
    """
    # 重建损失 - 使用MSE
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL散度
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # 总损失
    loss = recon_loss + kld_weight * kld_loss

    return loss, recon_loss, kld_loss

def load_pretrained_encoder_for_segmodel(seg_model, encoder_path='pretrained_resnet_encoder.pth'):
    """
    将预训练的编码器加载到分割模型中
    """
    # 加载预训练权重
    pretrained_dict = torch.load(encoder_path)

    # 提取共享编码器的状态字典
    encoder_dict = {}
    for k, v in pretrained_dict.items():
        # 只保留layer0-layer4的权重
        if any(layer in k for layer in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']):
            encoder_dict[k] = v

    # 加载到分割模型的共享编码器中
    seg_model.encoder_image.load_state_dict(encoder_dict, strict=False)

    # 使用相同的权重初始化其他编码器
    seg_model.encoder_dog.load_state_dict(encoder_dict, strict=False)
    seg_model.encoder_thresh.load_state_dict(encoder_dict, strict=False)

    return seg_model