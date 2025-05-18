import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

WIDTH =  512       #960
HEIGHT = 384       #736

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        # Store intermediate feature maps for skip connections
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

        # Freeze all encoder parameters
        for param in self.parameters():
            param.requires_grad = False

        self.out_channels = 512

    def forward(self, x):
        # Store intermediate features for skip connections
        x0 = self.layer0(x)  # 1/4
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        # Return all features for skip connections
        return x4, (x0, x1, x2, x3)


class TextTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(100, d_model))  # max token = 100

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pos_embedding[:T, :]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        return x.permute(1, 0, 2)


class FusionTransformer(nn.Module):
    def __init__(self, in_channels=512, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),  # /2
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),  # /4
        #     nn.ReLU(inplace=True),
        # )

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(512, embed_dim))                  # 1024

    def forward(self, x):
        B, C, H, W = x.shape
        # x = self.downsample(x)  # (B, C, H/4, W/4)
        H_ds, W_ds = x.shape[2], x.shape[3]
        x = self.proj(x)  # → (B, embed_dim, H/4, W/4)

        x = rearrange(x, 'b c h w -> b (h w) c')
        N = x.shape[1]
        pos = self.pos_embedding[:N, :].unsqueeze(0)
        x = x + pos
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_ds, w=W_ds)
        return x


class SkipConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # Upsample x to match skip connection size
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=256, num_classes=21):
        super().__init__()
        # Skip connection blocks
        self.skip3 = SkipConnectionBlock(in_channels + 256, 128)  # 1/16 resolution
        self.skip2 = SkipConnectionBlock(128 + 128, 64)  # 1/8 resolution
        self.skip1 = SkipConnectionBlock(64 + 64, 32)  # 1/4 resolution
        self.skip0 = SkipConnectionBlock(32 + 64, 32)  # 1/1 resolution

        # Final decoder layers
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x, skip_features, size):
        # Unpack skip features
        x0, x1, x2, x3 = skip_features

        # Apply skip connections
        x = self.skip3(x, x3)  # 1/16 resolution
        x = self.skip2(x, x2)  # 1/8 resolution
        x = self.skip1(x, x1)  # 1/4 resolution
        x = self.skip0(x, x0)  # Full resolution

        # Final convolution and upsampling to original size
        x = self.final(x)
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class MultiModalSegModel(nn.Module):
    def __init__(self, text_dim=512, num_classes=21):
        super().__init__()
        shared_encoder = ResNetEncoder()
        self.encoder_image = shared_encoder
        self.encoder_dog = shared_encoder
        self.encoder_thresh = shared_encoder

        self.encoder_text = TextTransformerEncoder(d_model=text_dim)

        self.channel_reduction = nn.Sequential(
            nn.Conv2d(512 * 3 + text_dim, 512, kernel_size=1),
            nn.ReLU()
        )

        self.fusion = FusionTransformer(in_channels=512, embed_dim=256)
        self.decoder = Decoder(in_channels=256, num_classes=num_classes)

    def forward(self, image, dog, thresh, text_embeds):
        B = image.size(0)

        # Get features and skip connections from shared encoder
        feat_img, skip_img = self.encoder_image(image)
        feat_dog, _ = self.encoder_dog(dog)
        feat_thresh, _ = self.encoder_thresh(thresh)

        text_feat = self.encoder_text(text_embeds)  # (B, T, D)
        text_feat = text_feat.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        text_feat = text_feat.expand(-1, -1, feat_img.size(2), feat_img.size(3))

        x = torch.cat([feat_img, feat_dog, feat_thresh, text_feat], dim=1)  # (B, 512*3 + 512, H/32, W/32)
        x = self.channel_reduction(x)  # (B, 512, H/32, W/32)
        fused = self.fusion(x)

        # Use skip connections from the image encoder path only
        out = self.decoder(fused, skip_img, size=image.shape[2:])
        return out