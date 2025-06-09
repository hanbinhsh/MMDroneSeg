import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
import torch.nn.functional as F

WIDTH = 480  # 960
HEIGHT = 368  # 736


class CrossAttentionFusion(nn.Module):
    """跨模态注意力融合模块"""

    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, query, key, value):
        B, C, H, W = query.shape

        # Reshape to sequence format
        q = rearrange(query, 'b c h w -> b (h w) c')
        k = rearrange(key, 'b c h w -> b (h w) c')
        v = rearrange(value, 'b c h w -> b (h w) c')

        # Multi-head attention
        q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)

        # Residual connection
        out = self.norm1(out + rearrange(query, 'b c h w -> b (h w) c'))

        # FFN
        out = self.norm2(out + self.ffn(out))

        # Reshape back
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """CBAM注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = CBAM(out_channels)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv_cat(out)
        out = self.attention(out)

        return out


class TextGuidedAttention(nn.Module):
    """文本引导的注意力模块"""

    def __init__(self, visual_dim=512, text_dim=512):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, 1)

        self.attention_conv = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim // 8, 1),
            nn.BatchNorm2d(visual_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(visual_dim // 8, 1, 1),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 3, padding=1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, text_feat):
        B, C, H, W = visual_feat.shape

        # Process text features
        if len(text_feat.shape) == 3:  # (B, T, D)
            text_feat = text_feat.mean(dim=1)  # Global average pooling

        text_proj = self.text_proj(text_feat)  # (B, visual_dim)
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B, visual_dim, H, W)

        # Visual features
        visual_proj = self.visual_proj(visual_feat)

        # Compute attention
        combined = visual_proj * text_proj
        attention_map = self.attention_conv(combined)

        # Apply attention
        attended_visual = visual_feat * attention_map

        # Fusion
        fused = torch.cat([attended_visual, text_proj], dim=1)
        output = self.fusion_conv(fused)

        return output


class VisionTransformerFusion(nn.Module):
    def __init__(self, in_channels=512, embed_dim=256, num_heads=8, num_layers=2, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position embedding
        max_patches = 512
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, embed_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Reconstruction head
        self.reconstruct = nn.ConvTranspose2d(embed_dim, embed_dim,
                                              kernel_size=patch_size, stride=patch_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x_patches = self.patch_embed(x)
        _, _, H_p, W_p = x_patches.shape

        # Flatten patches
        x_patches = rearrange(x_patches, 'b c h w -> b (h w) c')

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_patches = torch.cat([cls_tokens, x_patches], dim=1)

        # Add position embedding
        seq_len = x_patches.shape[1]
        if seq_len > self.pos_embedding.shape[1]:
            pos_embed = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embedding[:, :seq_len, :]

        x_patches = x_patches + pos_embed
        x_patches = self.dropout(x_patches)

        # Transformer
        x_patches = self.transformer(x_patches)

        # Layer norm
        x_patches = self.norm(x_patches)

        # Remove class token and reshape back to spatial format
        x_patches = x_patches[:, 1:, :]
        x_patches = rearrange(x_patches, 'b (h w) c -> b c h w', h=H_p, w=W_p)

        # Reconstruct to original spatial resolution
        x_reconstructed = self.reconstruct(x_patches)

        return x_reconstructed


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, pretrained_path=None):
        super().__init__()
        resnet = models.resnet34(pretrained=True)

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

    def load_pretrained_weights(self, pretrained_path):
        """加载预训练的编码器权重"""
        try:
            pretrained_dict = torch.load(pretrained_path)

            if 'encoder.layer0.0.weight' in pretrained_dict:
                encoder_dict = {}
                for k, v in pretrained_dict.items():
                    if k.startswith('encoder.'):
                        encoder_dict[k[8:]] = v
                pretrained_dict = encoder_dict

            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} layers from pretrained weights.")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")

    def forward(self, x):
        x0 = self.layer0(x)  # 1/4
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        return x4, (x0, x1, x2, x3)


class TextTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(100, d_model))

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pos_embedding[:T, :].unsqueeze(0)
        x = self.transformer(x)
        return x


class FusionTransformer(nn.Module):
    def __init__(self, in_channels=512, embed_dim=256, num_heads=16, num_layers=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(512, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        H_ds, W_ds = x.shape[2], x.shape[3]
        x = self.proj(x)

        x = rearrange(x, 'b c h w -> b (h w) c')
        N = x.shape[1]
        pos = self.pos_embedding[:N, :].unsqueeze(0)
        x = x + pos
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_ds, w=W_ds)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SkipConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = CBAM(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=256, num_classes=21):
        super().__init__()
        # Skip connection blocks with attention
        self.skip3 = SkipConnectionBlock(in_channels + 256, 128)
        self.skip2 = SkipConnectionBlock(128 + 128, 64)
        self.skip1 = SkipConnectionBlock(64 + 64, 32)
        self.skip0 = SkipConnectionBlock(32 + 64, 32)

        # Final decoder layers
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x, skip_features, size):
        x0, x1, x2, x3 = skip_features

        x = self.skip3(x, x3)
        x = self.skip2(x, x2)
        x = self.skip1(x, x1)
        x = self.skip0(x, x0)

        x = self.final(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class MultiModalSegModel(nn.Module):
    def __init__(self, text_dim=512, num_classes=21, pretrained_encoder_path=None):
        super().__init__()
        # 创建独立的编码器而不是共享
        self.encoder_image = ResNetEncoder(pretrained=True, pretrained_path=pretrained_encoder_path)
        self.encoder_dog = ResNetEncoder(pretrained=True, pretrained_path=pretrained_encoder_path)
        self.encoder_thresh = ResNetEncoder(pretrained=True, pretrained_path=pretrained_encoder_path)

        self.encoder_text = TextTransformerEncoder(d_model=text_dim)

        # 多模态特征融合
        self.image_msff = MultiScaleFeatureFusion(512, 512)
        self.dog_msff = MultiScaleFeatureFusion(512, 512)
        self.thresh_msff = MultiScaleFeatureFusion(512, 512)

        # 文本引导注意力
        self.text_guided_attention = TextGuidedAttention(512, text_dim)

        # 跨模态注意力融合
        self.cross_attention1 = CrossAttentionFusion(512, 8)
        self.cross_attention2 = CrossAttentionFusion(512, 8)

        # 特征维度减少
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.fusion = VisionTransformerFusion(in_channels=512, embed_dim=256, num_heads=8, num_layers=2)
        self.decoder = Decoder(in_channels=256, num_classes=num_classes)

    def forward(self, image, dog, thresh, text_embeds):
        B = image.size(0)

        # 获取各模态特征
        feat_img, skip_img = self.encoder_image(image)
        feat_dog, _ = self.encoder_dog(dog)
        feat_thresh, _ = self.encoder_thresh(thresh)

        # 文本特征处理
        text_feat = self.encoder_text(text_embeds)

        # 多尺度特征融合
        feat_img = self.image_msff(feat_img)
        feat_dog = self.dog_msff(feat_dog)
        feat_thresh = self.thresh_msff(feat_thresh)

        # 文本引导的注意力
        feat_img = self.text_guided_attention(feat_img, text_feat)

        # 跨模态注意力融合
        feat_dog = self.cross_attention1(feat_dog, feat_img, feat_img)
        feat_thresh = self.cross_attention2(feat_thresh, feat_img, feat_img)

        # 特征融合
        x = torch.cat([feat_img, feat_dog, feat_thresh], dim=1)
        x = self.channel_reduction(x)

        # Transformer融合
        fused = self.fusion(x)

        # 解码
        out = self.decoder(fused, skip_img, size=image.shape[2:])
        return out