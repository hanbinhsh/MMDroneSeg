import os
import torch
import argparse
import numpy as np
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import MultiModalSegModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

WIDTH =  480  # 960
HEIGHT = 368  # 736

# CLASS_NAMES = [
#     "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
#     "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
#     "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]

CLASS_NAMES = [
    "obstacles", "water", "soft-surfaces", "moving-objects", "landing-zones"
]

# 添加指定的颜色映射
CLASS_COLORS = {
    "obstacles": (155, 38, 182),  # 紫色
    "water": (14, 135, 204),  # 蓝色
    "soft-surfaces": (124, 252, 0),  # 亮绿色
    "moving-objects": (255, 20, 147),  # 粉红色
    "landing-zones": (169, 169, 169),  # 灰色
}


def load_model(model_path, num_classes=5, device='cuda', pretrained_encoder_path=None):
    """Load the trained segmentation model."""
    model = MultiModalSegModel(num_classes=num_classes, pretrained_encoder_path=pretrained_encoder_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_caption(image, device='cuda'):
    """Generate caption for the input image using BLIP."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


def generate_text_embedding(caption, device='cuda'):
    """Generate CLIP text embedding from the caption."""
    clip_model, _ = clip.load("ViT-B/32", device=device)

    text_token = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_token)

    return text_embedding


def preprocess_image(image_path):
    """Preprocess the input image for the model."""
    # Load image
    image = Image.open(image_path).convert('RGB')

    # 保存原始尺寸用于结果显示
    original_size = image.size  # (宽度,高度)
    original_image = np.array(image)

    # 调整到模型期望的输入尺寸
    target_size = (WIDTH, HEIGHT)  # (宽度,高度)
    image = image.resize(target_size, Image.BILINEAR)
    image_np = np.array(image)

    # Apply preprocessing steps from dataset.py
    # Gaussian blur
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Histogram equalization
    img_yuv = cv2.cvtColor(blurred, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Generate DoG (Difference of Gaussians)
    blur1 = cv2.GaussianBlur(image_np, (3, 3), 0)
    blur2 = cv2.GaussianBlur(image_np, (9, 9), 0)
    dog = cv2.subtract(blur1, blur2)

    # Generate threshold segmentation
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Convert to tensors with normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(Image.fromarray(equalized))
    dog_tensor = preprocess(Image.fromarray(dog))
    thresh_tensor = preprocess(Image.fromarray(thresh_rgb))

    # Resize to model input size (384, 512)
    resize_transform = transforms.Resize((384, 512))
    image_tensor = resize_transform(image_tensor.unsqueeze(0)).squeeze(0)
    dog_tensor = resize_transform(dog_tensor.unsqueeze(0)).squeeze(0)
    thresh_tensor = resize_transform(thresh_tensor.unsqueeze(0)).squeeze(0)

    return image_tensor, dog_tensor, thresh_tensor, Image.fromarray(original_image), original_size


def predict(model, image_path, device='cuda'):
    """Make segmentation prediction for a single image."""
    # Preprocess image
    image_tensor, dog_tensor, thresh_tensor, original_image, original_size = preprocess_image(image_path)

    # Generate caption
    caption = generate_caption(original_image, device)
    print(f"Generated caption: {caption}")

    # Generate text embedding
    text_embedding = generate_text_embedding(caption, device)

    # Process text embedding to match model input format
    # Expand to sequence length and add noise as in CustomCollator
    B = 1
    seq_len = 32  # From config['max_text_len']
    embed_dim = 512  # From config['text_embed_dim']
    expanded_embeds = text_embedding.unsqueeze(1).expand(B, seq_len, embed_dim)
    noise = torch.randn_like(expanded_embeds) * 0.05
    expanded_embeds = expanded_embeds + noise

    # Add batch dimension to tensors
    image_tensor = image_tensor.unsqueeze(0).to(device)
    dog_tensor = dog_tensor.unsqueeze(0).to(device)
    thresh_tensor = thresh_tensor.unsqueeze(0).to(device)
    expanded_embeds = expanded_embeds.to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor, dog_tensor, thresh_tensor, expanded_embeds)

        # 确保预测结果调整回原始图像尺寸
        if output.shape[2:] != (original_image.height, original_image.width):
            output = torch.nn.functional.interpolate(
                output,
                size=(original_image.height, original_image.width),
                mode='bilinear',
                align_corners=True
            )

        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    return prediction, original_image


def visualize_prediction(prediction, original_image, output_path=None):
    """Visualize the segmentation prediction."""
    # 使用指定的颜色映射而不是自动生成的颜色
    h, w = prediction.shape
    rgb_pred = np.zeros((h, w, 3), dtype=np.uint8)

    # 创建图例元素
    legend_elements = []

    # 找出预测中的唯一类别
    unique_classes = np.unique(prediction)

    # 为每个类别分配颜色
    for i in unique_classes:
        if i < len(CLASS_NAMES):  # 确保类别索引有效
            class_name = CLASS_NAMES[i]
            if class_name in CLASS_COLORS:
                # 从BGR格式转为RGB格式（因为OpenCV使用BGR）
                color_bgr = CLASS_COLORS[class_name]
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

                # 应用颜色到掩码
                mask = prediction == i
                rgb_pred[mask] = color_bgr

                # 为matplotlib图例创建RGB颜色
                color_norm = [c / 255.0 for c in color_bgr]
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color_norm, label=class_name))

    # 创建可视化
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_pred)
    plt.title('Segmentation Prediction')
    plt.axis('off')

    # 添加图例
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=min(5, len(unique_classes)))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    return rgb_pred


def main():
    parser = argparse.ArgumentParser(description='Predict segmentation for a single image')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on (cuda/cpu)')
    parser.add_argument('--pep', type=str, default='pretrained_resnet_encoder.pth',
                        help='Pretrained Encoder Path')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load model
    print("Loading model...")
    model = load_model(args.model, device=args.device, pretrained_encoder_path=args.pep)

    # Make prediction
    print(f"Processing image: {args.image}")
    prediction, original_image = predict(model, args.image, device=args.device)

    # Count classes
    unique_classes, counts = np.unique(prediction, return_counts=True)
    print("Classes detected:")
    for cls, count in zip(unique_classes, counts):
        if cls < len(CLASS_NAMES):
            percentage = (count / prediction.size) * 100
            print(f"  {CLASS_NAMES[cls]}: {percentage:.2f}% ({count} pixels)")

    # Visualize prediction
    print("Visualizing prediction...")
    visualize_prediction(prediction, original_image, args.output)

    print("Done!")


if __name__ == '__main__':
    main()