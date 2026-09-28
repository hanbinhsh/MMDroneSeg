import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def threshold_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return thresh_rgb

def process_image(image_path):
    # 读取图像并转换为RGB格式
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 可选预处理
    img_blurred = gaussian_blur(img_rgb)
    img_equalized = histogram_equalization(img_blurred)

    # 阈值分割
    segmented = threshold_segmentation(img_equalized)

    # 显示图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Threshold Segmentation")
    plt.imshow(segmented)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply threshold segmentation to an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    process_image(args.image_path)
