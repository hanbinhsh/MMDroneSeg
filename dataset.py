import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random


CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def get_dog_images(img):
    blur1 = cv2.GaussianBlur(img, (3, 3), 0)
    blur2 = cv2.GaussianBlur(img, (9, 9), 0)
    dog = cv2.subtract(blur1, blur2)
    return dog

def threshold_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert single channel to three channels
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return thresh_rgb

class VOC2012Dataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform

        image_set_path = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        with open(image_set_path, 'r') as f:
            self.image_ids = f.read().splitlines()

        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(self.root_dir, 'SegmentationClass')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{image_id}.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # 先调整图像尺寸，然后再进行其他处理
        # 使用固定尺寸以保持一致性
        target_size = (384, 512)  # 高度，宽度
        image = image.resize(target_size[::-1], Image.BILINEAR)  # PIL需要(宽度,高度)
        mask = mask.resize(target_size[::-1], Image.NEAREST)  # 对掩码使用最近邻插值以保持标签

        image = np.array(image)
        mask = np.array(mask)

        # 去噪、高斯模糊、直方图均衡
        image = gaussian_blur(image)
        image = histogram_equalization(image)

        # 生成 DoG 差分图
        dog_image = get_dog_images(image)

        # 生成阈值分割图 - 改为三通道
        thresh_image = threshold_segmentation(image)

        # 数据增强（随机水平翻转、旋转）
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            dog_image = np.fliplr(dog_image).copy()
            thresh_image = np.fliplr(thresh_image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.5:
            angle = random.randint(-10, 10)
            image = self.rotate(image, angle)
            dog_image = self.rotate(dog_image, angle)
            thresh_image = self.rotate(thresh_image, angle)
            mask = self.rotate(mask, angle)

        # 随机裁剪（保持图像与标签同步）
        # 随机裁剪为 (320, 480)，保持图像与标签一致
        crop_h, crop_w = 200, 300
        h, w = image.shape[:2]
        if h > crop_h and w > crop_w:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)

            image = image[top:top + crop_h, left:left + crop_w]
            dog_image = dog_image[top:top + crop_h, left:left + crop_w]
            thresh_image = thresh_image[top:top + crop_h, left:left + crop_w]
            mask = mask[top:top + crop_h, left:left + crop_w]
        else:
            # 可选：尺寸不足时中心裁剪或缩放处理（避免出错）
            image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            dog_image = cv2.resize(dog_image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            thresh_image = cv2.resize(thresh_image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        # 处理掩码 - 确保标签在有效范围内
        mask = np.clip(mask, 0, 20)  # 标签范围 [0, 20]
        mask[mask == 255] = 20  # 处理无效值为背景类

        # 归一化 + 转 Tensor
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = preprocess(Image.fromarray(image))
        dog_image = preprocess(Image.fromarray(dog_image))
        thresh_image = preprocess(Image.fromarray(thresh_image))  # 现在是三通道
        mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'dog': dog_image,
            'thresh': thresh_image,
            'mask': mask,
            'filename': f'{image_id}.jpg'
        }

    def rotate(self, img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
        return rotated