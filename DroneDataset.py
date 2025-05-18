import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import random

WIDTH =  512       #960
HEIGHT = 384       #736

# Drone dataset class mappings
CLASS_NAMES = [
    "obstacles", "water", "soft-surfaces", "moving-objects", "landing-zones"
]

# Color mapping for the drone dataset
CLASS_COLORS = {
    "obstacles": (155, 38, 182),
    "water": (14, 135, 204),
    "soft-surfaces": (124, 252, 0),
    "moving-objects": (255, 20, 147),
    "landing-zones": (169, 169, 169),
}


# Create RGB to class ID mapping with tolerance for color variations
def create_rgb_to_class_id_mapping():
    rgb_to_class_id = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        rgb_to_class_id[CLASS_COLORS[cls_name]] = i
    return rgb_to_class_id


RGB_TO_CLASS_ID = create_rgb_to_class_id_mapping()


def gaussian_blur(img):
    """Apply Gaussian blur to the image"""
    return cv2.GaussianBlur(img, (5, 5), 0)


def histogram_equalization(img):
    """Apply histogram equalization to the image"""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)


def get_dog_images(img):
    """Generate Difference of Gaussian image"""
    blur1 = cv2.GaussianBlur(img, (3, 3), 0)
    blur2 = cv2.GaussianBlur(img, (9, 9), 0)
    dog = cv2.subtract(blur1, blur2)
    return dog


def threshold_segmentation(img):
    """Apply threshold segmentation to the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert single channel to three channels
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return thresh_rgb


def color_to_label(mask_rgb):
    """Convert RGB mask to class ID mask with color tolerance"""
    mask_label = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)

    # For each pixel in the mask
    for cls_name, color in CLASS_COLORS.items():
        # Create a color distance map (Euclidean distance in RGB space)
        color_diff = np.sqrt(np.sum((mask_rgb.astype(np.float32) - np.array(color).astype(np.float32)) ** 2, axis=2))

        # Find pixels where color is close enough to the class color
        color_match = color_diff < 5

        # Assign class ID to matching pixels
        mask_label[color_match] = CLASS_NAMES.index(cls_name)

    return mask_label


class DroneDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset
            image_set (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform

        # Set up paths based on the dataset structure
        if image_set == 'train':
            self.image_dir = os.path.join(self.root_dir, 'original_images')
            self.mask_dir = os.path.join(self.root_dir, 'label_images_semantic')
        else:  # val set
            self.image_dir = os.path.join(self.root_dir, 'val_original')
            self.mask_dir = os.path.join(self.root_dir, 'val_label')

        # Get list of image files
        self.image_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.image_dir)
                                 if f.endswith('.png')])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f'{image_id}.png')
        mask_path = os.path.join(self.mask_dir, f'{image_id}.png')

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # Convert to numpy arrays
        image = np.array(image)
        mask_rgb = np.array(mask)

        # Target size (adjust based on your model requirements)
        target_size = (HEIGHT, WIDTH)  # height, width

        # Resize image and mask to target size
        # IMPORTANT: Resize RGB mask first, then convert to class ID
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        mask_rgb = cv2.resize(mask_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

        # Convert RGB mask to class ID mask AFTER resizing
        mask = color_to_label(mask_rgb)

        # Apply image preprocessing
        # image = gaussian_blur(image) # TODO 先不用模糊
        image = histogram_equalization(image)

        # Generate additional feature images
        dog_image = get_dog_images(image)
        thresh_image = threshold_segmentation(image)

        # Data augmentation for drone imagery (important for aerial views)
        # Apply more aggressive augmentation due to small dataset size
        if self.image_set == 'train':
            # Store the original mask_rgb for augmentation and later conversion
            orig_mask_rgb = mask_rgb.copy()

            # Horizontal flip
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                dog_image = np.fliplr(dog_image).copy()
                thresh_image = np.fliplr(thresh_image).copy()
                mask_rgb = np.fliplr(mask_rgb).copy()

            # Vertical flip (important for aerial/drone images)
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                dog_image = np.flipud(dog_image).copy()
                thresh_image = np.flipud(thresh_image).copy()
                mask_rgb = np.flipud(mask_rgb).copy()

            # Random rotation (more angles for aerial views)
            if random.random() > 0.25:
                angle = random.choice([0, 90, 180, 270])  # 90-degree rotations

                def rotate_90(img, angle):
                    if angle == 0:
                        return img
                    elif angle == 90:
                        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        return cv2.rotate(img, cv2.ROTATE_180)
                    elif angle == 270:
                        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                image = rotate_90(image, angle)
                dog_image = rotate_90(dog_image, angle)
                thresh_image = rotate_90(thresh_image, angle)
                mask_rgb = rotate_90(mask_rgb, angle)

            # Small angle rotation
            if random.random() > 0.5:
                angle = random.randint(-30, 30)

                def rotate(img, angle):
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    if len(img.shape) == 3:  # RGB image
                        return cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
                    else:  # For mask (single channel)
                        return cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

                image = rotate(image, angle)
                dog_image = rotate(dog_image, angle)
                thresh_image = rotate(thresh_image, angle)
                mask_rgb = rotate(mask_rgb, angle)  # Rotate RGB mask

            # Random crop
            crop_h, crop_w = 320, 480
            h, w = image.shape[:2]

            if h > crop_h and w > crop_w:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)

                image = image[top:top + crop_h, left:left + crop_w]
                dog_image = dog_image[top:top + crop_h, left:left + crop_w]
                thresh_image = thresh_image[top:top + crop_h, left:left + crop_w]
                mask_rgb = mask_rgb[top:top + crop_h, left:left + crop_w]
            else:
                # Resize if smaller than crop size
                image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                dog_image = cv2.resize(dog_image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                thresh_image = cv2.resize(thresh_image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                mask_rgb = cv2.resize(mask_rgb, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

            # Color jittering
            if random.random() > 0.5:
                # Brightness adjustment
                factor = random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)

            # Random contrast
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                mean = np.mean(image, axis=(0, 1), keepdims=True)
                image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

            # After all augmentations, convert RGB mask to label
            mask = color_to_label(mask_rgb)

        # Ensure mask has valid class IDs
        mask = np.clip(mask, 0, len(CLASS_NAMES) - 1)

        # Convert to tensor and normalize
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = preprocess(Image.fromarray(image.astype(np.uint8)))
        dog_image = preprocess(Image.fromarray(dog_image.astype(np.uint8)))
        thresh_image = preprocess(Image.fromarray(thresh_image.astype(np.uint8)))
        mask = torch.from_numpy(mask).long()

        # Debug function - print unique class IDs to verify correct labeling
        # unique_classes = np.unique(mask.numpy())
        # print(f"File: {image_id}.png, Unique classes: {unique_classes}")

        return {
            'image': image,
            'dog': dog_image,
            'thresh': thresh_image,
            'mask': mask,
            'filename': f'{image_id}.png'
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def visualize_sample(dataset, idx=0):
        sample = dataset[idx]
        image = sample['image']
        dog_image = sample['dog']
        thresh_image = sample['thresh']
        mask = sample['mask']
        filename = sample['filename']

        print(f"Visualizing sample: {filename}")
        print(f"Image tensor shape: {image.shape}, min: {image.min()}, max: {image.max()}")
        print(f"Mask tensor shape: {mask.shape}, unique values: {torch.unique(mask)}")

        # 正确处理归一化的张量
        def denormalize(tensor):
            """反归一化张量，从ImageNet归一化恢复到[0,1]范围"""
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return tensor * std + mean

        # 将张量转换为可视化的RGB图像
        def tensor_to_rgb(tensor):
            # 首先反归一化
            tensor = denormalize(tensor)
            # 转换为numpy并调整通道顺序
            img = tensor.permute(1, 2, 0).numpy()
            # 裁剪到[0,1]范围并转换为uint8
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            return img

        # 转换图像为可视化格式
        image_rgb = tensor_to_rgb(image)
        dog_rgb = tensor_to_rgb(dog_image)
        thresh_rgb = tensor_to_rgb(thresh_image)

        # 将mask（class id）转为RGB可视化 - 修复版本
        def label_to_color(mask):
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()

            # 打印mask的统计信息以帮助调试
            unique_ids = np.unique(mask)
            print(f"实际存在于mask中的类别ID: {unique_ids}")
            for uid in unique_ids:
                if uid < len(CLASS_NAMES):
                    print(f"类别ID {uid}: {CLASS_NAMES[uid]}, 颜色: {CLASS_COLORS[CLASS_NAMES[uid]]}")

            # 创建彩色掩码
            h, w = mask.shape
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)

            # 对每个类别ID分配颜色
            for cls_id in unique_ids:
                if cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                    color = CLASS_COLORS[cls_name]
                    # 创建该类别的掩码
                    class_mask = (mask == cls_id)
                    # 为掩码中对应位置赋予颜色
                    color_mask[class_mask] = color

            return color_mask

        # 检查通过直接调用color_to_label函数获取的掩码的情况
        # 这可以帮助我们确定问题是在DroneDataset类中还是在可视化函数中
        def debug_original_mask(img_path, mask_path):
            """检查原始掩码文件"""
            print(f"\n调试原始掩码文件: {mask_path}")
            # 加载原始RGB掩码
            mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
            print(f"原始RGB掩码形状: {mask_rgb.shape}, 类型: {mask_rgb.dtype}")

            # 打印一些像素值样本
            print(f"RGB掩码像素样本: {mask_rgb[0:5, 0:5]}")

            # 使用color_to_label函数转换为类别ID掩码
            mask_label = color_to_label(mask_rgb)
            print(f"转换后的类别ID掩码形状: {mask_label.shape}, 类型: {mask_label.dtype}")
            print(f"类别ID掩码中的唯一值: {np.unique(mask_label)}")

            # 转换回彩色显示
            mask_color = np.zeros((mask_label.shape[0], mask_label.shape[1], 3), dtype=np.uint8)
            for cls_id, cls_name in enumerate(CLASS_NAMES):
                mask_color[mask_label == cls_id] = CLASS_COLORS[cls_name]

            return mask_color

        # 获取原始掩码路径并进行调试
        image_id = dataset.image_ids[idx]
        mask_path = os.path.join(dataset.mask_dir, f'{image_id}.png')
        original_mask_color = debug_original_mask(None, mask_path)

        # 使用修复后的函数转换当前的mask
        mask_color = label_to_color(mask)

        # 打印有关mask的信息以进行调试
        print(f"\n最终mask中的唯一值: {np.unique(mask.numpy())}")
        print(f"类别颜色映射示例: {list(CLASS_COLORS.items())[:5]}...")

        # 显示图像
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()

        axs[0].imshow(image_rgb)
        axs[0].set_title(f"Original Image - {filename}")

        axs[1].imshow(dog_rgb)
        axs[1].set_title("DoG Image")

        axs[2].imshow(original_mask_color)
        axs[2].set_title("Original Mask (Debug)")

        axs[3].imshow(mask_color)
        axs[3].set_title("Current Segmentation Mask")

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    # 替换为你数据集的实际路径
    dataset_path = "../dataset/Drone/classes_dataset/classes_dataset/"
    dataset = DroneDataset(root_dir=dataset_path, image_set='val')
    visualize_sample(dataset, idx=0)
