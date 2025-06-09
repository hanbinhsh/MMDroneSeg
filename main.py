import os
import torch
from torch.utils.data import DataLoader
from dataset import VOC2012Dataset
from DroneDataset import DroneDataset
from train import train_model
import json
from datetime import datetime
import torchvision.transforms as transforms

WIDTH =  480       #960
HEIGHT = 368       #736

# 定义一个自定义的Collate函数类，可以在初始化时接收caption_generator
class CustomCollator:
    def __init__(self, config):
        self.config = config

        # 载入文本嵌入文件
        try:
            with open("./clip_embeddings/clip_text_embeddings_drone.json", "r") as f:
                self.text_embeddings = json.load(f)
        except FileNotFoundError:
            print("Warning: clip_text_embeddings.json not found.")
            self.text_embeddings = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch):
        resize_transform = transforms.Resize((HEIGHT, WIDTH))  # height, width
        images, dogs, threshs, masks = [], [], [], []
        text_embeds, filenames = [], []

        for item in batch:
            img = resize_transform(item['image'].unsqueeze(0)).squeeze(0)
            dog = resize_transform(item['dog'].unsqueeze(0)).squeeze(0)
            thresh = resize_transform(item['thresh'].unsqueeze(0)).squeeze(0)
            mask = item['mask'].unsqueeze(0).unsqueeze(0).float()
            mask = torch.nn.functional.interpolate(mask, size=(HEIGHT, WIDTH), mode='nearest')
            mask = mask.squeeze(0).squeeze(0).long()

            images.append(img)
            dogs.append(dog)
            threshs.append(thresh)
            masks.append(mask)

            filename = item['filename']  # 需要 dataset 返回 filename 字段
            filenames.append(filename)

        # 获取文本嵌入
        with torch.no_grad():
            for fname in filenames:
                if fname in self.text_embeddings:
                    text_embed = torch.tensor(self.text_embeddings[fname]).to(self.device)
                else:
                    print(f"Warning: No text embedding found for {fname}, using zero vector.")
                    text_embed = torch.zeros(self.config['text_embed_dim']).to(self.device)
                    exit(255)
                text_embeds.append(text_embed)

        text_embeds = torch.stack(text_embeds)

        text_embeds = text_embeds.squeeze(1)

        # 扩展成序列形式并添加扰动
        B = text_embeds.shape[0]
        seq_len = self.config['max_text_len']
        embed_dim = self.config['text_embed_dim']
        expanded_embeds = text_embeds.unsqueeze(1).expand(B, seq_len, embed_dim)
        noise = torch.randn_like(expanded_embeds) * 0.05
        expanded_embeds = expanded_embeds + noise

        return torch.stack(images), torch.stack(dogs), torch.stack(threshs), expanded_embeds, torch.stack(masks)


def main():
    # 配置参数
    config = {
        'data_root': '../dataset/Drone/classes_dataset/classes_dataset/',
        'batch_size': 24,
        'num_workers': 0,
        'lr': 3e-4,
        'lr_step': 30,
        'lr_gamma': 0.1,
        'epochs':1000,
        'early_stop_patience': 50,
        'num_classes': 5,  # 数据集类别
        'result_dir': f'./results/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'text_embed_dim': 512,  # CLIP文本嵌入维度
        'max_text_len': 20,  # 最大文本长度
        'pretrained_encoder_path': 'pretrained_resnet_encoder.pth',  # 预训练的编码器路径
    }

    # 创建结果保存目录
    os.makedirs(config['result_dir'], exist_ok=True)

    # 保存配置信息
    with open(os.path.join(config['result_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 检查预训练模型路径是否存在
    if 'pretrained_encoder_path' in config and os.path.exists(config['pretrained_encoder_path']):
        print(f"Using pretrained encoder weights from: {config['pretrained_encoder_path']}")
    else:
        print("Warning: Pretrained encoder weights not found, using default pretrained ResNet.")
        config['pretrained_encoder_path'] = None

    # 创建自定义collator
    collator = CustomCollator(config)

    # 创建数据集和数据加载器
    train_dataset = DroneDataset(
        root_dir=config['data_root'],
        image_set='train',
    )

    val_dataset = DroneDataset(
        root_dir=config['data_root'],
        image_set='val',
    )

    # 禁用pin_memory以避免CUDA张量的pin memory错误
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collator,
        pin_memory=False,  # 设置为False避免pin memory错误
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collator,
        pin_memory=False  # 设置为False避免pin memory错误
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"模型将保存到: {config['result_dir']}")

    # 开始训练
    train_model(train_loader, val_loader, config)

    print(f"训练完成! 模型保存在 {os.path.join(config['result_dir'], 'best_model.pth')}")


if __name__ == "__main__":
    main()
