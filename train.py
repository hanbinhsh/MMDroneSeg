import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MultiModalSegModel
from utils import compute_metrics  # 假设你已实现F1、IoU等指标计算函数
from utils import DiceLoss


def train_model(train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用预训练的编码器创建模型
    model = MultiModalSegModel(
        num_classes=config['num_classes'],
        pretrained_encoder_path=config.get('pretrained_encoder_path', None)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])

    best_val_miou = 0.0
    early_stop_counter = 0
    result_file = os.path.join(config['result_dir'], 'result.txt')
    os.makedirs(config['result_dir'], exist_ok=True)

    # class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    #                "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    #                "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    class_names = [
        "obstacles", "water", "soft-surfaces", "moving-objects", "landing-zones"
    ]

    with open(result_file, 'w') as f:
        f.write('Epoch\tTrainLoss\tTrainPixAcc\tTrainmIoU\tTrainF1\tValLoss\tValPixAcc\tValmIoU\tValF1\n')

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Train {epoch+1}/{config['epochs']}"):
            image, dog, thresh, text, label = [b.to(device) for b in batch]
            optimizer.zero_grad()
            output = model(image, dog, thresh, text)
            loss = criterion(output, label) + dice_loss(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct += (pred == label).sum().item()
            total += torch.numel(label)
            all_preds.append(pred.detach().cpu())
            all_labels.append(label.detach().cpu())

        train_pixacc = correct / total
        train_f1, train_miou, _ = compute_metrics(all_preds, all_labels, config['num_classes'])

        print(f"[ Train | {epoch + 1:03d} ] loss = {train_loss:.5f}, acc = {train_pixacc:.5f}, f1 = {train_f1:.5f}, miou = {train_miou:.5f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch+1}/{config['epochs']}"):
                image, dog, thresh, text, label = [b.to(device) for b in batch]
                output = model(image, dog, thresh, text)
                loss = criterion(output, label) + dice_loss(output, label)
                val_loss += loss.item()

                pred = torch.argmax(output, dim=1)
                correct += (pred == label).sum().item()
                total += torch.numel(label)
                val_preds.append(pred.detach().cpu())
                val_labels.append(label.detach().cpu())

        val_pixacc = correct / total
        val_f1, val_miou, class_iou = compute_metrics(val_preds, val_labels, config['num_classes'])

        print(f"[ Valid | {epoch + 1:03d} ] loss = {val_loss:.5f}, acc = {val_pixacc:.5f}, f1 = {val_f1:.5f}, mIoU = {val_miou:.5f}")

        print("Per-class IoU:")
        for i, name in enumerate(class_names):
            print(f"  {name}: {class_iou[i]:.4f}")

        with open(result_file, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_pixacc:.4f}\t{train_miou:.4f}\t{train_f1:.4f}\t"
                    f"{val_loss:.4f}\t{val_pixacc:.4f}\t{val_miou:.4f}\t{val_f1:.4f}\n")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            print(f"Get miou higher, saving model.")
            torch.save(model.state_dict(), os.path.join(config['result_dir'], 'best_model.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config['early_stop_patience']:
            print(f"Early stopping at epoch {epoch+1}.")
            break

        scheduler.step()