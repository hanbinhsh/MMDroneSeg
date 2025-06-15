import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MultiModalSegModel
from utils import compute_metrics
from utils import DiceLoss, FocalLoss, EdgeLoss


def train_model(train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用预训练的编码器创建模型
    model = MultiModalSegModel(
        num_classes=config['num_classes'],
        pretrained_encoder_path=config.get('pretrained_encoder_path', None)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    # Focal loss
    freq = np.array([0.0868, 0.233, 0.3438, 0.0171, 0.529])
    # 1. 倒数归一化
    # alpha = 1.0 / (freq + 1e-6)
    # alpha = alpha / alpha.sum()  # 归一化成概率分布
    # 2. 1-freq
    alpha = 1.0 - freq
    alpha = alpha / alpha.sum()
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    focal_loss = FocalLoss(gamma=2.0, alpha=alpha_tensor)

    # Edge loss
    edge_loss_fn = EdgeLoss(mode='l1').to(device)  # 或 'bce'

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    best_val_miou = 0.0
    early_stop_counter = 0
    result_file = os.path.join(config['result_dir'], 'result.txt')
    os.makedirs(config['result_dir'], exist_ok=True)

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
            # loss = criterion(output, label) + dice_loss(output, label)
            loss = criterion(output, label) + focal_loss(output, label)
            if epoch + 1 >= config['edge_start_epoch']:
                edge_loss = edge_loss_fn(output, label)
                loss += config['edge_weight'] * edge_loss

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
                # loss = criterion(output, label) + dice_loss(output, label)
                loss = criterion(output, label) + focal_loss(output, label)
                if epoch + 1 >= config['edge_start_epoch']:
                    edge_loss = edge_loss_fn(output, label)
                    loss += config['edge_weight'] * edge_loss

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
        # scheduler.step(val_loss)