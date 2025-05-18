import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MultiModalSegModel
from utils import compute_metrics, DiceLoss
from utils import EdgeLoss, BoundaryLoss


def train_model(train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiModalSegModel(num_classes=config['num_classes']).to(device)

    # Standard losses
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    # Edge-aware loss - choose one based on your requirements
    if config.get('edge_loss_type', 'canny') == 'canny':
        edge_loss = EdgeLoss(
            low_threshold=config.get('canny_low_threshold', 0.1),
            high_threshold=config.get('canny_high_threshold', 0.3),
            edge_weight=config.get('edge_loss_weight', 1.0),
            sigma=config.get('canny_sigma', 1.0)
        ).to(device)
    else:
        edge_loss = BoundaryLoss(
            weight=config.get('edge_loss_weight', 1.0)
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])

    best_val_miou = 0.0
    early_stop_counter = 0
    result_file = os.path.join(config['result_dir'], 'result.txt')
    os.makedirs(config['result_dir'], exist_ok=True)

    class_names = [
        "obstacles", "water", "soft-surfaces", "moving-objects", "landing-zones"
    ]

    with open(result_file, 'w') as f:
        f.write(
            'Epoch\tTrainLoss\tTrainCELoss\tTrainDiceLoss\tTrainEdgeLoss\tTrainPixAcc\tTrainmIoU\tTrainF1\tValLoss\tValPixAcc\tValmIoU\tValF1\n')

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_ce_loss = 0.0
        train_dice_loss_val = 0.0
        train_edge_loss_val = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Train {epoch + 1}/{config['epochs']}"):
            image, dog, thresh, text, label = [b.to(device) for b in batch]
            optimizer.zero_grad()
            output = model(image, dog, thresh, text)

            # Calculate individual loss components
            ce_loss = criterion(output, label)
            dice_loss_val = dice_loss(output, label)
            edge_loss_val = edge_loss(output, label)

            # Combine losses with configurable weights
            ce_weight = config.get('ce_loss_weight', 1.0)
            dice_weight = config.get('dice_loss_weight', 1.0)
            edge_weight = config.get('edge_loss_weight', 1.0)

            loss = ce_weight * ce_loss + dice_weight * dice_loss_val + edge_weight * edge_loss_val

            loss.backward()
            optimizer.step()

            # Track losses
            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            train_dice_loss_val += dice_loss_val.item()
            train_edge_loss_val += edge_loss_val.item()

            # Calculate metrics
            pred = torch.argmax(output, dim=1)
            correct += (pred == label).sum().item()
            total += torch.numel(label)
            all_preds.append(pred.detach().cpu())
            all_labels.append(label.detach().cpu())

        train_pixacc = correct / total
        train_f1, train_miou, _ = compute_metrics(all_preds, all_labels, config['num_classes'])

        # Normalize losses by number of batches
        num_batches = len(train_loader)
        train_loss /= num_batches
        train_ce_loss /= num_batches
        train_dice_loss_val /= num_batches
        train_edge_loss_val /= num_batches

        print(f"[ Train | {epoch + 1:03d} ] "
              f"loss = {train_loss:.5f}, CE = {train_ce_loss:.5f}, "
              f"Dice = {train_dice_loss_val:.5f}, Edge = {train_edge_loss_val:.5f}, "
              f"acc = {train_pixacc:.5f}, f1 = {train_f1:.5f}, miou = {train_miou:.5f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_ce_loss = 0.0
        val_dice_loss_val = 0.0
        val_edge_loss_val = 0.0
        correct = 0
        total = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch + 1}/{config['epochs']}"):
                image, dog, thresh, text, label = [b.to(device) for b in batch]
                output = model(image, dog, thresh, text)

                # Calculate loss components
                ce_loss = criterion(output, label)
                dice_loss_val = dice_loss(output, label)
                edge_loss_val = edge_loss(output, label)

                # Combined loss
                loss = ce_weight * ce_loss + dice_weight * dice_loss_val + edge_weight * edge_loss_val

                val_loss += loss.item()
                val_ce_loss += ce_loss.item()
                val_dice_loss_val += dice_loss_val.item()
                val_edge_loss_val += edge_loss_val.item()

                pred = torch.argmax(output, dim=1)
                correct += (pred == label).sum().item()
                total += torch.numel(label)
                val_preds.append(pred.detach().cpu())
                val_labels.append(label.detach().cpu())

        # Normalize validation losses
        val_loss /= len(val_loader)
        val_ce_loss /= len(val_loader)
        val_dice_loss_val /= len(val_loader)
        val_edge_loss_val /= len(val_loader)

        val_pixacc = correct / total
        val_f1, val_miou, class_iou = compute_metrics(val_preds, val_labels, config['num_classes'])

        print(f"[ Valid | {epoch + 1:03d} ] "
              f"loss = {val_loss:.5f}, CE = {val_ce_loss:.5f}, "
              f"Dice = {val_dice_loss_val:.5f}, Edge = {val_edge_loss_val:.5f}, "
              f"acc = {val_pixacc:.5f}, f1 = {val_f1:.5f}, mIoU = {val_miou:.5f}")

        print("Per-class IoU:")
        for i, name in enumerate(class_names):
            print(f"  {name}: {class_iou[i]:.4f}")

        with open(result_file, 'a') as f:
            f.write(
                f"{epoch + 1}\t{train_loss:.4f}\t{train_ce_loss:.4f}\t{train_dice_loss_val:.4f}\t{train_edge_loss_val:.4f}\t"
                f"{train_pixacc:.4f}\t{train_miou:.4f}\t{train_f1:.4f}\t"
                f"{val_loss:.4f}\t{val_pixacc:.4f}\t{val_miou:.4f}\t{val_f1:.4f}\n")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            print(f"Get miou higher, saving model.")
            torch.save(model.state_dict(), os.path.join(config['result_dir'], 'best_model.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config['early_stop_patience']:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

        scheduler.step()