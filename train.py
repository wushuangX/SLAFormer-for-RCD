"""
SLAFormer 训练脚本
支持命令行参数指定数据集路径

用法:
    uv run python train.py --data_root /path/to/WRCD --dataset WRCD
    uv run python train.py --data_root /absolute/path/to/dataset
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Dataset import NPYChangeDetectionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='SLAFormer Training')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录绝对路径')
    parser.add_argument('--dataset', type=str, default='WRCD',
                        choices=['WRCD', 'CRCD'],
                        help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练，指定检查点路径')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='起始 epoch (从检查点恢复时自动设置)')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值，验证 loss 连续多少个 epoch 不下降则停止')
    return parser.parse_args()


def get_image_transform():
    return transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x.copy()).float()),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_label_transform():
    return transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x.copy()).float()),
    ])


def compute_metrics(preds, targets, threshold=0.5):
    """计算变化检测指标：Precision, Recall, F1, CIoU, mIoU

    Args:
        preds: (B,1,H,W) raw logits
        targets: (B,1,H,W) binary labels {0,1}
        threshold: 分类阈值

    Returns:
        dict with keys: precision, recall, f1, ciou, miou
    """
    probs = torch.sigmoid(preds)
    pred_binary = (probs >= threshold).float()

    B = pred_binary.shape[0]

    tp_change = ((pred_binary == 1) & (targets == 1)).sum().float()
    fp_change = ((pred_binary == 1) & (targets == 0)).sum().float()
    fn_change = ((pred_binary == 0) & (targets == 1)).sum().float()

    tn_change = ((pred_binary == 0) & (targets == 0)).sum().float()
    total_pixels = torch.numel(pred_binary)

    eps = 1e-7

    precision = tp_change / (tp_change + fp_change + eps)
    recall = tp_change / (tp_change + fn_change + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    intersection_change = ((pred_binary == 1) & (targets == 1)).sum().float()
    union_change = ((pred_binary == 1).sum() + (targets == 1).sum() - intersection_change).float()
    ciou = intersection_change / (union_change + eps)

    iou_change = intersection_change / (union_change + eps)
    iou_background = tn_change / (tn_change + fp_change + fn_change + eps)
    miou = (iou_change + iou_background) / 2

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'ciou': iou_change.item(),
        'miou': miou.item(),
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (img1, img2, label) in enumerate(dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """验证/测试通用函数，返回 loss 和指标字典"""
    model.eval()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_ciou = 0
    total_miou = 0
    n_batches = 0

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            output = model(img1, img2)
            loss = criterion(output, label)
            total_loss += loss.item()

            metrics = compute_metrics(output, label)
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            total_ciou += metrics['ciou']
            total_miou += metrics['miou']
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_metrics = {
        'precision': total_precision / n_batches,
        'recall': total_recall / n_batches,
        'f1': total_f1 / n_batches,
        'ciou': total_ciou / n_batches,
        'miou': total_miou / n_batches,
    }
    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device):
    loss, metrics = evaluate(model, dataloader, criterion, device)
    return loss


def test(model, dataloader, criterion, device):
    """在测试集上评估，打印并返回指标"""
    loss, metrics = evaluate(model, dataloader, criterion, device)
    print('\n========== Test Set Results ==========')
    print(f'  Loss:     {loss:.4f}')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall:    {metrics["recall"]:.4f}')
    print(f'  F1:        {metrics["f1"]:.4f}')
    print(f'  CIoU:      {metrics["ciou"]:.4f}')
    print(f'  mIoU:      {metrics["miou"]:.4f}')
    print('======================================')
    return loss, metrics


def save_checkpoint(model, optimizer, epoch, val_loss, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    print(f'  已保存最新检查点: {latest_path}')

    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f'  已保存最佳检查点: {best_path}')


def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f'检查点不存在: {checkpoint_path}')
        return 0, float('inf')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', float('inf'))

    print(f'已从检查点恢复: epoch {epoch}, val_loss {val_loss:.4f}')
    return epoch, val_loss


def main():
    args = parse_args()

    print('=' * 60)
    print('SLAFormer Training')
    print('=' * 60)
    print(f'Data root: {args.data_root}')
    print(f'Dataset: {args.dataset}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning rate: {args.lr}')
    print(f'Device: {args.device}')
    print('=' * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = NPYChangeDetectionDataset(
        move='train',
        dataset=args.dataset,
        data_root=args.data_root,
        isAug=True,
        transform=get_image_transform()
    )

    val_dataset = NPYChangeDetectionDataset(
        move='val',
        dataset=args.dataset,
        data_root=args.data_root,
        isAug=False,
        transform=get_image_transform()
    )

    test_dataset = NPYChangeDetectionDataset(
        move='test',
        dataset=args.dataset,
        data_root=args.data_root,
        isAug=False,
        transform=get_image_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples:   {len(val_dataset)}')
    print(f'Test samples:  {len(test_dataset)}')
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches:   {len(val_loader)}')
    print(f'Test batches:  {len(test_loader)}')

    print('\n数据集加载完成，开始训练')

    from SwinT_3dcross import SwinT_FPANet

    model = SwinT_FPANet(img_size=256).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = args.start_epoch
    best_val_loss = float('inf')

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.resume, device)

    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f'TensorBoard 日志目录: {log_dir}')
    print(f'运行 "tensorboard --logdir logs/" 查看训练曲线')

    print(f'\n开始训练 from epoch {start_epoch}...')
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Metrics/val_precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/val_recall', val_metrics['recall'], epoch)
        writer.add_scalar('Metrics/val_f1', val_metrics['f1'], epoch)
        writer.add_scalar('Metrics/val_ciou', val_metrics['ciou'], epoch)
        writer.add_scalar('Metrics/val_miou', val_metrics['miou'], epoch)

        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val   Loss: {val_loss:.4f}  '
              f'Pre: {val_metrics["precision"]:.4f}  '
              f'Rec: {val_metrics["recall"]:.4f}  '
              f'F1: {val_metrics["f1"]:.4f}  '
              f'CIoU: {val_metrics["ciou"]:.4f}  '
              f'mIoU: {val_metrics["miou"]:.4f}')

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'  New best model! Val Loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'  No improvement, patience: {patience_counter}/{args.patience}')

        save_checkpoint(model, optimizer, epoch+1, val_loss, args.save_dir, is_best=is_best)

        if patience_counter >= args.patience:
            print(f'\n早停触发! 连续 {patience_counter} 个 epoch 验证 loss 未下降')
            break

    print('\n训练完成! 加载最佳模型在测试集上评估...')
    best_path = os.path.join(args.save_dir, 'checkpoint_best.pth')
    if os.path.exists(best_path):
        load_checkpoint(model, None, best_path, device)
        test(model, test_loader, criterion, device)
    else:
        print('未找到最佳检查点，跳过测试')

    writer.close()
    print(f'\n训练完成! Best Val Loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
