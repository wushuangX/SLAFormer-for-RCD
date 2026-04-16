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
    return parser.parse_args()


def get_transform():
    """获取数据预处理变换"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


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


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            output = model(img1, img2)
            loss = criterion(output, label)
            total_loss += loss.item()

    return total_loss / len(dataloader)


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
        isAug=True
    )

    val_dataset = NPYChangeDetectionDataset(
        move='val',
        dataset=args.dataset,
        data_root=args.data_root,
        isAug=False
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

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')

    print('\n数据集加载完成，可以开始训练')
    print('注意: 当前训练循环和模型尚未实现，需要根据SLAFormer架构填充')


if __name__ == '__main__':
    main()
