"""
SLAFormer 评估脚本
在测试集上加载检查点，计算指标并可视化分割结果

用法:
    uv run python eval.py \
        --checkpoint checkpoints/checkpoint_best.pth \
        --data_root /path/to/dataset \
        --dataset WRCD \
        --save_dir eval_results \
        --device cuda
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from Dataset import NPYChangeDetectionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='SLAFormer Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径 (.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录绝对路径')
    parser.add_argument('--dataset', type=str, default='WRCD',
                        choices=['WRCD', 'CRCD'],
                        help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='评估设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='eval_results',
                        help='结果保存目录')
    parser.add_argument('--num_vis', type=int, default=20,
                        help='可视化样本数量')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker 数')
    return parser.parse_args()


def get_image_transform():
    return transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x.copy()).float()),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def compute_metrics(preds, targets, threshold=0.5):
    probs = torch.sigmoid(preds)
    pred_binary = (probs >= threshold).float()

    tp = ((pred_binary == 1) & (targets == 1)).sum().float()
    fp = ((pred_binary == 1) & (targets == 0)).sum().float()
    fn = ((pred_binary == 0) & (targets == 1)).sum().float()
    tn = ((pred_binary == 0) & (targets == 0)).sum().float()

    eps = 1e-7

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    intersection_change = tp
    union_change = pred_binary.sum() + targets.sum() - intersection_change
    ciou = intersection_change / (union_change + eps)

    iou_change = intersection_change / (union_change + eps)
    iou_background = tn / (tn + fp + fn + eps)
    miou = (iou_change + iou_background) / 2

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'ciou': iou_change.item(),
        'miou': miou.item(),
        'pred_pos_rate': pred_binary.mean().item(),
        'label_pos_rate': targets.mean().item(),
    }


def compute_batch_iou(pred_binary, targets):
    intersection = ((pred_binary == 1) & (targets == 1)).sum().float()
    union = pred_binary.sum() + targets.sum() - intersection
    iou = intersection / (union + 1e-7)
    return iou.item()


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    out = tensor * std_t + mean_t
    return torch.clamp(out, 0, 1)


def visualize_predictions(model, dataloader, device, save_dir, num_vis=20, threshold=0.5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    count = 0

    with torch.no_grad():
        for batch_idx, (img1, img2, label) in enumerate(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            output = model(img1, img2)
            probs = torch.sigmoid(output)
            preds = (probs >= threshold).float()

            B = img1.shape[0]
            for i in range(B):
                if count >= num_vis:
                    break

                img1_vis = denormalize(img1[i].cpu())
                img2_vis = denormalize(img2[i].cpu())
                gt_mask = label[i].cpu()
                pred_mask = preds[i].cpu()

                h, w = 256, 256
                grid = torch.zeros(3, h, w * 6)

                def to_rgb(t):
                    if t.shape[0] == 1:
                        return t.repeat(3, 1, 1)
                    return t

                grid[:, :, 0:w] = to_rgb(img1_vis)
                grid[:, :, w:2*w] = to_rgb(img2_vis)
                grid[:, :, 2*w:3*w] = to_rgb(gt_mask)
                grid[:, :, 3*w:4*w] = to_rgb(pred_mask)

                diff = torch.abs(pred_mask.float() - gt_mask.float())
                diff_map = torch.zeros(3, h, w)
                diff_map[0] = diff.squeeze(0)
                diff_map[1] = gt_mask.squeeze(0) * 0.5
                diff_map = torch.clamp(diff_map, 0, 1)
                grid[:, :, 4*w:5*w] = diff_map

                overlap = pred_mask * gt_mask
                overlap_map = torch.zeros(3, h, w)
                overlap_map[0] = pred_mask.squeeze(0) * 0.7
                overlap_map[1] = gt_mask.squeeze(0) * 0.7
                overlap_map = torch.clamp(overlap_map, 0, 1)
                grid[:, :, 5*w:6*w] = overlap_map

                import torchvision.utils as vutils
                vutils.save_image(grid,
                                  os.path.join(save_dir, f'sample_{count:04d}.png'),
                                  normalize=False)

                count += 1

            if count >= num_vis:
                break

    print(f'已保存 {count} 张可视化结果到 {save_dir}/')


def evaluate(model, dataloader, device, threshold=0.5, verbose=True):
    model.eval()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_ciou = 0
    total_miou = 0
    n_batches = 0

    criterion = nn.BCEWithLogitsLoss()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (img1, img2, label) in enumerate(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            output = model(img1, img2)
            loss = criterion(output, label)
            total_loss += loss.item()

            probs = torch.sigmoid(output)
            all_probs.append(probs.cpu())
            all_labels.append(label.cpu())

            metrics = compute_metrics(output, label, threshold)
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            total_ciou += metrics['ciou']
            total_miou += metrics['miou']
            n_batches += 1

            if verbose and batch_idx == 0:
                print(f'  [诊断 batch 0] output range: [{output.min().item():.4f}, {output.max().item():.4f}]')
                print(f'  [诊断 batch 0] sigmoid range: [{probs.min().item():.4f}, {probs.max().item():.4f}]')
                print(f'  [诊断 batch 0] label range: [{label.min().item():.4f}, {label.max().item():.4f}]')
                print(f'  [诊断 batch 0] pred_pos_rate: {metrics["pred_pos_rate"]:.4f}')
                print(f'  [诊断 batch 0] label_pos_rate: {metrics["label_pos_rate"]:.4f}')

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if verbose:
        print(f'  [全局诊断] sigmoid 均值: {all_probs.mean().item():.4f}, '
              f'中位数: {all_probs.median().item():.4f}')
        print(f'  [全局诊断] label 均值: {all_labels.mean().item():.4f}')

    n = n_batches
    return {
        'loss': total_loss / n,
        'precision': total_precision / n,
        'recall': total_recall / n,
        'f1': total_f1 / n,
        'ciou': total_ciou / n,
        'miou': total_miou / n,
    }


def main():
    args = parse_args()

    print('=' * 60)
    print('SLAFormer Evaluation')
    print('=' * 60)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Data root:  {args.data_root}')
    print(f'Dataset:    {args.dataset}')
    print(f'Device:     {args.device}')
    print(f'Save dir:   {args.save_dir}')
    print(f'Threshold:  {args.threshold}')
    print('=' * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    test_dataset = NPYChangeDetectionDataset(
        move='test',
        dataset=args.dataset,
        data_root=args.data_root,
        isAug=False,
        transform=get_image_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f'Test samples: {len(test_dataset)}')
    print(f'Test batches:  {len(test_loader)}')

    from SwinT_3dcross import SwinT_FPANet
    model = SwinT_FPANet(img_size=256).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    unexpected_keys = [k for k in state_dict if k not in model_state]
    if unexpected_keys:
        print(f'跳过检查点中的 {len(unexpected_keys)} 个额外键: {unexpected_keys[:5]}...')
    loaded_state = {k: v for k, v in state_dict.items() if k in model_state}
    missing_keys = [k for k in model_state if k not in loaded_state]
    if missing_keys:
        print(f'警告: 检查点中缺少 {len(missing_keys)} 个键: {missing_keys[:5]}...')
    model.load_state_dict(loaded_state, strict=False)
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    print(f'已加载检查点 (epoch={epoch}, val_loss={val_loss})')

    print('\n开始评估测试集...')
    metrics = evaluate(model, test_loader, device, threshold=args.threshold, verbose=True)

    print('\n========== Test Set Metrics ==========')
    print(f'  Loss:      {metrics["loss"]:.4f}')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall:    {metrics["recall"]:.4f}')
    print(f'  F1:        {metrics["f1"]:.4f}')
    print(f'  CIoU:      {metrics["ciou"]:.4f}')
    print(f'  mIoU:      {metrics["miou"]:.4f}')
    print('======================================')

    metrics_path = os.path.join(args.save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'checkpoint: {args.checkpoint}\n')
        f.write(f'epoch: {epoch}\n')
        f.write(f'val_loss: {val_loss}\n')
        for k, v in metrics.items():
            f.write(f'{k}: {v:.6f}\n')
    print(f'\n指标已保存到: {metrics_path}')

    print(f'\n正在生成 {args.num_vis} 张可视化结果...')
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    visualize_predictions(model, test_loader, device, vis_dir,
                         num_vis=args.num_vis, threshold=args.threshold)

    print('\n评估完成!')


if __name__ == '__main__':
    main()
