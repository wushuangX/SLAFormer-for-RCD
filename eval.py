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


def make_grid_row(img1_t, img2_t, gt, pred, alpha=0.5):
    """将3张图拼接为一个 (3, H, 3W) 的可视化张量"""
    h, w = img1_t.shape[1:]
    cells = []

    def to_rgb(t):
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t

    cells.append(to_rgb(img1_t))
    cells.append(to_rgb(img2_t))
    cells.append(to_rgb(gt))

    pred_color = torch.zeros(3, h, w)
    pred_color[0] = pred.squeeze(0)
    gt_color = torch.zeros(3, h, w)
    gt_color[1] = gt.squeeze(0)

    cells.append(to_rgb(pred))
    cells.append(to_rgb(gt))

    diff = torch.abs(pred.float() - gt.float())
    diff_color = diff.repeat(3, 1, 1) * 3
    cells.append(torch.clamp(diff_color, 0, 1))

    row = torch.cat(cells, dim=2)
    return row


def visualize_predictions(model, dataloader, device, save_dir, num_vis=20, threshold=0.5):
    """在测试集上推理并保存可视化结果"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    all_metrics = []

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

                idx_in_batch = i
                vis_row = []

                img1_vis = denormalize(img1[i].cpu())
                img2_vis = denormalize(img2[i].cpu())

                gt_mask = label[i].cpu()
                pred_mask = preds[i].cpu()

                combined = torch.zeros(3, 256, 256 * 6)

                combined[:, :, 0:256] = img1_vis
                combined[:, :, 256:512] = img2_vis
                combined[:, :, 512:768] = gt_mask.repeat(3, 1, 1) if gt_mask.shape[0] == 1 else gt_mask
                combined[:, :, 768:1024] = pred_mask.repeat(3, 1, 1) if pred_mask.shape[0] == 1 else pred_mask

                overlap = pred_mask * gt_mask
                diff_map = torch.zeros(3, 256, 256)
                diff_map[0] = pred_mask.squeeze(0)
                diff_map[1] = gt_mask.squeeze(0)
                diff_map = torch.clamp(diff_map, 0, 1)
                combined[:, :, 1024:1280] = diff_map

                import torchvision.utils as vutils
                vutils.save_image(combined,
                                  os.path.join(save_dir, f'sample_{count:04d}.png'),
                                  nrow=1, normalize=False)

                batch_iou = compute_batch_iou(preds[i:i+1], label[i:i+1])
                all_metrics.append(batch_iou)
                count += 1

            if count >= num_vis:
                break

    print(f'已保存 {count} 张可视化结果到 {save_dir}/')


def evaluate(model, dataloader, device, threshold=0.5):
    """在测试集上评估，返回指标"""
    model.eval()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_ciou = 0
    total_miou = 0
    n_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            output = model(img1, img2)
            loss = criterion(output, label)
            total_loss += loss.item()

            metrics = compute_metrics(output, label, threshold)
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            total_ciou += metrics['ciou']
            total_miou += metrics['miou']
            n_batches += 1

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
    print('=' * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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
    model.load_state_dict(loaded_state, strict=False)
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    print(f'已加载检查点 (epoch={epoch}, val_loss={val_loss})')

    print('\n开始评估测试集...')
    metrics = evaluate(model, test_loader, device, threshold=args.threshold)

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
        for k, v in metrics.items():
            f.write(f'{k}: {v:.4f}\n')
    print(f'\n指标已保存到: {metrics_path}')

    print(f'\n正在生成 {args.num_vis} 张可视化结果...')
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    visualize_predictions(model, test_loader, device, vis_dir,
                         num_vis=args.num_vis, threshold=args.threshold)

    print('\n评估完成!')


if __name__ == '__main__':
    main()
