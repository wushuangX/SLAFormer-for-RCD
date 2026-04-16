"""
数据集路径配置文件
支持通过命令行参数覆盖
"""

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='SLAFormer Training')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录绝对路径 (会覆盖config中的设置)')
    parser.add_argument('--dataset', type=str, default='WRCD',
                        choices=['WRCD', 'CRCD'],
                        help='数据集名称')
    return parser.parse_known_args()[0]

CLI_ARGS = parse_args()

DATASET_PATHS = {
    'WRCD': {
        'root': 'data/Wuhan_Change_Detection_Patched.nosync',
        'format': 'npy',
        'train_A': '/train/A',
        'train_B': '/train/B',
        'train_label': '/train/label',
        'test_A': '/test/A',
        'test_B': '/test/B',
        'test_label': '/test/label',
        'val_A': '/val/A',
        'val_B': '/val/B',
        'val_label': '/val/label',
    },
    'CRCD': {
        'root': '/path/to/your/CRCD/dataset',
        'format': 'npy',
        'train_A': '/train/A',
        'train_B': '/train/B',
        'train_label': '/train/label',
        'test_A': '/test/A',
        'test_B': '/test/B',
        'test_label': '/test/label',
    },
}

DEFAULT_DATASET = 'WRCD'

def get_data_root():
    """获取数据集根目录，CLI参数优先"""
    if CLI_ARGS.data_root:
        return CLI_ARGS.data_root
    return DATASET_PATHS[CLI_ARGS.dataset if CLI_ARGS.dataset else DEFAULT_DATASET]['root']
