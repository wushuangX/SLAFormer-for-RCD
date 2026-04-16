import os
import cv2
import numpy as np
import torch
import torch.utils.data
import random

from config import DATASET_PATHS, DEFAULT_DATASET


def read_directory(directory_name, label=False):
    array_of_img = []
    files = os.listdir(directory_name)
    files.sort(key=lambda x: int(x[0:-4]))
    for filename in files:
        img = cv2.imread(directory_name + "/" + filename)
        if label:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        array_of_img.append(img)
    return array_of_img


def read_npy_directory(directory_name, label=False):
    """Read numpy (.npy) files from directory."""
    array_of_data = []
    files = os.listdir(directory_name)
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in files:
        data = np.load(os.path.join(directory_name, filename))
        if label:
            if len(data.shape) == 3 and data.shape[0] == 3:
                data = data[0]
            data = (data > 0).astype(np.uint8)
        array_of_data.append(data)
    return array_of_data


dataset_LEVIR = 'CD_dataset/LEVIR'
dataset_WHU = 'CD_dataset/WHU'
dataset_PRCV = 'CD_dataset/SemiCD'
dataset_MTWHU = 'CD_dataset/MTWHU'
dataset_MTOSCDOS = 'CD_dataset/MTOSCDOS'
dataset_MTOSCDSO = 'CD_dataset/MTOSCDSO'
dataset_CAU = 'CD_dataset/CAU'

def get_dataset_paths(dataset_name):
    """Get sub-path configuration for a dataset."""
    if dataset_name in DATASET_PATHS:
        paths = DATASET_PATHS[dataset_name]
        return {
            'root': paths['root'],
            'format': paths.get('format', 'image'),
            'train_1': paths['train_A'],
            'train_2': paths['train_B'],
            'train_label': paths['train_label'],
            'test_1': paths['test_A'],
            'test_2': paths['test_B'],
            'test_label': paths['test_label'],
            'val_1': paths.get('val_A'),
            'val_2': paths.get('val_B'),
            'val_label': paths.get('val_label'),
        }
    return None


dataset_WRCD = DATASET_PATHS['WRCD']['root']
dataset_CRCD = DATASET_PATHS['CRCD']['root']








class LevirWhuGzDataset(torch.utils.data.Dataset):
    def __init__(self, move='train', dataset='WRCD', transform=None, isAug=False, isSwinT=False):
        super(LevirWhuGzDataset, self).__init__()
        seq_img_1 = []
        seq_img_2 = []
        seq_label = []
        self.isaug = isAug
        self.isSwinT = isSwinT
        self.move = move

        paths = get_dataset_paths(dataset)
        dataset_root = paths['root'] if paths else None

        if dataset == 'LEVIR':
            root = dataset_LEVIR
            if move == 'train':
                seq_img_1 = read_directory(root + paths['train_1'])
                seq_img_2 = read_directory(root + paths['train_2'])
                seq_label = read_directory(root + paths['train_label'], label=True)
            elif move == 'test':
                seq_img_1 = read_directory(root + paths['test_1'])
                seq_img_2 = read_directory(root + paths['test_2'])
                seq_label = read_directory(root + paths['test_label'], label=True)
        elif dataset == 'WHU':
            root = dataset_WHU
            if move == 'train':
                seq_img_1 = read_directory(root + paths['train_1'])
                seq_img_2 = read_directory(root + paths['train_2'])
                seq_label = read_directory(root + paths['train_label'], label=True)
            elif move == 'test':
                seq_img_1 = read_directory(root + paths['test_1'])
                seq_img_2 = read_directory(root + paths['test_2'])
                seq_label = read_directory(root + paths['test_label'], label=True)
        elif dataset == 'Gz':
            root = dataset_PRCV
            if move == 'train':
                seq_img_1 = read_directory(root + paths['train_1'])
                seq_img_2 = read_directory(root + paths['train_2'])
                seq_label = read_directory(root + paths['train_label'], label=True)
            elif move == 'test':
                seq_img_1 = read_directory(root + paths['test_1'])
                seq_img_2 = read_directory(root + paths['test_2'])
                seq_label = read_directory(root + paths['test_label'], label=True)
        elif dataset == 'MTWHU':
            root = dataset_MTWHU
            if move == 'train':
                seq_img_1 = read_directory(root + paths['train_1'])
                seq_img_2 = read_directory(root + paths['train_2'])
                seq_label = read_directory(root + paths['train_label'], label=True)
            elif move == 'test':
                seq_img_1 = read_directory(root + paths['test_1'])
                seq_img_2 = read_directory(root + paths['test_2'])
                seq_label = read_directory(root + paths['test_label'], label=True)
        elif dataset == 'CAU':
            root = dataset_CAU
            if move == 'train':
                seq_img_1 = read_directory(root + paths['train_1'])
                seq_img_2 = read_directory(root + paths['train_2'])
                seq_label = read_directory(root + paths['train_label'], label=True)
            elif move == 'test':
                seq_img_1 = read_directory(root + paths['test_1'])
                seq_img_2 = read_directory(root + paths['test_2'])
                seq_label = read_directory(root + paths['test_label'], label=True)
        elif dataset == 'CRCD':
            root = dataset_CRCD
            if move == 'train':
                seq_img_1 = read_directory(root + paths['train_1'])
                seq_img_2 = read_directory(root + paths['train_2'])
                seq_label = read_directory(root + paths['train_label'], label=True)
            elif move == 'test':
                seq_img_1 = read_directory(root + paths['test_1'])
                seq_img_2 = read_directory(root + paths['test_2'])
                seq_label = read_directory(root + paths['test_label'], label=True)
        elif dataset == 'WRCD':
            root = dataset_WRCD
            if move == 'train':
                seq_img_1 = read_directory(root + paths['train_1'])
                seq_img_2 = read_directory(root + paths['train_2'])
                seq_label = read_directory(root + paths['train_label'], label=True)
            elif move == 'test':
                seq_img_1 = read_directory(root + paths['test_1'])
                seq_img_2 = read_directory(root + paths['test_2'])
                seq_label = read_directory(root + paths['test_label'], label=True)

        self.seq_img_1 = seq_img_1
        self.seq_img_2 = seq_img_2
        self.seq_label = seq_label
        self.transform = transform

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        imgs_1 = self.seq_img_1[index]
        imgs_2 = self.seq_img_2[index]
        label = self.seq_label[index]

        # 随机进行数据增强，为2时不处理
        if self.isaug:
            flipCote = random.choice([-1, 0, 1, 2])
            if flipCote != 2:
                imgs_1 = self.augment(imgs_1, flipCote)
                imgs_2 = self.augment(imgs_2, flipCote)
                label = self.augment(label, flipCote)
                label = label.reshape(label.shape[0], label.shape[1], 1)
        h, w = label.shape[0], label.shape[1]
        if self.isSwinT:
            label_levels = [int(h / 8), int(h / 16), int(h / 32), int(h / 32)]
        else:
            label_levels = [int(h / 4), int(h / 4), int(h / 16), int(h / 16)]



        if self.move == 'train':
            train_masks = []
            if self.transform is not None:
                imgs_1 = self.transform(imgs_1)
                imgs_2 = self.transform(imgs_2)
                label = self.transform(label)

            return imgs_1, imgs_2, label
        else:
            test_masks = []
            if self.transform is not None:
                imgs_1 = self.transform(imgs_1)
                imgs_2 = self.transform(imgs_2)
                label = self.transform(label)

            return imgs_1, imgs_2, label

    def __len__(self):
        return len(self.seq_label)


class NPYChangeDetectionDataset(torch.utils.data.Dataset):
    """NPY格式变化检测数据集 - 用于加载预处理好的numpy patches"""

    def __init__(self, move='train', dataset='WRCD', transform=None, isAug=False, isSwinT=False, data_root=None):
        super(NPYChangeDetectionDataset, self).__init__()
        self.isaug = isAug
        self.isSwinT = isSwinT
        self.move = move
        self.transform = transform

        paths = get_dataset_paths(dataset)
        if not paths:
            raise ValueError(f"Unknown dataset: {dataset}")

        root = data_root if data_root else paths['root']
        if move == 'train':
            img1_dir = root + paths['train_1']
            img2_dir = root + paths['train_2']
            label_dir = root + paths['train_label']
        elif move == 'val':
            img1_dir = root + paths['val_1']
            img2_dir = root + paths['val_2']
            label_dir = root + paths['val_label']
        else:
            img1_dir = root + paths['test_1']
            img2_dir = root + paths['test_2']
            label_dir = root + paths['test_label']

        self.seq_img_1 = read_npy_directory(img1_dir)
        self.seq_img_2 = read_npy_directory(img2_dir)
        self.seq_label = read_npy_directory(label_dir, label=True)

    def augment(self, image, flipCode):
        if flipCode == 1:
            return np.flip(image, axis=1)
        elif flipCode == 0:
            return np.flip(image, axis=0)
        elif flipCode == -1:
            return np.flip(np.flip(image, axis=0), axis=1)
        return image

    def __getitem__(self, index):
        imgs_1 = self.seq_img_1[index].copy()
        imgs_2 = self.seq_img_2[index].copy()
        label = self.seq_label[index].copy()

        if self.isaug:
            flipCode = random.choice([-1, 0, 1, 2])
            if flipCode != 2:
                imgs_1 = self.augment(imgs_1, flipCode)
                imgs_2 = self.augment(imgs_2, flipCode)
                label = self.augment(label, flipCode)
                if len(label.shape) == 2:
                    label = label.reshape(label.shape[0], label.shape[1], 1)

        h, w = label.shape[0], label.shape[1]
        if self.isSwinT:
            label_levels = [int(h / 8), int(h / 16), int(h / 32), int(h / 32)]
        else:
            label_levels = [int(h / 4), int(h / 4), int(h / 16), int(h / 16)]

        if self.transform is not None:
            imgs_1 = self.transform(imgs_1)
            imgs_2 = self.transform(imgs_2)
            label = self.transform(label)

        return imgs_1, imgs_2, label

    def __len__(self):
        return len(self.seq_label)

