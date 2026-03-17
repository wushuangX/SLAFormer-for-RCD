import os
import cv2
import torch.utils.data
import random


# this function is for read image,the input is directory name
def read_directory(directory_name, label=False):
    array_of_img = []  # this if for store all of the image data
    # this loop is for read each image in this folder,directory_name is the folder name with images.
    files = os.listdir(directory_name)
    files.sort(key=lambda x: int(x[0:-4]))
    for filename in files:
        # print(filename) #just for test
        # img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        # print(img)
        if label:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        array_of_img.append(img)
        # print(img)
        # print(array_of_img[0].shape)
    return array_of_img


dataset_LEVIR = 'CD_dataset/LEVIR'
dataset_WHU = 'CD_dataset/WHU'
dataset_PRCV = 'CD_dataset/SemiCD'
dataset_MTWHU = 'CD_dataset/MTWHU'
dataset_MTOSCDOS = 'CD_dataset/MTOSCDOS'
dataset_MTOSCDSO = 'CD_dataset/MTOSCDSO'
dataset_CAU = 'CD_dataset/CAU'

dataset_CRCD = '/opt/data/private/zq/Datasets/CRCD'#4090
# dataset_CRCD = '/opt/data/private/RSTeam/zq/CD_Methods/CD_dataset/CRCD'#3090


dataset_WRCD = '/opt/data/private/zq/Datasets/WRCD'#4090
# dataset_WRCD = '/opt/data/private/RSTeam/zq/CD_Methods/CD_dataset/WRCD'#3090



'''3090'''
# dataset_train_1 = '/2012/train/image'
# dataset_train_2 = '/2014/train/image'
# dataset_train_label = '/2012_2014label/train'
# dataset_test_1 = '/2012/test/image'
# dataset_test_2 = '/2014/test/image'
# dataset_test_label = '/2012_2014label/test'

'''4090'''
# dataset_train_1 = '/train1/A'
# dataset_train_2 = '/train1/B'
# dataset_train_label = '/train1/label'
# dataset_test_1 = '/test1/A'
# dataset_test_2 = '/test1/B'
# dataset_test_label = '/test1/label/'

#CRCD
dataset_train_1 = '/train/A'
dataset_train_2 = '/train/B'
dataset_train_label = '/train/label'
dataset_test_1 = '/test/A'
dataset_test_2 = '/test/B'
dataset_test_label = '/test/label/'








class LevirWhuGzDataset(torch.utils.data.Dataset):
    def __init__(self, move='train', dataset='Gz', transform=None, isAug=False, isSwinT=False):
        super(LevirWhuGzDataset, self).__init__()
        seq_img_1 = []  # to pacify Pycharm
        seq_img_2 = []  # to pacify Pycharm
        seq_label = []  # to pacify Pycharm
        self.isaug = isAug
        self.isSwinT = isSwinT
        self.move = move
        if dataset == 'LEVIR':
            if move == 'train':
                seq_img_1 = read_directory(dataset_LEVIR + dataset_train_1)
                seq_img_2 = read_directory(dataset_LEVIR + dataset_train_2)
                seq_label = read_directory(dataset_LEVIR + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_LEVIR + dataset_test_1)
                seq_img_2 = read_directory(dataset_LEVIR + dataset_test_2)
                seq_label = read_directory(dataset_LEVIR + dataset_test_label, label=True)
        elif dataset == 'WHU':
            if move == 'train':
                seq_img_1 = read_directory(dataset_WHU + dataset_train_1)
                seq_img_2 = read_directory(dataset_WHU + dataset_train_2)
                seq_label = read_directory(dataset_WHU + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_WHU + dataset_test_1)
                seq_img_2 = read_directory(dataset_WHU + dataset_test_2)
                seq_label = read_directory(dataset_WHU + dataset_test_label, label=True)
        elif dataset == 'Gz':
            if move == 'train':
                seq_img_1 = read_directory(dataset_PRCV + dataset_train_1)
                seq_img_2 = read_directory(dataset_PRCV + dataset_train_2)
                seq_label = read_directory(dataset_PRCV + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_PRCV + dataset_test_1)
                seq_img_2 = read_directory(dataset_PRCV + dataset_test_2)
                seq_label = read_directory(dataset_PRCV + dataset_test_label, label=True)
        elif dataset == 'MTWHU':
            if move == 'train':
                seq_img_1 = read_directory(dataset_MTWHU + dataset_train_1)
                seq_img_2 = read_directory(dataset_MTWHU + dataset_train_2)
                seq_label = read_directory(dataset_MTWHU + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_MTWHU + dataset_test_1)
                seq_img_2 = read_directory(dataset_MTWHU + dataset_test_2)
                seq_label = read_directory(dataset_MTWHU + dataset_test_label, label=True)
        elif dataset == 'CAU':
            if move == 'train':
                seq_img_1 = read_directory(dataset_CAU + dataset_train_1)
                seq_img_2 = read_directory(dataset_CAU + dataset_train_2)
                seq_label = read_directory(dataset_CAU + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_CAU + dataset_test_1)
                seq_img_2 = read_directory(dataset_CAU + dataset_test_2)
                seq_label = read_directory(dataset_CAU + dataset_test_label, label=True)

        elif dataset == 'CRCD':
            if move == 'train':
                seq_img_1 = read_directory(dataset_CRCD + dataset_train_1)
                seq_img_2 = read_directory(dataset_CRCD + dataset_train_2)
                seq_label = read_directory(dataset_CRCD + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_CRCD + dataset_test_1)
                seq_img_2 = read_directory(dataset_CRCD + dataset_test_2)
                seq_label = read_directory(dataset_CRCD + dataset_test_label, label=True)
        elif dataset == 'WRCD':
            if move == 'train':
                seq_img_1 = read_directory(dataset_WRCD + dataset_train_1)
                seq_img_2 = read_directory(dataset_WRCD + dataset_train_2)
                seq_label = read_directory(dataset_WRCD + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_WRCD + dataset_test_1)
                seq_img_2 = read_directory(dataset_WRCD + dataset_test_2)
                seq_label = read_directory(dataset_WRCD + dataset_test_label, label=True)
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

