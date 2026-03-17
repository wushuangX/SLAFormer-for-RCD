import numpy as np
import torch
from skimage import morphology
import cv2 as cv
import numpy as np
import cv2 as cv
import torch.nn as nn
class SkeletonExtraction(nn.Module):
    def __init__(self, kernel_size=3):
        super(SkeletonExtraction, self).__init__()
        # 定义形态学膨胀使用的卷积核
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))

    def forward(self, features):
        batch_size, channels, height, width = features.shape

        # 将输出初始化为与输入同样大小的Tensor
        thinned_features = torch.zeros_like(features, device=features.device)

        for i in range(batch_size):
            for j in range(channels):
                # 提取当前批次和通道的特征图并转换为NumPy数组
                feature_map = features[i, j, :, :].detach().cpu().numpy()

                # 转换为8位灰度图像
                feature_map = np.uint8(feature_map * 255)  # 假设特征值在[0, 1]之间

                # 应用骨架提取
                ret, binary = cv.threshold(feature_map, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)#h灰度图转换为黑白图
                thinned = cv.ximgproc.thinning(binary)

                # 加粗骨架线
                thinned = cv.dilate(thinned, self.kernel, iterations=1)

                # 转换为浮点型并归一化到[0, 1]，再转换为torch张量
                thinned_tensor = torch.from_numpy(thinned / 255.0).to(features.device)

                # 将结果存入 thinned_features
                thinned_features[i, j, :, :] = thinned_tensor

        return thinned_features

