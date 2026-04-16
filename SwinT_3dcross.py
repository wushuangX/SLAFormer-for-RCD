import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cv2 import resize
from einops import rearrange
import torchvision
from microsoft_swintransformer import swin_t, swin_b, swin_l, swin_s
from thop import clever_format
from thop import profile
from Pvtv2 import pvt_v2_b2
from modules.Cross_fuse import cross_fuse_3d
from modules.layers import FaLayer_sum


class semantic_context_enhance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
# from modules.Cross_fuse_2d import cross_fuse_2d
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

import cv2
import numpy as np
import os
import time


class oneXone_conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(oneXone_conv, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(out_features)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.drop(x)
        x = self.Conv2(x)
        x = self.drop(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = oneXone_conv(in_features=dim, out_features=16 * dim) if dim_scale == 2 else nn.Identity()
        self.output_dim = dim
        self.norm = norm_layer([4, input_resolution[0] * 4, input_resolution[1] * 4])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        return x


class SwinT_FPANet(nn.Module):  # ablation-4：完整模型
    # res2net based encoder decoder
    def __init__(self, backbone='res2net50', pretrain=True, img_size=256, patch_size=4, embed_dims=64, name=''):
        super(SwinT_FPANet, self).__init__()
        self.backbone = pvt_v2_b2()

        self.output4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                      dim_scale=4, dim=embed_dims)
        self.output = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
        self.out1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1))
        filters = [64, 128, 256, 512, 1024]
        self.FCA1 = FaLayer_sum(filters[0])  # ------64
        self.FCA2 = FaLayer_sum(filters[1])  # ------128
        self.FCA3 = FaLayer_sum(filters[2])  # ------256
        self.FCA4 = FaLayer_sum(filters[3])  # ------512

        self.coattpre1 = FaLayer_sum(filters[0])  # ------64
        self.coattpre2 = FaLayer_sum(filters[1])  # ------128
        self.coattpre3 = FaLayer_sum(filters[2])  # ------256
        self.coattpre4 = FaLayer_sum(filters[3])  # ------512

        self.coattpost1 = FaLayer_sum(filters[0])  # ------64
        self.coattpost2 = FaLayer_sum(filters[1])  # ------128
        self.coattpost3 = FaLayer_sum(filters[2])  # ------256
        self.coattpost4 = FaLayer_sum(filters[3])  # ------512

        self.cross_fuse_c1 = cross_fuse_3d(in_channels=128)
        self.cross_fuse_c2 = cross_fuse_3d(in_channels=256)
        self.cross_fuse_c3 = cross_fuse_3d(in_channels=512)
        self.cross_fuse_c4 = cross_fuse_3d(in_channels=1024)

        self.sce = semantic_context_enhance()

        self.Mid_Conv1 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0)
        self.Mid_Conv2 = nn.Conv2d(128 * 2, 128, kernel_size=1, stride=1, padding=0)
        self.Mid_Conv3 = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1, padding=0)
        self.Mid_Conv4 = nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1, padding=0)
    def up_x4(self, x):
        B, C, H, W = x.shape
        x = self.up(x)
        x = self.output(x)
        return x

    def forward(self, i1, i2):
        x1 = self.backbone(i1)
        x2 = self.backbone(i2)
        diff_1 = self.FCA1(x2[0]) - self.FCA1(x1[0])  #1,64,1,1
        x1_0 = x1[0] + x1[0] * (self.FCA1(x2[0] - x1[0]) + diff_1)# 1,64,64,64
        x2_0 = x2[0] + x2[0] * (self.FCA1(x2[0] - x1[0]) + diff_1)
        dF1 = self.cross_fuse_c1([x1_0, x2_0])# 1,64,64,64

        diff_2 = self.FCA2(x2[1]) - self.FCA2(x1[1])  # 1,128,32,32
        x1_1 = x1[1] + x1[1] * (self.FCA2(x2[1] - x1[1]) + diff_2)
        x2_1 = x2[1] + x2[1] * (self.FCA2(x2[1] - x1[1]) + diff_2)
        dF2 = self.cross_fuse_c2([x1_1, x2_1])

        diff_3 = self.FCA3(x2[2]) - self.FCA3(x1[2])  # 1,256,16,16
        x1_2 = x1[2] + x1[2] * (self.FCA3(x2[2] - x1[2]) + diff_3)
        x2_2 = x2[2] + x2[2] * (self.FCA3(x2[2] - x1[2]) + diff_3)
        dF3 = self.cross_fuse_c3([x1_2, x2_2])

        diff_4 = self.FCA4(x2[3]) - self.FCA4(x1[3])  # 1,512,8,8
        x1_3 = x1[3] + x1[3] * (self.FCA4(x2[3] - x1[3]) + diff_4)
        x2_3 = x2[3] + x2[3] * (self.FCA4(x2[3] - x1[3]) + diff_4)
        dF4 = self.cross_fuse_c4([x1_3, x2_3])

        output4 = self.output4(F.interpolate(dF4, size=dF3.size()[2:], mode='bilinear'))  # (6,64,16,16)
        output3 = self.output3(dF3) + output4  # 6,64,16,16
        output3 = F.interpolate(output3, size=dF2.size()[2:], mode='bilinear')  # (6,64,32,32)
        output2 = self.output2(dF2) + output3
        output2 = F.interpolate(output2, scale_factor=2, mode='bilinear', align_corners=False)
        output2 = F.interpolate(output2, size=dF1.size()[2:], mode='bilinear')  # (6,64,64,64)
        output1 = dF1 + output2  # (6,64,64,64)
        out1 = self.up_x4(output1)
        return out1

if __name__ == "__main__":
    # ------------------------
    # 配置参数
    # ------------------------
    batch_size = 1
    channels = 3
    height, width = 256, 256
    warmup_runs = 10
    test_runs = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.inference_mode():
        # 创建模拟输入
        x1 = torch.rand(batch_size, channels, height, width).to(device)
        x2 = torch.rand(batch_size, channels, height, width).to(device)

        # 初始化模型
        model = SwinT_FPANet().to(device)
        model.eval()

        # 前向推理一次（检查输出）
        out_result = model(x1, x2)

        # 计算 FLOPs 和参数量
        flops, params = profile(model, (x1, x2,))
        print("-" * 50)
        print(f'FLOPs  = {flops / 1e9:.2f} G')   # GigaFLOPs
        print(f'Params = {params / 1e6:.2f} M')  # Million Params

        # 预热（不计入时间）
        for _ in range(warmup_runs):
            _ = model(x1, x2)

        # 正式测试推理速度
        start_time = time.time()
        for _ in range(test_runs):
            _ = model(x1, x2)
        end_time = time.time()

        # 计算 FPS（帧率）
        fps = test_runs / (end_time - start_time)
        print(f'Inference FPS = {fps:.2f}')
# if __name__ == "__main__":
#     model = SwinT_FPANet()
#     print(model)