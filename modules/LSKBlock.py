import torch
import torch.nn as nn
from thop import clever_format
from thop import profile
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x
    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1,
                                      padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        # self.sru = SRU(dim)
    def forward(self, x):
        #依次通过两个大核卷积 （1,512,8,8)
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        #attn1、attn2分别1x1卷积进行空间特征向量的通道融合
        attn1 = self.conv1(attn1)#c:512->256
        attn2 = self.conv2(attn2)
        #按通道拼接attn1、attn2
        attn = torch.cat([attn1, attn2], dim=1)#（1,512,8,8)
        #dim=1,按列输出平均值、最大值
        #dim=0,按行输出平均值、最大值
        avg_attn = torch.mean(attn, dim=1, keepdim=True)#(1,1,8,8)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)#(1,2,8,8)
        sig = self.conv_squeeze(agg).sigmoid()#将2个通道的池化特征转换为N个空间注意力特征图
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        # attn = self.sru(attn)

        return x * attn


if __name__ == '__main__':
    block = LSKblock(64).cuda()
    input = torch.rand(3, 64, 32, 32).cuda() #输入B C H W
    output = block(input)
    print(output.size())
    flops, params = profile(block, (input,))
    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')