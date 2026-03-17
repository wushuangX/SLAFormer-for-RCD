import torch
from thop import profile
from torch import nn


class cross_fuse_3d(nn.Module):
    def __init__(self, in_channels):
        super(cross_fuse_3d, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
        )
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
        )
        self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):
        tensor1 = x[0]
        tensor2 = x[1]
        b, c, h, w = tensor1.shape
        tensor1 = tensor1.view(b, c, h*w)#(1,64,4096)
        tensor2 = tensor2.view(b, c, h*w)
        cross_x = torch.cat((tensor1, tensor2), dim=2)#(1,64,8192)
        cross_x = cross_x.view(b, c*2, h, w)#(1,128,64,64)
        cross_x = cross_x.unsqueeze(1)#(1,1,128,64,64)
        cross_x = self.conv3d(cross_x)#(1,1,128,64,64)
        cross_x = cross_x.squeeze(1)#(1,128,64,64)
        cross_x = self.fuse_conv(cross_x)#(1,64,64,64)
        cross_x = self.dropout(cross_x)
        cross_x = self.fuse_conv1(cross_x)#(1,64,64,64)
        cross_x = self.dropout(cross_x)
        return cross_x

if __name__ == '__main__':
    input1 = torch.randn(1, 3, 256, 256)  # b c h w
    input2 = torch.randn(1, 3, 256, 256)  # b c h w
    input = (input1, input2)
    wtconv = cross_fuse_3d(in_channels=6)
    output = wtconv(input)
    print(output.size())
    flops, params = profile(wtconv, (input,))
    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')