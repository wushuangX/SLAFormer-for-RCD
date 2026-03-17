import torch
from torch import nn
import torch.nn.functional as F
from thop import clever_format
from thop import profile
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)#(b,2,8,8)
        x_out = self.spatial(x_compress)#(b,1,8,8)
        scale = F.sigmoid(x_out)  # #(b,1,8,8)
        return x * scale
if __name__ == "__main__":
    with torch.no_grad():
        x1 = torch.rand(1, 3, 256, 256).to("cuda")
        # x2 = torch.rand(1, 3, 256, 256).to("cuda")

        model = SpatialGate().to("cuda")
        output = model(x1)
        print(output.shape)
        flops, params = profile(model, (x1,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f'FLOPs: {flops}, Parameters: {params}')