import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EDSR(nn.Module):
    def __init__(self, scale_factor):
        super(EDSR, self).__init__()
        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=3//2)

        ################### Stacking Residual Blocks ######################
        m = []
        for _ in range(32):
            m.append(ResidualBlock(256, 256))
        self.ResBlocks = nn.Sequential(*m)
        ###################################################################

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=3//2)

        ################### Upsample Blocks ######################
        m = []
        for _ in range(int(math.log(self.scale_factor, 2))):
            m.append(UpsampleBlock_x2(256, 256))
        self.Upsample_Module = nn.Sequential(*m)
        ###################################################################

        self.conv3 = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=3//2)
        
    def forward(self, x):
        out_0 = self.conv1(x)
        out_1 = self.ResBlocks(out_0) # 32 residual blocks
        out_2 = self.conv2(out_1)
        out = out_2 + out_0 # global skip connection
        out = self.Upsample_Module(out)
        out = self.conv3(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3//2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out.mul(self.res_scale) # residual scaling
        out = out + x # local skip connection

        return out

class UpsampleBlock_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock_x2, self).__init__()
        self.Conv = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, stride=1, padding=3//2)
        self.PixelShuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.Conv(x)
        out = self.PixelShuffle(out)

        return out
