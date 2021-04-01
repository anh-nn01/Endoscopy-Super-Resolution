import math
import torch
import torch.nn as nn

"""
SR Generator
"""
class Generator(nn.Module):
    def __init__(self, scale_factor): # scale_factor must be a factor of 2
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=9//2)
        self.PReLU = nn.PReLU(64)

        self.ResidualBlock_1 = ResidualBlock(64, 64)
        self.ResidualBlock_2 = ResidualBlock(64, 64)
        self.ResidualBlock_3 = ResidualBlock(64, 64)
        self.ResidualBlock_4 = ResidualBlock(64, 64)
        self.ResidualBlock_5 = ResidualBlock(64, 64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3//2)
        self.bn = nn.BatchNorm2d(64)

        ################## Upsampling ##################
        m = []
        for _ in range(int(math.log(scale_factor, 2))):
            m.append(UpsampleBlock_x2(64,64))
        self.Upsample_Module = nn.Sequential(*m)
        ################################################

        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=3//2)
            

    def forward(self, x):

        ################## Shallow Conv ##################
        out_0 = self.conv1(x)
        out_0 = self.PReLU(out_0)

        ################## Residual Feature Transformation (local skip connections are in ResidualBlock) ##################
        out = self.ResidualBlock_1(out_0)
        out = self.ResidualBlock_2(out)
        out = self.ResidualBlock_3(out)
        out = self.ResidualBlock_4(out)
        out = self.ResidualBlock_5(out)

        ################## Global Residual Learning ##################
        out = self.conv2(out)
        out = self.bn(out)
        out = out + out_0 # global skip connection

        ################## Upsampling ##################
        out = self.Upsample_Module(out)

        out = self.conv3(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=3//2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.PReLU = nn.PReLU(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3//2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.PReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + x # local skip connection

        return out

class UpsampleBlock_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock_x2, self).__init__()
        self.conv = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, stride=1, padding=3//2)
        self.PixelShuffle = nn.PixelShuffle(2)
        self.PReLU = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.PixelShuffle(out)
        out = self.PReLU(out)

        return out

#################################################################################################################

"""
SR Discriminator
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size, 1))