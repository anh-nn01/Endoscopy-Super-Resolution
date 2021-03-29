import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=3):
        super(SRCNN, self).__init__()

        self.interpolation = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corner = True)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size = 9, padding = 9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 1//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size = 5, padding = 5//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.interpolation(x)
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        
        return out
        