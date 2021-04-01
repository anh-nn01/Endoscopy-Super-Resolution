import math
import torch
import torch.nn as nn

"""
Num Residual Groups:  G = 10
Num Channel Attention Blocks: B = 20
Growth rate: C = 64
Residual Factor: r = 16 (in channel attention block)
"""
class RCAN(nn.Module):
    def __init__(self, scale_factor, C = 64, num_residual_groups = 10, num_attention_modules = 20, reduction_factor = 16):
        super(RCAN, self).__init__()
        self.initial_conv = nn.Conv2d(3, C, kernel_size = 3, stride = 1, padding = 3//2)
        
        RGs = [ ResidualGroup(C, num_attention_modules, reduction_factor) for _ in range(num_residual_groups) ]
        self.ResidualGroups = nn.Sequential(*RGs) 
        
        self.inter_conv = nn.Conv2d(C, C, kernel_size = 3, stride = 1, padding = 3//2)

        if(scale_factor == 2):
            self.Upscale_Module = UpsampleBlock_x2(C)
        elif(scale_factor == 3):
            self.Upscale_Module = UpsampleBlock_x3(C)
        elif(scale_factor == 4):
            self.Upscale_Module = nn.Sequential(
                UpsampleBlock_x2(C), 
                UpsampleBlock_x2(C)
            )
        elif(scale_factor == 8):
            self.Upscale_Module = nn.Sequential(
                UpsampleBlock_x2(C), 
                UpsampleBlock_x2(C), 
                UpsampleBlock_x2(C)
            )
        else:
            raise NotImplementedError

        self.reconstruction = nn.Conv2d(C, 3, kernel_size = 3, stride = 1, padding = 3//2)

    def forward(self, x):
        out_0 = self.initial_conv(x) # initial feature transformation
        out_1 = self.ResidualGroups(out_0) # Residual Groups with Channel Attention modules
        out_1 = self.inter_conv(out_1) # improve feature from residual groups
        out = out_1 + out_0 # long skip connection

        out = self.Upscale_Module(out)
        out = self.reconstruction(out)

        return out

"""
Residual Group
"""
class ResidualGroup(nn.Module):
    def __init__(self, C, num_attention_modules, reduction_factor):
        super(ResidualGroup, self).__init__()
        Attention_Modules = [RCAB(num_channels = C, reduction_factor = reduction_factor) for _ in range(num_attention_modules)]
        self.RCAB_Modules = nn.Sequential(
            *Attention_Modules
        )
        self.Conv = nn.Conv2d(C, C, kernel_size = 3, stride = 1, padding = 3//2)



    def forward(self, x):
        out = self.RCAB_Modules(x) # Residual Channel Attention Blocks
        out = self.Conv(out) # Convolution to improve spatial quality
        out = out + x # short skip connection -> focus on discriminative features

        return out


"""
Residual Channel Attention Block
"""
class RCAB(nn.Module):
    """
    reduction_factor: the channel reduction factor for channel-wise attention module
    num_channels: number of channel which need attention module
    """
    def __init__(self, num_channels, reduction_factor):
        super(RCAB, self).__init__()
        self.initial_transformation = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size= 3, stride= 1, padding= 3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size= 3, stride= 1, padding= 3//2)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction_factor, kernel_size = 1, padding = 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction_factor, num_channels, kernel_size = 1, padding = 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.initial_transformation(x) # initial transformation for better channel attention training
        channel_attention_map = self.channel_attention(f) # attention weight
        f = f * channel_attention_map # apply attention to the feature
        x = x + f # combine attention feature map to the original feature map
        
        return x

class UpsampleBlock_x2(nn.Module):
    def __init__(self, C):
        super(UpsampleBlock_x2, self).__init__()
        self.conv = nn.Conv2d(C, 4 * C, kernel_size=3, stride=1, padding=3//2)
        self.PixelShuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.PixelShuffle(out)

        return out

class UpsampleBlock_x3(nn.Module):
    def __init__(self, C):
        super(UpsampleBlock_x3, self).__init__()
        self.conv = nn.Conv2d(C, 9 * C, kernel_size=3, stride=1, padding=3//2)
        self.PixelShuffle = nn.PixelShuffle(3)

    def forward(self, x):
        out = self.conv(x)
        out = self.PixelShuffle(out)

        return out