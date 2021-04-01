import math
import torch
import torch.nn as nn

class RDN(nn.Module):
    # D: number of Residual Dense Blocks (RDB)
    # G0: num_channels of output of Initial Conv, Local Feature Fusion, and Global Feature Fusion
    # G: num_channels of output of Conv layer in RDB
    # C: number of conv blocks in RDB
    def __init__(self, scale_factor, D=16, G0=64, G=64, C=8):
        super(RDN, self).__init__()

        ################ Shallow Feature Extractor ################
        self.Conv1 = nn.Conv2d(3, G0, kernel_size=3, stride=1, padding=3//2)
        self.Conv2 = nn.Conv2d(G0, G0, kernel_size=3, stride=1, padding=3//2)
        self.RDB_Module = nn.ModuleList(
            [RDB(G0, G, C) for _ in range(D)]
        )
        self.GlobalBottleneck = nn.Conv2d(D*G0, G0, kernel_size=1)
        self.Conv3 = nn.Conv2d(G0, G0, kernel_size=3, stride=1, padding=3//2)

        if(scale_factor == 2):
            self.UpNet = UpsampleBlock_x2(G0)
        elif(scale_factor == 3):
            self.UpNet = UpsampleBlock_x3(G0)
        elif(scale_factor == 4):
            self.UpNet = nn.Sequential(
                UpsampleBlock_x2(G0), 
                UpsampleBlock_x2(G0)
            )
        elif(scale_factor == 8):
            self.UpNet = nn.Sequential(
                UpsampleBlock_x2(G0), 
                UpsampleBlock_x2(G0), 
                UpsampleBlock_x2(G0)
            )
        else:
            raise NotImplementedError

        self.Reconstruction = nn.Conv2d(G0, 3, kernel_size=3, stride=1, padding=3//2)

    def forward(self, x):
        F_ = self.Conv1(x)

        F = [] # output from each conv/RDB layer
        F.append(self.Conv2(F_)) # self.Conv2(F_) = F[0]
        for d in range(len(self.RDB_Module)):
            F.append(self.RDB_Module[d](F[-1])) # append the last output -> F[1], F[2], ..., F[D]

        GFF = self.GlobalBottleneck(torch.cat(tuple(f for f in F[1:]), dim=1)) # Global Feature Fusion, starting from F_1 to F_D
        F_GF = self.Conv3(GFF)
        F_DF = F_GF + F_

        out = self.UpNet(F_DF)
        out = self.Reconstruction(out)

        return out
        
        
        
"""
Residual Dense Block - the Basic Block of the RDN
"""
class RDB(nn.Module):
    """
    G0: num_channels ouputed by Shallow Conv & Local/Global Feature Fusion (num_channels for input/output of RDB)
    G: num_channels outputed by each Conv layer in RDB
    C: number of conv layers in each RDB (Residual Dense Block)
    """
    def __init__(self, G0, G, C):
        super(RDB, self).__init__()
        conv_layers = []
        for i in range(C):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(G0 + i*G, G, kernel_size=3, stride=1, padding=3//2),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv_layers = nn.ModuleList([*conv_layers])
        self.bottleneck = nn.Conv2d(G0 + C*G, G0, kernel_size=1)

    def forward(self, x):
        C = len(self.conv_layers)
        conv_out = []

        # Dense Connections
        for i in range(C):
            out = self.conv_layers[i](torch.cat((x, *conv_out), dim=1))
            conv_out.append(out)

        # Local Feature Fusion
        out = self.bottleneck(torch.cat((x, *conv_out), dim=1))
        out = out + x

        return out


class UpsampleBlock_x2(nn.Module):
    def __init__(self, G0):
        super(UpsampleBlock_x2, self).__init__()
        self.conv = nn.Conv2d(G0, 4 * G0, kernel_size=3, stride=1, padding=3//2)
        self.PixelShuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.PixelShuffle(out)

        return out

class UpsampleBlock_x3(nn.Module):
    def __init__(self, G0):
        super(UpsampleBlock_x3, self).__init__()
        self.conv = nn.Conv2d(G0, 9 * G0, kernel_size=3, stride=1, padding=3//2)
        self.PixelShuffle = nn.PixelShuffle(3)

    def forward(self, x):
        out = self.conv(x)
        out = self.PixelShuffle(out)

        return out
        

