import math
import torch
import torch.nn as nn

class RDN(nn.Module):
    def __init__(self, scale_factor):
        super(RDN, self).__init__()

        ################ Shallow Feature Extractor ################
        self.Conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3//2)
        self.Conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3//2)



    def forward(self, x):
        pass

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
        self.conv_layers = nn.Sequential(*conv_layers)
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


class UpNet(nn.Module):
    def __init__(self, scale_factor):
        super(UpNet, self).__init__()


    def forward(self, x):
        pass
        

