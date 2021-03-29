import math
import torch
import torch.nn as nn

class DDBPN(nn.Module):
    def __init__(self, scale_factor):
        super(DDBPN, self).__init__()
        ####################### Initial Feature Extraction #######################
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=3//2),
            nn.PReLU(256),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=1//2),
            nn.PReLU(64)
        )

        ####################### Up and Down Projection #######################
        self.UpProjection_1 = UpProjection(scale_factor, 64, 64, bottleneck=False)
        self.DownProjection_1 = DownProjection(scale_factor, 64, 64, bottleneck=False)

        self.UpProjection_2 = UpProjection(scale_factor, 64, 64, bottleneck=False)
        self.DownProjection_2 = DownProjection(scale_factor, 64*2, 64, bottleneck=True)

        self.UpProjection_3 = UpProjection(scale_factor, 64*2, 64, bottleneck=True)
        self.DownProjection_3 = DownProjection(scale_factor, 64*3, 64, bottleneck=True)

        self.UpProjection_4 = UpProjection(scale_factor, 64*3, 64, bottleneck=True)
        self.DownProjection_4 = DownProjection(scale_factor, 64*4, 64, bottleneck=True)

        self.UpProjection_5 = UpProjection(scale_factor, 64*4, 64, bottleneck=True)
        self.DownProjection_5 = DownProjection(scale_factor, 64*5, 64, bottleneck=True)

        self.UpProjection_6 = UpProjection(scale_factor, 64*5, 64, bottleneck=True)
        self.DownProjection_6 = DownProjection(scale_factor, 64*6, 64, bottleneck=True)

        self.UpProjection_7 = UpProjection(scale_factor, 64*6, 64, bottleneck=True)
        
        ####################### Reconstruction #######################
        self.final_conv = nn.Conv2d(64*7, 3, kernel_size=3, stride=1, padding=3//2)


    def forward(self, x):
        x = self.initial_conv(x)
        H1 = self.UpProjection_1(x)
        L1 = self.DownProjection_1(H1)

        H2 = self.UpProjection_2(L1)
        L2 = self.DownProjection_2(torch.cat((H1, H2), dim=1))

        H3 = self.UpProjection_3(torch.cat((L1, L2), dim=1))
        L3 = self.DownProjection_3(torch.cat((H1, H2, H3), dim=1))

        H4 = self.UpProjection_4(torch.cat((L1, L2, L3), dim=1))
        L4 = self.DownProjection_4(torch.cat((H1, H2, H3, H4), dim=1))

        H5 = self.UpProjection_5(torch.cat((L1, L2, L3, L4), dim=1))
        L5 = self.DownProjection_5(torch.cat((H1, H2, H3, H4, H5), dim=1))

        H6 = self.UpProjection_6(torch.cat((L1, L2, L3, L4, L5), dim=1))
        L6 = self.DownProjection_6(torch.cat((H1, H2, H3, H4, H5, H6), dim=1))

        H7 = self.UpProjection_7(torch.cat((L1, L2, L3, L4, L5, L6), dim=1))

        H = torch.cat((H1, H2, H3, H4, H5, H6, H7), dim=1)
        H = self.final_conv(H)

        return H



class UpProjection(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, bottleneck=True):
        super(UpProjection, self).__init__()

        if(bottleneck):
            self.BottleNeck = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.PReLU(out_channels)
            )
            mid_channels = out_channels
        else:
            self.BottleNeck = None
            mid_channels = in_channels

        # kernel, stride, padding for different scale
        dictionary = {
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2)
        }
        kernel_size, stride, padding = dictionary[scale_factor]

        self.Deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size, stride, padding),
            nn.PReLU(out_channels)
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size, stride, padding),
            nn.PReLU(mid_channels)
        )
        self.Deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size, stride, padding),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        if(self.BottleNeck is not None):
            x = self.BottleNeck(x) # 1x1 conv

        H_0 = self.Deconv_1(x) # first upsample
        L_0 = self.Conv(H_0) # downsample
        e = L_0.sub(x) # residual / LR difference
        H_1 = self.Deconv_2(e) # augmented LR difference

        H = H_0 + H_1 # final

        return H




class DownProjection(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, bottleneck=True):
        super(DownProjection, self).__init__()

        if(bottleneck):
            self.BottleNeck = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.PReLU(out_channels)
            )
            mid_channels = out_channels
        else:
            self.BottleNeck = None
            mid_channels = in_channels

        # kernel, stride, padding for different scale
        dictionary = {
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2)
        }
        kernel_size, stride, padding = dictionary[scale_factor]

        self.Conv_1 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding),
            nn.PReLU(out_channels)
        )
        self.Deconv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, mid_channels, kernel_size, stride, padding),
            nn.PReLU(mid_channels)
        )
        self.Conv_2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        if self.BottleNeck is not None:
            x = self.BottleNeck(x) # 1x1 conv

        L_0 = self.Conv_1(x) # first downsample
        H_0 = self.Deconv(L_0) # first upsample
        e = H_0.sub(x) # residual / HR difference
        L_1 = self.Conv_2(e) # augmented HR difference
        
        L = L_0 + L_1 # final

        return L