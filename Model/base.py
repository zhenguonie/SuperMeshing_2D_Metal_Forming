import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()

        self.MLP = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        
        self.CONV = torch.nn.Conv2d(in_channels=2,
                                     out_channels=1,
                                     kernel_size=(7,7),
                                     stride=(1,1),
                                     padding=3)

    def forward(self, x):
        wa_C = F.adaptive_avg_pool2d(x, 1)
        wm_C = F.adaptive_max_pool2d(x, 1)
        wa_C_MLP = self.MLP(wa_C)
        wm_C_MLP = self.MLP(wm_C)
        w_C_MLP = wa_C_MLP+wm_C_MLP
        w_C_MLP = torch.sigmoid(w_C_MLP)
        
        x_C = x*w_C_MLP
        
        wa_S = torch.mean(x_C, 1)
        wm_S = torch.max(x_C, 1)
        w_S = torch.cat((wa_S.unsqueeze(1), wm_S[0].data.unsqueeze(1)),1)
        w_S_CONV = self.CONV(w_S)
        w_S_CONV = torch.sigmoid(w_S_CONV)
        x_S = x_C*w_S_CONV

        return x_S

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.CBAM_block = CBAMBlock(planes)
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        
        out = self.CBAM_block(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

