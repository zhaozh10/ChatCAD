import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LSE_R = 6.0


def normalization(feautures):
    B, _, H, W = feautures.size()
    outs = feautures.squeeze(1)
    outs = outs.view(B, -1)
    outs_min = outs.min(dim=1, keepdim=True)[0]
    outs_max = outs.max(dim=1, keepdim=True)[0]
    norm = outs_max - outs_min
    norm[norm == 0] = 1e-5
    outs = (outs - outs_min) / norm
    outs = outs.view(B, 1, H, W)
    return outs


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FABlock(nn.Module):
    def __init__(self, in_channels, norm_layer=None, reduction=8):
        super(FABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(in_channels, 1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.conv2 = conv1x1(in_channels, in_channels)

        self.conv3 = conv1x1(in_channels, 1)
        self.conv4 = conv3x3(1, 1)
        self.bn4 = norm_layer(1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # channel attention
        y = self.conv1(x).view(B, 1, -1)
        y = F.softmax(y, dim=-1)
        y = y.permute(0, 2, 1).contiguous()
        y = torch.matmul(x.view(B, C, -1), y).view(B, -1)
        y = self.channel_fc(y)
        y = torch.sigmoid(y).unsqueeze(2).unsqueeze(3).expand_as(x)

        x_y = self.conv2(x)
        x_y = x_y * y

        # position attention
        x_y_z = self.conv3(x_y)
        z = self.conv4(x_y_z)
        z = self.bn4(z)
        z = torch.sigmoid(z)
        x_y_z = x_y_z * z

        out = self.gamma * x_y_z + x
        attention_outs = normalization(self.gamma * x_y_z)

        return out


class FAM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FAM, self).__init__()
        self.fab = FAModule(in_planes, out_planes)
        self.header = nn.Sequential(
            conv3x3(in_planes + out_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        x_fa = self.fab(x)
        x = self.header(torch.cat((x, x_fa), dim=1))
        return x


class FAModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(FAModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conva = nn.Sequential(
            conv3x3(in_channels, out_channels), norm_layer(out_channels)
        )
        self.fa = FABlock(out_channels, norm_layer)
        self.convb = nn.Sequential(
            conv3x3(out_channels, out_channels), norm_layer(out_channels)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.fa(output)
        output = self.convb(output)
        return output


class LSE_Pooling(nn.Module):
    def __init__(self):
        super(LSE_Pooling, self).__init__()

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        inputs = inputs.view(B, C, -1)
        inputs = LSE_R * inputs
        outputs = (torch.logsumexp(inputs, dim=2) - math.log(H * W)) / LSE_R
        return outputs
