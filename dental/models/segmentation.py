import torch
import torch.nn as nn 
import torch.nn.functional as F 

class UNetConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, normalization=None, pooling=False):
        super(UNetConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if normalization == 'batch':
            self.n1 = nn.BatchNorm2d(out_channels)
            self.n2 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.n1 = nn.InstanceNorm2d(out_channels)
            self.n2 = nn.InstanceNorm2d(out_channels)
        
        self.norm = normalization
        self.pool = pooling

    
    def forward(self, x):

        if self.pool:
            x = F.max_pool2d(x, 2)

        if self.norm is not None:
            out = F.relu(self.n2(self.conv2(F.relu(self.n1(self.conv1(x))))))
        else:
            out = F.relu(self.conv2(F.relu(self.conv1(x))))
        
        return out

class UNetUpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, mode='upsample', normalization=None):
        super(UNetUpBlock, self).__init__()
        if mode=='upsample':
            self.up = F.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetConvBlock(in_channels, out_channels, normalization=normalization)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class ToothInstanceSegmentation(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, normalization='batch', mode='trans'):
        super(ToothInstanceSegmentation, self).__init__()
        self.dconv1 = UNetConvBlock(in_channels, 64, normalization=normalization)
        self.dconv2 = UNetConvBlock(64, 128, normalization=normalization, pooling=True)
        self.dconv3 = UNetConvBlock(128, 256, normalization=normalization, pooling=True)
        self.dconv4 = UNetConvBlock(256, 512, normalization=normalization, pooling=True)
        factor = 2 if mode=='upsample' else 1
        self.dconv5 = UNetConvBlock(512, 1024//factor, normalization=normalization, pooling=True)
        self.uconv1 = UNetUpBlock(1024, 512//factor, normalization=normalization, mode=mode)
        self.uconv2 = UNetUpBlock(512, 256//factor, normalization=normalization, mode=mode)
        self.uconv3 = UNetUpBlock(256, 128//factor, normalization=normalization, mode=mode)
        self.uconv4 = UNetUpBlock(128, 64, normalization=normalization, mode=mode)
        self.out = nn.Conv2d(64, n_classes, 1)
        self.__init_parameters()
    
    def __init_parameters(self):
        for m in self.modules():

            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x1 = self.dconv1(x)
        x2 = self.dconv2(x1)
        x3 = self.dconv3(x2)
        x4 = self.dconv4(x3)
        x5 = self.dconv5(x4)
        x = self.uconv1(x5, x4)
        x = self.uconv2(x, x3)
        x = self.uconv3(x, x2)
        x = self.uconv4(x, x1)
   
        return self.out(x)
