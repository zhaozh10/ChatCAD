
import torch
import torch.nn as nn




class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, repeat):
        super().__init__()
        layer0 = [torch.nn.Upsample(scale_factor=2)]
        for i in range(1, repeat):
            layer0 += [
                torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.ReLU(inplace=True),
            ]
        layer1 = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        ]
        self.layer0 = nn.Sequential(*layer0)
        self.layer1 = nn.Sequential(*layer1)

    def forward(self, x, _x):
        # (18,512,8,8)
        x = self.layer0(x)
        x = self.layer1(x + _x)
        return x


class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, repeat):
        super().__init__()
        layer0 = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        ]
        for i in range(1, repeat - 1):
            layer0 += [
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
            ]
        layer0 += [
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ]
        layer1 = [
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        ]
        self.layer0 = nn.Sequential(*layer0)
        self.layer1 = nn.Sequential(*layer1)

    def forward(self, x):
        _x = self.layer0(x)
        x = self.layer1(_x)
        return x, _x


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 1
        out_channels = 8
        repeats = (2, 2, 3, 3, 3)
        layers = (64, 128, 256, 512, 512)
        self.down = nn.ModuleList()
        self.down.append(Down(in_channels, layers[0], repeats[0]))
        for i in range(1, len(layers)):
            self.down.append(Down(layers[i - 1], layers[i], repeats[i]))
        self.up = nn.ModuleList()

        for i in reversed(range(1, len(layers))):
            self.up.append(Up(layers[i], layers[i - 1], repeats[i]))
        self.up.append(Up(layers[0], out_channels, repeats[i]))

    def forward(self, x):
        n = len(self.down)
        sc = []
        for i in range(n):
            x, _x = self.down[i](x)
            sc.append(_x)
        for i in range(n):
            x = self.up[i](x, sc[n - i - 1])

        return x

if __name__ == "__main__":
    x = torch.randn(18, 1, 256, 256)
    # net = UNet()
    # state=torch.load("./unet_best.pth")
    # net = torch.load("./Seg/result/best.pth", map_location=torch.device("cpu"))
    net = torch.load("Segmentation/result/best.pth", map_location=torch.device("cpu"))
    
    
    y = net(x)
    print(y.shape)

