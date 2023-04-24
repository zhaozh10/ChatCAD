import timm
import torch.nn as nn
from Knee.utils.utils import get_stage_size
from Knee.utils.zxClass import Config


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-g", type=str, default="1")
    parser.add_argument("-t", type=bool, default=False)
    parser.add_argument("-c", type=str, default="tasks/CNN/config/Patch_R18_V2.yaml")
    parser.add_argument("-net", type=str, default="")
    args = parser.parse_args()
    return args


class PatchLayer(nn.Module):
    def __init__(self, in_feat, stride):
        super().__init__()
        self.in_feat = in_feat
        self.stride = stride
        if stride != 1:
            out_feat = in_feat * 2
            self.downsample = nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            )
        else:
            out_feat = in_feat
        self.out_feat = out_feat
        self.block_0 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.block_1 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.act = nn.ReLU(inplace=True)
        self.block_2 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x
        out = self.block_0(x)
        out = self.block_1(out)
        if self.stride != 1:
            out += self.downsample(identity)
        out = self.act(out)
        out = self.block_2(out)
        out = self.block_3(out)
        return out


# class PatchConv(nn.Module):
#     def __init__(self, in_feat, stride):
#         super().__init__()
#         self.in_feat = in_feat
#         self.stride = stride
#         if stride != 1:
#             out_feat = in_feat * 2
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
#                 nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#             )
#         else:
#             out_feat = in_feat
#         self.out_feat = out_feat
#         self.block_0 = nn.Sequential(
#             nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#         )
#         self.block_1 = nn.Sequential(
#             nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#         )
#         self.act = nn.ReLU(inplace=True)
#         self.block_2 = nn.Sequential(
#             nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#         )
#         self.block_3 = nn.Sequential(
#             nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         identity = x
#         out = self.block_0(x)
#         out = self.block_1(out)
#         if self.stride != 1:
#             out += self.downsample(identity)
#         out = self.act(out)
#         out = self.block_2(out)
#         out = self.block_3(out)
#         return out


class CNN(nn.Module):
    # def __init__(self, cfg: Config):
    def __init__(self):
        super().__init__()

        # CNN basic blocks
        # _cnn = timm.create_model(cfg.net, drop_block=None, pretrained=True)
        _cnn = timm.create_model("resnet18", drop_block=None, pretrained=True)
        layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), _cnn.bn1, _cnn.act1, _cnn.maxpool)
        self.cnn = nn.Sequential(layer0, _cnn.layer1, _cnn.layer2, _cnn.layer3, _cnn.layer4)
        # ds, ps = get_stage_size(self.cnn, cfg)
        # self.size_cnn = ps

        # pooling
        self.pooling = nn.AdaptiveMaxPool2d([1, 1])

        # FC
        # self.classify = nn.Linear(ds[-1], 3)
        self.classify = nn.Linear(512, 3)

    def forward(self, data):
        x = data.patch
        for layer in self.cnn:
            x = layer(x)
        x = self.pooling(x).squeeze()
        x = self.classify(x)
        return x


class PatchConvs(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        num_layers = int(cfg.net[-1])
        _cnn = timm.create_model('resnet18', pretrained=True)
        layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), _cnn.bn1, _cnn.act1, _cnn.maxpool)
        layers = [layer0, _cnn.layer1, _cnn.layer2, _cnn.layer3, _cnn.layer4]
        if num_layers < 5:
            layers = layers[: num_layers + 1]
        else:
            feats = 512
            for i in range(num_layers - 4):
                layers.append(PatchLayer(feats, 2))
                feats *= 2
        self.cnn = nn.Sequential(*layers)
        ds, ps = get_stage_size(self.cnn, cfg)
        self.size_cnn = ps

        # pooling
        self.pooling = nn.AdaptiveMaxPool2d([1, 1])
        # FC
        self.classify = nn.Linear(ds[-1], 3)

    def forward(self, data):
        x = data.patch
        for layer in self.cnn:
            x = layer(x)
        x = self.pooling(x).squeeze()
        x = self.classify(x)
        return x


# class PatchConvs(nn.Module):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         num_layers = int(cfg.net[-1])
#         channel = 64
#         strides = [1, 2, 2, 2, 2, 2, 2]
#         layer0 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
#         )
#         self.cnn = [layer0]
#         # CNN basic blocks
#         for i in range(num_layers):
#             self.cnn.append(PatchConv(channel, strides[i]))
#             if strides[i] != 1:
#                 channel *= 2

#         # print(self.cnn)
#         self.cnn = nn.Sequential(*self.cnn)
#         ds, ps = get_stage_size(self.cnn, cfg)
#         self.size_cnn = ps

#         # pooling
#         self.pooling = nn.AdaptiveMaxPool2d([1, 1])

#         # FC
#         self.classify = nn.Linear(ds[-1], 3)

#     def forward(self, data):
#         x = data.patch
#         for layer in self.cnn:
#             x = layer(x)
#         x = self.pooling(x).squeeze()
#         x = self.classify(x)
#         return x


class PatchConv(nn.Module):  # high performance
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        if self.in_feat != self.out_feat:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            )
        self.block_0 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.block_1 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.act = nn.ReLU(inplace=True)
        self.block_2 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feat, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x
        out = self.block_0(x)
        out = self.block_1(out)
        if self.in_feat != self.out_feat:
            out += self.downsample(identity)
        out = self.act(out)
        out = self.block_2(out)
        out = self.block_3(out)
        return out
