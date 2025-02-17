import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)#[2,128,48,32]->[2,32,12,8]

        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, -1))
        x = x.mean(dim=-1)

        out = self.head(x)#[2,32]->[2,5]

        return out

class Loc_SCN(nn.Module):
    def __init__(self, num_classes, in_channels,
                alg_confidences=False, vol_confidences=False):
        super(Loc_SCN, self).__init__()

        self.Nchannels = 16
        # Initial convolution layers
        self.conv0 = ConvLayer_BN(in_channels, self.Nchannels, 3, 1, 1)

        self.down1 = downBlock(self.Nchannels, self.Nchannels * 2, 3, 2, 1)
        self.down2 = downBlock(self.Nchannels * 2, self.Nchannels * 4, 3, 2, 1)
        self.down3 = downBlock(self.Nchannels * 4, self.Nchannels * 8, 3, 2, 1)
        self.down4 = downBlock(self.Nchannels * 8, self.Nchannels * 16, 3, 2, 1)

        self.up3 = UpsampleBlock(self.Nchannels * 16, self.Nchannels * 8)
        self.up2 = UpsampleBlock(self.Nchannels * 8, self.Nchannels * 4)
        self.up1 = UpsampleBlock(self.Nchannels * 4, self.Nchannels * 2)
        self.up0 = UpsampleBlock(self.Nchannels * 2, self.Nchannels)

        # scn
        self.local_heatmaps = nn.Conv2d(self.Nchannels, num_classes, [1, 1], [1, 1], [0, 0], bias=True)
        self.local_downsampled = nn.AdaptiveAvgPool2d((48,32))
        self.sconv0 = nn.Conv2d(num_classes, 128, [5, 5], [1, 1], [2, 2], bias=False)
        self.sconv1 = nn.Conv2d(128, 128, [5, 5], [1, 1], [2, 2], bias=False)
        self.sconv2 = nn.Conv2d(128, 128, [5, 5], [1, 1], [2, 2], bias=False)
        self.spatial_downsampled = nn.Conv2d(128, num_classes, [5, 5], [1, 1], [2, 2], bias=False)
        self.act_tanh = nn.Tanh()
        self.act_leakyrelu = nn.LeakyReLU(0.1)

        if alg_confidences:
            self.alg_confidences = GlobalAveragePoolingHead(256, num_classes)
        if vol_confidences:
            self.vol_confidences = GlobalAveragePoolingHead(256, 16)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        y3 = self.up3(x4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)# [16,768,512]

        local_heatmaps = self.local_heatmaps(y0)
        tmp = self.local_downsampled(local_heatmaps)
        tmp = self.act_leakyrelu(self.sconv0(tmp))
        tmp = self.act_leakyrelu(self.sconv1(tmp))
        tmp = self.act_leakyrelu(self.sconv2(tmp))

        alg_confidences = None
        if hasattr(self, "alg_confidences"):
            alg_confidences = self.alg_confidences(x4)#[bs,256,48,32]->[bs,5]
        vol_confidences = None
        if hasattr(self, "vol_confidences"):
            vol_confidences = self.vol_confidences(x4)#[bs,256,48,32]->[bs,16]
        spatial_small = self.act_tanh(self.spatial_downsampled(tmp))
        spatial_heatmaps = F.interpolate(spatial_small, local_heatmaps.shape[2:5], mode='bilinear')
        out = local_heatmaps*spatial_heatmaps
        return out, y0, alg_confidences, vol_confidences

class ConvLayer_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer_BN, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn(self.conv2d(x)))
        return y

class ShortcutBlock(nn.Module):
    def __init__(self, channels):
        super(ShortcutBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn1(self.conv1(x)))
        y = self.leakyrelu(self.bn2(self.conv2(y)))
        y = y + x
        return y

class downBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_blocks=2):
        super(downBlock, self).__init__()

        self.down1 = ConvLayer_BN(in_channels, out_channels, kernel_size, stride, padding)
        layers = []
        for i in range(num_blocks):
            layers.append(ShortcutBlock(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(self.down1(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):

        super(UpsampleBlock, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = ConvLayer_BN(in_channels + out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        layers = []
        for i in range(num_blocks):
            layers.append(ShortcutBlock(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, x1):
        # y = self.upsample(x)
        y = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x1), 1)
        y = self.conv(self.conv1(y))
        return y

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    te = Loc_SCN(num_classes=1, in_channels=1)
    summary(te, (1,640,640),batch_size=2,device="cpu")
    # print(te.state_dict())