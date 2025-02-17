import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class V2VModel(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(V2VModel, self).__init__()

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
        self.local_heatmaps = nn.Conv3d(self.Nchannels, num_classes, [1, 1, 1], [1, 1, 1], [0, 0, 0], bias=True)
        self.local_downsampled = nn.AdaptiveAvgPool3d((16,16,16))
        self.sconv0 = nn.Conv3d(num_classes, 128, [5, 5, 5], [1, 1, 1], [2, 2, 2], bias=False)
        self.sconv1 = nn.Conv3d(128, 128, [5, 5, 5], [1, 1, 1], [2, 2, 2], bias=False)
        self.sconv2 = nn.Conv3d(128, 128, [5, 5, 5], [1, 1, 1], [2, 2, 2], bias=False)
        self.spatial_downsampled = nn.Conv3d(128, num_classes, [5, 5, 5], [1, 1, 1], [2, 2, 2], bias=False)
        self.act_tanh = nn.Tanh()
        self.act_leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        y3 = self.up3(x4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0) #[bs,16,64,64,64]

        local_heatmaps = self.local_heatmaps(y0) #[bs,5,64,64,64]
        tmp = self.local_downsampled(local_heatmaps) #[bs,5,16,16,16]
        tmp = self.act_leakyrelu(self.sconv0(tmp))
        tmp = self.act_leakyrelu(self.sconv1(tmp))
        tmp = self.act_leakyrelu(self.sconv2(tmp))

        spatial_small = self.act_tanh(self.spatial_downsampled(tmp))
        spatial_heatmaps = F.interpolate(spatial_small, local_heatmaps.shape[2:6], mode='trilinear')
        out = local_heatmaps*spatial_heatmaps
        return out

class ConvLayer_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer_BN, self).__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn(self.conv3d(x)))
        return y

class ShortcutBlock(nn.Module):
    def __init__(self, channels):
        super(ShortcutBlock, self).__init__()

        self.conv1 = nn.Conv3d(channels, channels // 2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels // 2)
        self.conv2 = nn.Conv3d(channels // 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

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
        y = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        y = torch.cat((y, x1), 1)
        y = self.conv(self.conv1(y))
        return y

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    te = V2VModel(num_classes=5, in_channels=16)
    summary(te, (16,64,64,64),batch_size=2,device="cpu")
    # print(te.state_dict())