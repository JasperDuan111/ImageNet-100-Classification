import torch
from torch import nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                                   padding, groups=in_channels, bias=bias)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class XceptionEntryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_with_relu=True):
        super().__init__()
        self.sepconv1 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ) if start_with_relu else SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.sepconv2 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.maxpool(x)
        x += residual
        return x

class XceptionMiddleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sepconv1 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.sepconv2 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.sepconv3 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        residual = x
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        x += residual
        return x
    
class XceptionExitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv1 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.sepconv2 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.maxpool(x)
        x += residual
        return x

class XceptionNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),  # 149x149x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # 147x147x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            XceptionEntryBlock(64, 128, start_with_relu=False),  # 74x74x128
            XceptionEntryBlock(128, 256),                        # 37x37x256
            XceptionEntryBlock(256, 728)                         # 19x19x728
        )

        self.middle = nn.Sequential(
            *[XceptionMiddleBlock(728) for _ in range(8)]       # 八个中间块，保持19x19x728
        )

        self.exit = nn.Sequential(
            XceptionExitBlock(728, 1024),                       # 10x10x1024
            SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
