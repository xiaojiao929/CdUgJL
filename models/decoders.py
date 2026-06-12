import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class SegDecoder(nn.Module):
    """UNet-style segmentation decoder with skip connections."""

    def __init__(self, enc_channels, bottleneck_ch, num_classes):
        super().__init__()
        c0, c1, c2, c3 = enc_channels         # 64, 128, 256, 512
        bn = bottleneck_ch                     # 1024

        self.up3 = UpBlock(bn, c3, c3 // 2)   # 1024+512 -> 256
        self.up2 = UpBlock(c3 // 2, c2, c2 // 2)  # 256+256 -> 128
        self.up1 = UpBlock(c2 // 2, c1, c1 // 2)  # 128+128 -> 64
        self.up0 = UpBlock(c1 // 2, c0, c0)       # 64+64 -> 64

        self.head = nn.Conv2d(c0, num_classes, 1)

    def forward(self, bottleneck, skips):
        s0, s1, s2, s3 = skips
        x = self.up3(bottleneck, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


class QuantDecoder(nn.Module):
    """Regression decoder for tumor quantification (X, Y, Area)."""

    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.fc(self.pool(x))
