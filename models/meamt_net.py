#Amber
# MIT License
# Copyright (c) 2025 Amber Xiao

import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoders import SegmentationDecoder, QuantificationDecoder
from .evidence_head import EvidenceHead

# -----------------------
# Basic Building Blocks
# -----------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# -----------------------
# Edge-guided Attention
# -----------------------

class EdgeAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.edge_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        edge_map = self.sigmoid(self.edge_conv(feat))
        return feat * edge_map + feat

# -----------------------
# Mamba-style Global Module (Placeholder)
# -----------------------

class MambaGlobalBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # placeholder for actual mamba
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        global_feat = self.global_fc(x).view(b, c, 1, 1)
        return x + global_feat.expand_as(x)

# -----------------------
# MEaMt-Net Backbone
# -----------------------

class MEaMtNet(nn.Module):
    def __init__(self, in_channels=2, base_ch=32, num_classes=2):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, base_ch)
        self.encoder2 = Downsample(base_ch, base_ch * 2)
        self.encoder3 = Downsample(base_ch * 2, base_ch * 4)
        self.encoder4 = Downsample(base_ch * 4, base_ch * 8)

        self.mamba_block = MambaGlobalBlock(base_ch * 8)
        self.edge_attention = EdgeAttention(base_ch * 8)

        self.decoder3 = Upsample(base_ch * 8, base_ch * 4)
        self.decoder2 = Upsample(base_ch * 4, base_ch * 2)
        self.decoder1 = Upsample(base_ch * 2, base_ch)

        self.seg_decoder = SegmentationDecoder(in_ch=base_ch, out_ch=num_classes)
        self.quant_decoder = QuantificationDecoder(in_ch=base_ch)
        self.evi_head = EvidenceHead(in_ch=base_ch)

    def forward(self, x):
        e1 = self.encoder1(x)      # -> [B, C, H, W]
        e2 = self.encoder2(e1)     # -> [B, 2C, H/2, W/2]
        e3 = self.encoder3(e2)     # -> [B, 4C, H/4, W/4]
        e4 = self.encoder4(e3)     # -> [B, 8C, H/8, W/8]

        edge_feat = self.edge_attention(e4)
        global_feat = self.mamba_block(edge_feat)

        d3 = self.decoder3(global_feat, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        seg_out = self.seg_decoder(d1)
        quant_out = self.quant_decoder(d1)
        evidence_out = self.evi_head(d1)

        return {
            "seg": seg_out, 
            "quant": quant_out, 
            "evidence": evidence_out
        }
