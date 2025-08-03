#Amber
# MIT License
# Copyright (c) 2025 Amber   Xiao


import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDecoder(nn.Module):
    """
    Decoder for semantic segmentation task.
    """
    def __init__(self, in_channels, out_channels, num_classes):
        super(SegmentationDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        out = self.final_conv(x)
        return out


class QuantificationDecoder(nn.Module):
    """
    Decoder for lesion quantification outputs:
    - MD (tumor maximum diameter)
    - X, Y (tumor centroid coordinates)
    - Area (tumor area)
    """
    def __init__(self, in_channels, hidden_dim=64):
        super(QuantificationDecoder, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global feature

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Output 4 regression targets
        self.fc2 = nn.Linear(hidden_dim, 4)  # [MD, X, Y, Area]

    def forward(self, x):
        x = self.pool(x)              # shape: [B, C, 1, 1]
        x = torch.flatten(x, 1)       # shape: [B, C]
        x = self.relu(self.fc1(x))    # shape: [B, hidden_dim]
        out = self.fc2(x)             # shape: [B, 4]
        return out


# Optional: Combine both decoders into a multitask decoder
class MultiTaskDecoder(nn.Module):
    def __init__(self, in_channels, seg_out_channels, num_classes, quant_hidden=64):
        super(MultiTaskDecoder, self).__init__()
        self.seg_decoder = SegmentationDecoder(in_channels, seg_out_channels, num_classes)
        self.quant_decoder = QuantificationDecoder(in_channels, quant_hidden)

    def forward(self, x):
        seg_output = self.seg_decoder(x)
        quant_output = self.quant_decoder(x)
        return seg_output, quant_output
