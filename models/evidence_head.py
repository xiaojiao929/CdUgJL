#Amber
# MIT License
# Copyright (c) 2025 Amber Xiao

"""



import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidenceHead(nn.Module):
    """
    Evidence-based head for uncertainty-aware segmentation.
    Predicts Dirichlet parameters instead of softmax probabilities.
    """
    def __init__(self, in_channels, out_channels, evidence_fn='relu'):
        super(EvidenceHead, self).__init__()
        self.evidence_fn = evidence_fn

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Returns the Dirichlet evidence (non-negative) for each class.
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        evidence = self.output_conv(x)
        evidence = self.apply_evidence_activation(evidence)
        return evidence

    def apply_evidence_activation(self, x):
        """
        Applies a non-negative activation function to ensure evidence ≥ 0.
        """
        if self.evidence_fn == 'relu':
            return F.relu(x)
        elif self.evidence_fn == 'softplus':
            return F.softplus(x)
        elif self.evidence_fn == 'exp':
            return torch.exp(x)
        else:
            raise NotImplementedError(f"Unsupported evidence function: {self.evidence_fn}")
