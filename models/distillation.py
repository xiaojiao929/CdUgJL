#Amber
# MIT License
# Copyright (c) 2025 Amber Xiao

"""
distillation.py

Implements the uncertainty-aware knowledge distillation mechanism for MEaMt-Net.
It incorporates uncertainty-based guidance and consistency regularization between teacher and student models.
This file handles both feature-level and prediction-level distillation.

Author: Amber Xiao
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyAwareDistiller(nn.Module):
    """
    Module for uncertainty-guided knowledge distillation.
    Incorporates logit-based distillation and consistency loss using uncertainty weights.
    """

    def __init__(self, temperature=2.0):
        super(UncertaintyAwareDistiller, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='none')  # used for soft logit matching

    def forward(self, student_logits, teacher_logits, uncertainty=None):
        """
        Compute the distillation loss between teacher and student logits,
        optionally guided by uncertainty.

        Args:
            student_logits (Tensor): [B, C, H, W] logits from student model
            teacher_logits (Tensor): [B, C, H, W] logits from teacher model
            uncertainty (Tensor or None): [B, 1, H, W] uncertainty map (lower => confident)

        Returns:
            distill_loss (Tensor): scalar loss value
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # Compute pixel-wise KL divergence
        kl = self.kl_div(student_soft, teacher_soft).sum(1, keepdim=True)  # [B, 1, H, W]

        if uncertainty is not None:
            # Normalize uncertainty map to [0,1]
            normalized_uncertainty = torch.sigmoid(-uncertainty)  # higher confidence → higher weight
            kl = kl * normalized_uncertainty

        return kl.mean()

class FeatureConsistencyLoss(nn.Module):
    """
    Enforces similarity between intermediate features of teacher and student models.
    """

    def __init__(self, reduction='mean'):
        super(FeatureConsistencyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, student_feat, teacher_feat, mask=None):
        """
        Compute L2 loss between student and teacher feature maps.

        Args:
            student_feat (Tensor): [B, C, H, W]
            teacher_feat (Tensor): [B, C, H, W]
            mask (Tensor or None): Optional weighting mask [B, 1, H, W]

        Returns:
            feature_loss (Tensor): scalar loss
        """
        diff = (student_feat - teacher_feat) ** 2  # [B, C, H, W]

        if mask is not None:
            diff = diff * mask

        loss = diff.mean() if self.reduction == 'mean' else diff.sum()
        return loss
