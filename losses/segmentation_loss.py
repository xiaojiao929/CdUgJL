#Amber
# MIT License
# Copyright (c) 2025 Amber Xiao

"""
Dice + Cross-Entropy loss implementation for segmentation tasks.
This compound loss improves both region-level overlap and pixel-wise classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    """
    Combined Dice loss and Cross-Entropy loss for segmentation tasks.
    """

    def __init__(self, weight_dice=1.0, weight_ce=1.0, smooth=1e-5):
        """
        Args:
            weight_dice (float): Weight for Dice loss component.
            weight_ce (float): Weight for Cross-Entropy loss component.
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted segmentation map (N, C, H, W).
            target (Tensor): Ground truth label map (N, H, W).

        Returns:
            Tensor: Combined loss value.
        """
        if pred.size(1) == 1:
            pred_soft = torch.sigmoid(pred)
            target_onehot = target.unsqueeze(1).float()
        else:
            pred_soft = F.softmax(pred, dim=1)
            target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # Dice Loss
        intersection = torch.sum(pred_soft * target_onehot, dim=(2, 3))
        union = torch.sum(pred_soft + target_onehot, dim=(2, 3))
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        # Cross Entropy Loss
        if pred.size(1) == 1:
            ce_loss = F.binary_cross_entropy(pred_soft, target_onehot)
        else:
            ce_loss = F.cross_entropy(pred, target)

        # Combined Loss
        total_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        return total_loss
