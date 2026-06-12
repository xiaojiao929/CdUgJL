import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        num_cls = pred.shape[1]
        target_oh = F.one_hot(target.long(), num_cls).permute(0, 3, 1, 2).float()

        inter = (pred * target_oh).sum(dim=(2, 3))
        denom = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_w = dice_weight
        self.ce_w = ce_weight

    def forward(self, pred, target):
        return self.dice_w * self.dice(pred, target) + self.ce_w * self.ce(pred, target.long())
