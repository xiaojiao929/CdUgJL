import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureKDLoss(nn.Module):
    """MSE loss between teacher and student feature projections."""

    def forward(self, student_proj, teacher_proj):
        if teacher_proj is None:
            return torch.tensor(0.0, device=student_proj.device)
        return F.mse_loss(student_proj, teacher_proj.detach())


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss.
    Positive pair: teacher-student projections from the same sample.
    Negative pairs: projections from other samples in the batch.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_proj, teacher_proj):
        if teacher_proj is None:
            return torch.tensor(0.0, device=student_proj.device)

        B = student_proj.shape[0]
        s = F.normalize(student_proj, dim=1)
        t = F.normalize(teacher_proj.detach(), dim=1)

        logits = torch.mm(s, t.T) / self.temperature      # [B, B]
        labels = torch.arange(B, device=s.device)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        return (loss_s2t + loss_t2s) / 2.0


class DistillationLoss(nn.Module):
    def __init__(self, temperature=0.07, feat_weight=1.0, contra_weight=1.0):
        super().__init__()
        self.feat_kd = FeatureKDLoss()
        self.contrastive = ContrastiveLoss(temperature)
        self.feat_w = feat_weight
        self.contra_w = contra_weight

    def forward(self, outputs):
        s_proj = outputs.get('student_proj')
        t_proj = outputs.get('teacher_proj')
        if s_proj is None or t_proj is None:
            return torch.tensor(0.0)
        feat_loss = self.feat_kd(s_proj, t_proj)
        contra_loss = self.contrastive(s_proj, t_proj)
        return self.feat_w * feat_loss + self.contra_w * contra_loss
