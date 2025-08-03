#Amber
# Copyright (c) 2025 Amber Xiao

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    KL divergence + Contrastive loss for model component distillation.

    - KL loss encourages alignment of predicted distributions between teacher and student.
    - Contrastive loss encourages matching representations between modalities via InfoNCE.

    Args:
        temperature (float): Temperature for contrastive loss.
        contrastive_weight (float): Weight for the contrastive loss.
        kl_weight (float): Weight for the KL divergence loss.
    """
    def __init__(self, temperature=0.5, contrastive_weight=1.0, kl_weight=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.kl_weight = kl_weight
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, student_feat, teacher_feat):
        """
        Args:
            student_logits (Tensor): Output logits from student model (B, C, H, W)
            teacher_logits (Tensor): Output logits from teacher model (B, C, H, W)
            student_feat (Tensor): Student feature vector (B, D)
            teacher_feat (Tensor): Teacher feature vector (B, D)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of detailed components
        """
        # KL divergence loss
        student_log_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_prob = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = self.kl_criterion(student_log_prob, teacher_prob) * (self.temperature ** 2)

        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(student_feat, teacher_feat)

        total = self.kl_weight * kl_loss + self.contrastive_weight * contrastive_loss
        return total, {
            'kl_loss': kl_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total': total.item()
        }

    def compute_contrastive_loss(self, student, teacher):
        """
        Computes contrastive loss using InfoNCE.
        Positive pairs: student-teacher within same sample.
        Negative pairs: cross-sample mismatches.
        """
        B = student.size(0)
        student = F.normalize(student, dim=1)
        teacher = F.normalize(teacher, dim=1)

        logits = torch.matmul(student, teacher.T) / self.temperature
        labels = torch.arange(B).to(student.device)
        loss = F.cross_entropy(logits, labels)
        return loss
