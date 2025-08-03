#Amber
# Copyright (c) 2025 Amber Xiao

"""
evidence_loss.py

This module implements evidential loss functions for uncertainty-aware learning.
Specifically, it includes the Evidential Mean Square Error (EMSE) and
the Kullback–Leibler divergence for regularizing uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def relu_evidence(logits):
    """
    Converts raw network outputs into evidence using ReLU activation.
    Evidence is used to parameterize the Dirichlet distribution.
    """
    return F.relu(logits)


def kl_divergence(alpha, num_classes, coeff=1.0):
    """
    Computes the KL divergence between the predicted Dirichlet distribution and the uniform Dirichlet prior.

    Args:
        alpha (Tensor): Predicted Dirichlet parameters of shape (B, C)
        num_classes (int): Number of classes (C)
        coeff (float): Regularization coefficient for KL loss
    Returns:
        Tensor: KL divergence loss
    """
    beta = torch.ones((1, num_classes), dtype=torch.float32).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return coeff * kl.mean()


def edl_mse_loss(predict_logits, targets, epoch, num_classes, annealing_step):
    """
    Computes the evidential mean square error loss with KL regularization.

    Args:
        predict_logits (Tensor): Raw network outputs before softmax (B, C)
        targets (Tensor): Ground truth labels (B,)
        epoch (int): Current training epoch
        num_classes (int): Total number of classes
        annealing_step (int): Number of epochs for KL annealing
    Returns:
        Tensor: Total evidential loss (MSE + KL)
    """
    evidence = relu_evidence(predict_logits)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    pred = alpha / S

    # One-hot encode the ground truth
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

    # Mean squared error between predictions and one-hot labels
    mse_loss = torch.sum((targets_one_hot - pred) ** 2, dim=1, keepdim=True)
    var = alpha * (S - alpha) / (S * S * (S + 1))
    mse_loss += torch.sum(var, dim=1, keepdim=True)

    # Annealing coefficient for KL term
    annealing_coeff = min(1.0, epoch / annealing_step)
    kl_loss = kl_divergence(alpha, num_classes)

    return mse_loss.mean() + annealing_coeff * kl_loss


class EvidentialLoss(nn.Module):
    """
    Wrapper class for evidential classification loss used in uncertainty modeling.
    """
    def __init__(self, num_classes=4, annealing_step=10):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def forward(self, predict_logits, targets, epoch):
        """
        Compute evidential loss.

        Args:
            predict_logits (Tensor): Raw output from evidence head (B, C)
            targets (Tensor): Ground truth labels (B,)
            epoch (int): Current training epoch

        Returns:
            Tensor: Evidential loss
        """
        return edl_mse_loss(predict_logits, targets, epoch, self.num_classes, self.annealing_step)
