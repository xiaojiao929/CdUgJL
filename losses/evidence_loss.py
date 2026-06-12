import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_dirichlet(alpha, num_classes):
    """KL(Dir(alpha) || Dir(1)) regularization."""
    sum_alpha = alpha.sum(dim=1)
    lgamma_sum = torch.lgamma(sum_alpha)
    lgamma_k = torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
    lgamma_alpha = torch.lgamma(alpha).sum(dim=1)
    lgamma_ones = torch.lgamma(torch.ones_like(alpha)).sum(dim=1)

    kl = (lgamma_sum - lgamma_k - lgamma_alpha + lgamma_ones
          + ((alpha - 1.0) * (torch.digamma(alpha)
             - torch.digamma(sum_alpha.unsqueeze(1)))).sum(dim=1))
    return kl.mean()


class EvidenceLoss(nn.Module):
    """
    Combines uncertainty-weighted MSE (segmentation) and KL regularization
    for the Dirichlet-based evidential uncertainty head.
    """

    def __init__(self, beta=0.6, kl_weight=0.01):
        super().__init__()
        self.beta = beta
        self.kl_weight = kl_weight

    def forward(self, outputs, seg_target, quant_target=None):
        device = seg_target.device
        total = torch.tensor(0.0, device=device)

        if 'seg_alpha' not in outputs:
            return total

        seg_alpha = outputs['seg_alpha']                    # [B, C, H, W]
        seg_uncertainty = outputs['seg_uncertainty']        # [B, 1, H, W]
        num_cls = seg_alpha.shape[1]

        # Uncertainty-weighted segmentation CE
        seg_prob = seg_alpha / seg_alpha.sum(dim=1, keepdim=True)
        ce = F.cross_entropy(torch.log(seg_prob + 1e-8), seg_target.long(), reduction='none')
        weight = (1.0 - seg_uncertainty.squeeze(1)).clamp(0, 1)
        total = total + (ce * weight).mean()

        # KL regularization on segmentation Dirichlet
        alpha_flat = seg_alpha.permute(0, 2, 3, 1).reshape(-1, num_cls)
        total = total + self.kl_weight * kl_divergence_dirichlet(alpha_flat, num_cls)

        # Quantification evidence loss — skip entries with target == -1
        if quant_target is not None and 'quant_alpha' in outputs:
            valid = (quant_target >= 0).all(dim=1)           # [B]
            if valid.any():
                quant_alpha = outputs['quant_alpha'][valid]  # [B', quant_dim]
                qt = quant_target[valid].float()
                quant_S = quant_alpha.sum(dim=1, keepdim=True)
                quant_mean = quant_alpha / quant_S
                quant_loss = F.mse_loss(quant_mean, qt)
                total = total + self.beta * quant_loss
                total = total + self.kl_weight * kl_divergence_dirichlet(quant_alpha, quant_alpha.shape[1])

        return total
