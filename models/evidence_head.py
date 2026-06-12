import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceHead(nn.Module):
    """
    Dirichlet-based evidential uncertainty estimation.

    Maps segmentation logits and quantification predictions to Dirichlet
    concentration parameters (alpha). Epistemic uncertainty is derived from
    the total evidence S = sum(alpha_i) and number of classes K:
        u = K / S
    """

    def __init__(self, num_classes, quant_dim):
        super().__init__()
        self.num_classes = num_classes
        self.quant_dim = quant_dim

        self.seg_evidence = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 1),
            nn.Softplus(),
        )
        self.quant_evidence = nn.Sequential(
            nn.Linear(quant_dim, quant_dim),
            nn.Softplus(),
        )

    def forward(self, seg_logits, quant_pred):
        # Segmentation evidence
        seg_alpha = self.seg_evidence(seg_logits) + 1.0     # alpha >= 1
        seg_S = seg_alpha.sum(dim=1, keepdim=True)
        seg_prob = seg_alpha / seg_S
        seg_uncertainty = self.num_classes / seg_S          # [B, 1, H, W]

        # Quantification evidence
        quant_alpha = self.quant_evidence(quant_pred) + 1.0
        quant_S = quant_alpha.sum(dim=1, keepdim=True)
        quant_uncertainty = self.quant_dim / quant_S        # [B, 1]

        return {
            'seg_alpha': seg_alpha,
            'seg_prob': seg_prob,
            'seg_uncertainty': seg_uncertainty,
            'quant_alpha': quant_alpha,
            'quant_uncertainty': quant_uncertainty,
            'evidence': {'seg': seg_alpha, 'quant': quant_alpha},
            'uncertainty': {'seg': seg_uncertainty, 'quant': quant_uncertainty},
        }
