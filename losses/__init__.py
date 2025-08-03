#Amber
# MIT License
# Copyright (c) 2025 Amber Xiao

"""
__init__.py for the losses module.

This file exposes all loss function classes used in MEaMt-Net, including:
- Segmentation loss
- Evidence loss
- Distillation loss
"""

from .segmentation_loss import DiceCELoss
from .evidence_loss import EvidenceLoss
from .distillation_loss import (
    UncertaintyAwareDistiller,
    FeatureConsistencyLoss,
)

__all__ = [
    "DiceCELoss",
    "EvidenceLoss",
    "UncertaintyAwareDistiller",
    "FeatureConsistencyLoss",
]
