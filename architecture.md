# Architecture Overview – CdUgJL

## Overview

CdUgJL (Contrast-driven Uncertainty-guided Joint Learning) is a unified multi-task learning framework designed to perform liver tumor segmentation, quantification, and classification from limited-labeled multi-modal MRI data. The core of CdUgJL is an edge-guided attention backbone combined with a Mamba-based global modeling module, enhanced by uncertainty-aware consistency and cross-modal knowledge distillation.

---

## Key Modules

### 1. Backbone Network: Edge-Guided Attention + Mamba

- The backbone integrates edge-enhanced spatial attention to highlight tumor boundaries and employs **Mamba** (a state-space sequence model) to capture long-range dependencies and temporal dynamics.
- Inputs: Non-contrast MRI modalities (e.g., T2FS + DWI)
- Output: Deep feature maps for downstream tasks

### 2. Multi-task Decoders

- **Segmentation Decoder:** Predicts tumor masks with hierarchical feature fusion
- **Quantification Decoder:** Estimates spatial (X, Y) and area measurements using regression heads
- **Classification Head (optional):** Predicts categorical tumor attributes (e.g., benign vs malignant)

### 3. Evidential Uncertainty Estimation

- Implemented via a dedicated `evidence_head.py`
- Uses Dirichlet-based modeling to produce confidence-aware predictions
- Contributes to both supervised loss (e.g., MSE) and regularization (KL divergence with prior)

### 4. Cross-modal Knowledge Distillation (CdUgJL Module)

- **Teacher:** Fully-supervised model trained on contrast-enhanced MRI (not used at inference)
- **Student:** Learns from both labeled and unlabeled non-contrast images
- **Distillation Losses:**
  - Feature-level KD with attention alignment
  - Contrastive loss for instance discrimination using positive (same region) and negative (different region) pairs
  - KL divergence on evidence outputs

### 5. Uncertainty-guided Consistency Learning

- Guides student training using reliable regions (low epistemic uncertainty)
- Consistency enforced between weak/strong augmentations across spatial and task dimensions

---

## Training and Evaluation

- **Training Modes:**
  - Fully-supervised: With 100% labels
  - Semi-supervised: With 10% or 20% labels + unlabeled data

- **Metrics:**
  - Segmentation: Dice, HD95, ASD
  - Quantification: MAE of X, Y, and Area
  - Classification (optional): Accuracy, AUC

- **Datasets:**
  - Public: [LLD-MMRI](https://github.com/LMMMEng/LLD-MMRI-Dataset)
  - Private: McGill In-house Dataset (not released)

---

## Diagram

Please refer to the figure in the main paper (Figure 1) or supplementary file for a full visual depiction of the pipeline.

---

## Reference

If you use this code or architecture, please cite:

