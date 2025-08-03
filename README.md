
# Contrastive-Driven and Uncertainty-Guided Joint Learning for Semi-Supervised Liver Tumor Segmentation,Uncertainty and Quantification
<img width="654" height="269" alt="method" src="https://github.com/user-attachments/assets/cf8f853e-06d7-4143-a3be-16d23995400d" />

## 🔗 Dataset
Our method achieves state-of-the-art performance on the [LLD-MMRI dataset](https://github.com/LMMMEng/LLD-MMRI-Dataset) under both fully- and semi-supervised settings.


## 📁 Project Structure
```
CdUgJL/
├── README.md  
├── requirements.txt  
├── setup.py  
├── configs/
│   └── default.yaml  
├── data/
│   ├── __init__.py  
│   ├── dataset.py  
│   └── transforms.py  
├── models/
│   ├── __init__.py  
│   ├── meamt_net.py  
│   ├── decoders.py  
│   ├── evidence_head.py  
│   └── distillation.py  
├── losses/
│   ├── __init__.py  
│   ├── segmentation_loss.py  
│   ├── distillation_loss.py  
│   └── evidence_loss.py  
├── utils/
│   └── metrics.py  
├── train.py  
├── test.py  
├── eval/
│   ├── inference.py  
│   └── evaluation_metrics.py  
├── scripts/
│   ├── run_train.sh  
│   └── run_test.sh  
├── checkpoints/
│   └── meamtnet_best.pth  
└── docs/
    ├── architecture.md  
    └── ablation_results.md  
```

## Key Modules

### 1. Backbone Network: Edge-Guided Attention + Mamba

- The backbone integrates edge-enhanced spatial attention to highlight tumor boundaries and employs **Mamba** (a state-space sequence model) to capture long-range dependencies and temporal dynamics.
- Inputs: Non-contrast MRI modalities (e.g., T2FS + DWI)
- Output: Deep feature maps for downstream tasks

### 2. Multi-task Decoders

- **Segmentation Decoder:** Predicts tumor masks with hierarchical feature fusion
- **Quantification Decoder:** Estimates spatial (X, Y) and area measurements using regression heads

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

Please take a look at the figure in the main paper (Figure 1) or supplementary file for a full visual depiction of the pipeline.









