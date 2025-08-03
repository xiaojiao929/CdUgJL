
# Contrastive-Driven and Uncertainty-Guided Joint Learning for Semi-Supervised Liver Tumor Segmentation,Uncertainty and Quantification
<img width="654" height="269" alt="method" src="https://github.com/user-attachments/assets/cf8f853e-06d7-4143-a3be-16d23995400d" />

## 🔗 Dataset
Our method achieves state-of-the-art performance on the [LLD-MMRI dataset](https://github.com/LMMMEng/LLD-MMRI-Dataset) under both fully- and semi-supervised settings.

## 🗂️Project Structure
MEaMt-Net/
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
│   ├── metrics.py
│   ├── logger.py
│   ├── visualizer.py
│   └── scheduler.py
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






