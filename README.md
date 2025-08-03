
# Contrastive-Driven and Uncertainty-Guided Joint Learning for Semi-Supervised Liver Tumor Segmentation,Uncertainty and Quantification
<img width="654" height="269" alt="method" src="https://github.com/user-attachments/assets/cf8f853e-06d7-4143-a3be-16d23995400d" />

## 🚀 Highlights

- **Cross-modal Knowledge Distillation**: Transfers knowledge from contrast-enhanced MRI (CEMRI) to non-contrast images (T2FS, DWI).
- **Evidential Uncertainty Modeling**: Quantifies prediction reliability for segmentation and quantification.
- **Joint Learning Framework**: Supports simultaneous tumor segmentation and biomarker quantification.
- **Contrastive Representation Learning**: Enhances cross-modal feature alignment using positive/negative pair design.

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
├── metrics.py 
├── train.py  
├── test.py  
├── eval/
│   ├── inference.py  
│   └── evaluation_metrics.py  
├── scripts/
│   ├── run_train.sh  
│   └── run_test.sh  
└── docs/
    ├── architecture.md  
  
```

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/xiaojiao929/CdUgJL.git
cd CdUgJL

### 2. Set up environment (via Conda)

conda env create -f environment.yml
conda activate cd_ugjl

Or use pip:

pip install -r requirements.txt

🏋️‍♂️ Training
To train the model from scratch:
python train.py --config configs/default.yaml

To resume from a checkpoint:
python train.py --config default.yaml --resume checkpoints/meamtnet_best.pth

🔍 Testing
To evaluate the model:
python test.py --config default.yaml --checkpoint checkpoints/meamtnet_best.pth

###  5.Compute segmentation and quantification metrics:
python eval/evaluation_metrics.py \
  --ground_truth ./data/test \
  --predictions ./outputs/




