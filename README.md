
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
1. Create Environment
You can install the required dependencies via either requirements.txt or environment.yml.

# Option 1: pip (recommended)
pip install -r requirements.txt

# Option 2: conda (optional)
conda env create -f environment.yml
conda activate cd_ugjl

2. Download Dataset
Download the LLD-MMRI dataset from: 👉 https://github.com/LMMMEng/LLD-MMRI-Dataset

Place the dataset under the data/ directory and modify the path in configs/default.yaml accordingly:
data_root: ./data/LLD-MMRI

3. Train the Model
Run the training script:
bash scripts/run_train.sh
Or directly:
python train.py --config default.yaml

4. Evaluate the Model
Run the test script:
bash scripts/run_test.sh









