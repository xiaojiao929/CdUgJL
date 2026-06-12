# Contrastive-Driven and Uncertainty-Guided Joint Learning for Semi-Supervised Liver Tumor Segmentation and Quantification on Non-Contrast MRI

## Highlights

- **Cross-modal Knowledge Distillation**: Transfers knowledge from contrast-enhanced MRI (CE-MRI) to non-contrast modalities (T2FS, DWI).
- **Evidential Uncertainty Modeling**: Dirichlet-based quantification of prediction reliability for both segmentation and quantification tasks.
- **Joint Learning Framework**: Simultaneous tumor segmentation and biomarker quantification (X, Y, Area).
- **Contrastive Representation Learning**: InfoNCE-based cross-modal feature alignment using teacher-student positive/negative pairs.
- **Semi-supervised**: Trains with 10% or 20% labeled data.

## Dataset

[LLD-MMRI dataset](https://github.com/LMMMEng/LLD-MMRI-Dataset)

## Project Structure

```
CdUgJL/
├── configs/
│   ├── __init__.py
│   ├── default.py
│   └── default.yaml
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── meamt_net.py       # Backbone: Edge-Guided Attention + Mamba encoder/decoder
│   ├── decoders.py        # SegDecoder + QuantDecoder
│   ├── evidence_head.py   # Dirichlet-based uncertainty head
│   └── distillation.py    # Teacher-student distillation + contrastive projector
├── losses/
│   ├── __init__.py
│   ├── segmentation_loss.py   # DiceCELoss
│   ├── distillation_loss.py   # Feature KD + InfoNCE contrastive loss
│   └── evidence_loss.py       # Uncertainty-weighted + KL Dirichlet loss
├── eval/
│   ├── __init__.py
│   ├── inference.py           # run_inference (with TTA) + evaluate_model
│   └── evaluation_metrics.py  # Standalone evaluation script
├── utils/
│   ├── __init__.py
│   ├── metrics.py    # Dice, HD95, ASD, MAE
│   ├── logger.py     # Logger + TensorBoard
│   └── scheduler.py  # Cosine / step / poly LR schedulers
├── scripts/
│   ├── run_train.sh
│   └── run_test.sh
├── metrics.py
├── train.py
├── test.py
├── requirements.txt
├── environment.yml
└── setup.py
```

## Installation

```bash
conda env create -f environment.yml
conda activate meamt-net
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Data Preparation

Organize your dataset as follows:

```
data/LLD-MMRI/
  train/
    patient_001/
      T2FS.npy    # [H, W] float32
      DWI.npy
      seg.npy     # [H, W] int64, 0=background, 1=tumor
      quant.npy   # [3] float32: (x_center, y_center, area), normalized to [0,1]
  val/
    ...
  test/
    ...
```

## Training

```bash
python train.py --config configs/default.yaml
```

Resume from checkpoint:

```bash
python train.py --config configs/default.yaml --resume checkpoints/meamtnet_best.pth
```

## Testing

```bash
python test.py --config configs/default.yaml \
               --checkpoint checkpoints/meamtnet_best.pth \
               --save_results --tta
```

## Evaluation

```bash
python eval/evaluation_metrics.py \
  --ground_truth ./data/LLD-MMRI/test \
  --predictions ./outputs
```

## Metrics

| Task | Metric |
|------|--------|
| Segmentation | Dice, HD95, ASD |
| Quantification | MAE (X, Y, Area) |
