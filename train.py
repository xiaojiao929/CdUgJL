#Amber
# Copyright (c) 2025 Amber Xiao

"""
train.py

Main training script for MEaMt-Net.
This file initializes models, loads data, defines loss functions, and runs training and validation loops.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse

from models.meamt_net import MEaMtNet
from models.distillation import Distiller
from data.dataset import MedicalImageDataset
from data.transforms import get_transforms
from losses.segmentation_loss import DiceCELoss
from losses.distillation_loss import DistillationLoss
from losses.evidence_loss import EvidenceLoss
from utils.logger import Logger
from utils.scheduler import get_scheduler
from utils.metrics import dice_score
from eval.inference import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MEaMt-Net")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize logger
    logger = Logger(cfg['log_dir'])

    # Load training and validation datasets
    train_tfms, val_tfms = get_transforms()
    train_dataset = MedicalImageDataset(cfg['data_root'], mode='train', transforms=train_tfms, label_ratio=cfg['label_ratio'])
    val_dataset = MedicalImageDataset(cfg['data_root'], mode='val', transforms=val_tfms)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize models
    teacher_model = MEaMtNet(cfg['model']).to(device)
    student_model = MEaMtNet(cfg['model']).to(device)

    # Set teacher model to eval
    teacher_model.eval()

    # Wrap in distillation module
    distiller = Distiller(teacher=teacher_model, student=student_model).to(device)

    # Define loss functions
    loss_seg = DiceCELoss()
    loss_distill = DistillationLoss(temperature=cfg['temperature'], alpha=cfg['alpha'])
    loss_evi = EvidenceLoss(beta=cfg['beta'])

    # Optimizer and scheduler
    optimizer = optim.Adam(distiller.parameters(), lr=cfg['lr'])
    scheduler = get_scheduler(optimizer, cfg)

    best_dice = 0.0

    for epoch in range(cfg['epochs']):
        distiller.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images, masks, quant_labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            outputs = distiller(images)

            # Loss terms
            seg_loss = loss_seg(outputs['seg'], masks)
            distill_loss = loss_distill(outputs, masks)
            evidence_loss = loss_evi(outputs['evidence'], masks, outputs['uncertainty'])

            # Total loss
            total_loss = seg_loss + distill_loss + evidence_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()
        logger.log(f"[Epoch {epoch+1}] Training Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation
        dice = evaluate_model(student_model, val_loader, device)
        logger.log(f"[Epoch {epoch+1}] Validation DSC: {dice:.4f}")

        # Save best model
        if dice > best_dice:
            best_dice = dice
            torch.save(student_model.state_dict(), os.path.join(cfg['save_dir'], 'meamtnet_best.pth'))

    logger.log("Training complete. Best DSC: {:.4f}".format(best_dice))


if __name__ == "__main__":
    main()
