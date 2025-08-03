#Amber
# Copyright (c) 2025 Amber Xiao

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.meamt_net import MEaMtNet
from data.dataset import MedicalImageDataset
from utils.metrics import compute_segmentation_metrics, compute_quantification_metrics
from utils.visualizer import save_prediction_visuals
from configs.default import get_config

def load_model(checkpoint_path, config, device):
    model = MEaMtNet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    test_dataset = MedicalImageDataset(
        data_root=config.DATA.TEST_DIR,
        phase='test',
        transforms=None  # You can apply test-time transforms if needed
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load trained model
    model = load_model(config.CHECKPOINT_PATH, config, device)

    # Output storage
    results_dir = config.OUTPUT_DIR
    os.makedirs(results_dir, exist_ok=True)

    all_seg_metrics = []
    all_q_metrics = []

    for batch in tqdm(test_loader, desc="Running inference"):
        image = batch['image'].to(device)
        seg_gt = batch['seg'].to(device)
        q_gt = batch['quant'].to(device)

        with torch.no_grad():
            outputs = model(image)
            seg_pred = outputs['seg']
            q_pred = outputs['quant']

        # Compute metrics
        seg_metrics = compute_segmentation_metrics(seg_pred, seg_gt)
        q_metrics = compute_quantification_metrics(q_pred, q_gt)

        all_seg_metrics.append(seg_metrics)
        all_q_metrics.append(q_metrics)

        # Save visualization
        save_prediction_visuals(image, seg_pred, seg_gt, save_dir=results_dir)

    # Aggregate results
    avg_seg_metrics = {k: sum(d[k] for d in all_seg_metrics) / len(all_seg_metrics) for k in all_seg_metrics[0]}
    avg_q_metrics = {k: sum(d[k] for d in all_q_metrics) / len(all_q_metrics) for k in all_q_metrics[0]}

    print("\nAverage Segmentation Metrics:")
    for k, v in avg_seg_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nAverage Quantification Metrics:")
    for k, v in avg_q_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    config = get_config()
    run_inference(config)
