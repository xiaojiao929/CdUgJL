#Amber
# ------------------------------------------------------------------------------
# Copyright (c) 2025 Amber Xiao
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import get_cfg_defaults
from data.dataset import MedicalImageDataset
from models.meamt_net import MEaMtNet
from utils.metrics import compute_all_metrics
from eval.inference import run_inference
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="MEaMt-Net Testing Script")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_results', action='store_true', help='Flag to save test predictions')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    # Setup logger
    logger = setup_logger("MEaMtNet-Test", output=cfg.OUTPUT_DIR, filename="test_log.txt")
    logger.info("Running MEaMt-Net Evaluation")
    logger.info(f"Using configuration: {cfg}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    test_dataset = MedicalImageDataset(cfg.DATA.TEST_DIR, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS)

    # Initialize model
    model = MEaMtNet(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Inference and evaluation
    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            gt_seg = batch['seg'].to(device)
            gt_quant = batch['quant'].to(device)

            pred_seg, pred_quant, _ = run_inference(model, inputs)

            # Compute metrics
            metrics = compute_all_metrics(pred_seg, gt_seg, pred_quant, gt_quant)
            all_metrics.append(metrics)

            # Optionally save predictions
            if args.save_results:
                save_path = os.path.join(cfg.OUTPUT_DIR, "results")
                os.makedirs(save_path, exist_ok=True)
                torch.save(pred_seg.cpu(), os.path.join(save_path, batch['id'][0] + "_seg.pt"))
                torch.save(pred_quant.cpu(), os.path.join(save_path, batch['id'][0] + "_quant.pt"))

    # Aggregate results
    mean_metrics = {key: sum([m[key] for m in all_metrics]) / len(all_metrics) for key in all_metrics[0]}
    logger.info("Evaluation Results:")
    for k, v in mean_metrics.items():
        logger.info(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
