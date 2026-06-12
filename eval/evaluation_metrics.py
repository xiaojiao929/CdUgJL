import argparse
import os
import numpy as np
import torch
from pathlib import Path


def evaluate_segmentation(pred_dir, gt_dir, num_classes=2):
    from utils.metrics import dice_score, hausdorff95, average_surface_distance

    results = {'dice': [], 'hd95': [], 'asd': []}
    pred_files = sorted(Path(pred_dir).glob('*_seg.pt'))

    for pf in pred_files:
        pid = pf.stem.replace('_seg', '')
        gf = Path(gt_dir) / f'{pid}_seg.npy'
        if not gf.exists():
            continue
        pred = torch.load(pf)
        gt = torch.from_numpy(np.load(gf))
        pred_cls = pred.argmax(dim=0) if pred.dim() == 3 else pred
        results['dice'].append(dice_score(pred.unsqueeze(0), gt.unsqueeze(0), num_classes))
        results['hd95'].append(hausdorff95(pred_cls.numpy(), gt.numpy()))
        results['asd'].append(average_surface_distance(pred_cls.numpy(), gt.numpy()))

    return {k: float(np.nanmean(v)) for k, v in results.items()}


def evaluate_quantification(pred_dir, gt_dir):
    from utils.metrics import mae_quantification

    preds, gts = [], []
    for pf in sorted(Path(pred_dir).glob('*_quant.pt')):
        pid = pf.stem.replace('_quant', '')
        gf = Path(gt_dir) / f'{pid}_quant.npy'
        if not gf.exists():
            continue
        preds.append(torch.load(pf).numpy())
        gts.append(np.load(gf))

    if not preds:
        return {'mae_x': float('nan'), 'mae_y': float('nan'), 'mae_area': float('nan')}

    preds = np.stack(preds)
    gts = np.stack(gts)
    return {
        'mae_x': float(np.abs(preds[:, 0] - gts[:, 0]).mean()),
        'mae_y': float(np.abs(preds[:, 1] - gts[:, 1]).mean()),
        'mae_area': float(np.abs(preds[:, 2] - gts[:, 2]).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    seg_metrics = evaluate_segmentation(args.predictions, args.ground_truth, args.num_classes)
    quant_metrics = evaluate_quantification(args.predictions, args.ground_truth)

    print("Segmentation Metrics:")
    for k, v in seg_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Quantification Metrics:")
    for k, v in quant_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
