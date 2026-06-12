import torch
import torch.nn.functional as F
import numpy as np
from utils.metrics import compute_all_metrics


def run_inference(model, inputs, tta=False):
    """
    Run model inference, optionally with test-time augmentation (horizontal + vertical flip).
    Returns (seg_logits, quant_pred, outputs_dict).
    """
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        seg = out['seg']
        quant = out['quant']

        if tta:
            # Horizontal flip
            out_h = model(inputs.flip(-1))
            seg = seg + out_h['seg'].flip(-1)
            quant = quant + out_h['quant']
            # Vertical flip
            out_v = model(inputs.flip(-2))
            seg = seg + out_v['seg'].flip(-2)
            quant = quant + out_v['quant']
            seg = seg / 3.0
            quant = quant / 3.0

    return seg, quant, out


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes=2):
    """Run full validation loop and return mean Dice score."""
    model.eval()
    all_metrics = []

    for batch in loader:
        images = batch['image'].to(device)
        gt_seg = batch['seg'].to(device)
        gt_quant = batch['quant'].to(device)

        pred_seg, pred_quant, _ = run_inference(model, images)

        valid = (gt_seg >= 0).any()
        if not valid:
            continue

        metrics = compute_all_metrics(pred_seg, gt_seg, pred_quant, gt_quant, num_classes)
        all_metrics.append(metrics)

    if not all_metrics:
        return 0.0

    return float(np.mean([m['dice'] for m in all_metrics]))
