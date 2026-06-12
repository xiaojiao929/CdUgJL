import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred, target, num_classes=2, ignore_index=-1):
    """Compute mean Dice score over foreground classes."""
    scores = []
    pred_cls = pred.argmax(dim=1) if pred.dim() == 4 else pred
    for c in range(1, num_classes):
        mask = target != ignore_index
        p = (pred_cls == c) & mask
        t = (target == c) & mask
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        if denom == 0:
            continue
        scores.append((2.0 * inter / denom).item())
    return float(np.mean(scores)) if scores else 0.0


def _to_binary_np(mask, cls=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return (mask == cls).astype(np.uint8)


def hausdorff95(pred, target, cls=1):
    p = _to_binary_np(pred, cls)
    t = _to_binary_np(target, cls)
    if p.sum() == 0 or t.sum() == 0:
        return float('nan')
    dt_p = distance_transform_edt(1 - p)
    dt_t = distance_transform_edt(1 - t)
    d_pt = dt_t[p > 0]
    d_tp = dt_p[t > 0]
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


def average_surface_distance(pred, target, cls=1):
    p = _to_binary_np(pred, cls)
    t = _to_binary_np(target, cls)
    if p.sum() == 0 or t.sum() == 0:
        return float('nan')
    dt_p = distance_transform_edt(1 - p)
    dt_t = distance_transform_edt(1 - t)
    d_pt = dt_t[p > 0].mean()
    d_tp = dt_p[t > 0].mean()
    return float((d_pt + d_tp) / 2.0)


def mae_quantification(pred, target):
    """MAE for X, Y, Area quantification (ignores -1 entries)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    valid = target >= 0
    if not valid.any():
        return float('nan')
    return float(np.abs(pred[valid] - target[valid]).mean())


def compute_all_metrics(pred_seg, gt_seg, pred_quant, gt_quant, num_classes=2):
    pred_cls = pred_seg.argmax(dim=1) if pred_seg.dim() == 4 else pred_seg
    return {
        'dice': dice_score(pred_seg, gt_seg, num_classes),
        'hd95': hausdorff95(pred_cls.cpu().numpy(), gt_seg.cpu().numpy()),
        'asd': average_surface_distance(pred_cls.cpu().numpy(), gt_seg.cpu().numpy()),
        'mae_quant': mae_quantification(pred_quant, gt_quant),
    }
