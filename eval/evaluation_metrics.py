#Amber
# Copyright (c) 2025 Amber Xiao

import torch
import numpy as np
import torch.nn.functional as F
from medpy.metric.binary import hd95, assd, dc

def compute_segmentation_metrics(pred, target):
    """
    Compute segmentation metrics: Dice, HD95, ASD
    """
    pred_bin = torch.sigmoid(pred) > 0.5
    pred_bin = pred_bin.cpu().numpy().astype(np.uint8)
    target = target.cpu().numpy().astype(np.uint8)

    dice_list, hd95_list, asd_list = [], [], []

    for p, t in zip(pred_bin, target):
        if p.shape != t.shape:
            p = np.resize(p, t.shape)

        try:
            dice_list.append(dc(p, t))
        except:
            dice_list.append(0.0)

        try:
            hd95_list.append(hd95(p, t))
        except:
            hd95_list.append(0.0)

        try:
            asd_list.append(assd(p, t))
        except:
            asd_list.append(0.0)

    return {
        "DSC": np.mean(dice_list),
        "HD95": np.mean(hd95_list),
        "ASD": np.mean(asd_list),
    }

def compute_quantification_metrics(pred, target):
    """
    Compute quantification metrics: MAE for (MD, X, Y, Area)
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    mae = np.abs(pred - target)

    if mae.ndim == 1 or mae.shape[1] == 1:  # scalar regression (e.g., MD)
        return {"MD": float(np.mean(mae))}
    else:
        return {
            "MD": float(np.mean(mae[:, 0])),
            "X": float(np.mean(mae[:, 1])),
            "Y": float(np.mean(mae[:, 2])),
            "Area": float(np.mean(mae[:, 3])),
        }
