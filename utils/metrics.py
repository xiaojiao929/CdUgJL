#Amber
# Copyright (c) 2025 Amber Xiao

"""
metrics.py

This module implements evaluation metrics used for segmentation and quantification tasks,
including DSC, HD95, ASD, MAE of center point (X, Y), max-diameter (D), and area (S).
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
from skimage import measure


def dice_score(pred, gt):
    """
    Compute Dice Similarity Coefficient (DSC).

    Args:
        pred (ndarray): Predicted binary mask
        gt (ndarray): Ground truth binary mask

    Returns:
        float: Dice coefficient
    """
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    intersection = np.logical_and(pred, gt).sum()
    return 2. * intersection / (pred.sum() + gt.sum() + 1e-8)


def hd95(pred, gt):
    """
    Compute 95th percentile Hausdorff Distance (HD95).

    Args:
        pred (ndarray): Predicted binary mask
        gt (ndarray): Ground truth binary mask

    Returns:
        float: HD95
    """
    if np.count_nonzero(pred) == 0 or np.count_nonzero(gt) == 0:
        return 0.0

    pred_contour = measure.find_contours(pred, 0.5)[0]
    gt_contour = measure.find_contours(gt, 0.5)[0]

    hd1 = directed_hausdorff(pred_contour, gt_contour)[0]
    hd2 = directed_hausdorff(gt_contour, pred_contour)[0]

    return np.percentile([hd1, hd2], 95)


def asd(pred, gt):
    """
    Compute Average Surface Distance (ASD).

    Args:
        pred (ndarray): Predicted binary mask
        gt (ndarray): Ground truth binary mask

    Returns:
        float: ASD value
    """
    if np.count_nonzero(pred) == 0 or np.count_nonzero(gt) == 0:
        return 0.0

    pred_contour = measure.find_contours(pred, 0.5)[0]
    gt_contour = measure.find_contours(gt, 0.5)[0]

    distances = []
    for p in pred_contour:
        distances.append(np.min(np.linalg.norm(gt_contour - p, axis=1)))
    for g in gt_contour:
        distances.append(np.min(np.linalg.norm(pred_contour - g, axis=1)))

    return np.mean(distances)


def compute_centroid(mask):
    """
    Compute the centroid (center of mass) of the binary mask.

    Args:
        mask (ndarray): Binary segmentation mask

    Returns:
        tuple: (x, y) coordinates of the centroid
    """
    indices = np.argwhere(mask)
    if indices.shape[0] == 0:
        return (0.0, 0.0)
    y, x = indices.mean(axis=0)
    return (x, y)


def compute_max_diameter(mask):
    """
    Compute the maximum diameter of a binary mask.

    Args:
        mask (ndarray): Binary mask

    Returns:
        float: Maximum Euclidean distance between any two foreground pixels
    """
    coords = np.argwhere(mask)
    if coords.shape[0] <= 1:
        return 0.0
    dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
    return np.max(dists)


def compute_area(mask):
    """
    Compute the area (number of foreground pixels).

    Args:
        mask (ndarray): Binary mask

    Returns:
        float: Area
    """
    return float(np.sum(mask > 0))


def quantification_metrics(pred_mask, gt_mask):
    """
    Compute quantification errors including:
        - Center point Euclidean distance (MD)
        - MAE of X and Y coordinates
        - Max-diameter error
        - Area error

    Args:
        pred_mask (ndarray): Predicted binary mask
        gt_mask (ndarray): Ground truth binary mask

    Returns:
        dict: Dictionary containing all quantification metrics
    """
    pred_cx, pred_cy = compute_centroid(pred_mask)
    gt_cx, gt_cy = compute_centroid(gt_mask)
    md = np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)

    dx = abs(pred_cx - gt_cx)
    dy = abs(pred_cy - gt_cy)
    dD = abs(compute_max_diameter(pred_mask) - compute_max_diameter(gt_mask))
    dS = abs(compute_area(pred_mask) - compute_area(gt_mask))

    return {
        "MD": md,
        "X": dx,
        "Y": dy,
        "D": dD,
        "S": dS,
    }
