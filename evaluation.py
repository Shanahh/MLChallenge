from typing import List
import numpy as np

def object_iou(y_true: List[np.ndarray], y_pred: List[np.ndarray]):
    """
    Takes lists of model predictions and ground truth labels and returns object IoU score.
    """
    ious = []
    for gt, pred in zip(y_true, y_pred):
        gt_object = gt > 0
        pred_object = pred > 0

        intersection = np.logical_and(gt_object, pred_object).sum()
        union = np.logical_or(gt_object, pred_object).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0  # Both are empty = perfect match
        else:
            iou = intersection / union

        ious.append(iou)

    return float(np.mean(ious))

def background_iou(y_true: List[np.ndarray], y_pred: List[np.ndarray]):
    """
    Takes lists of model predictions and ground truth labels and returns background IoU score.
    """
    ious = []
    for gt, pred in zip(y_true, y_pred):
        gt_bg = gt == 0
        pred_bg = pred == 0

        intersection = np.logical_and(gt_bg, pred_bg).sum()
        union = np.logical_or(gt_bg, pred_bg).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return float(np.mean(ious))

def mean_iou(y_true: List[np.ndarray], y_pred: List[np.ndarray]):
    return (object_iou(y_true, y_pred) + background_iou(y_true, y_pred))/2.0
