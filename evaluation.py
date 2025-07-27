import numpy as np
import torch

def object_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the object IoU score over a batch of predictions and ground truth masks.

    Parameters:
        y_true (np.ndarray): Ground truth masks with shape (batch_size, H, W).
        y_pred (np.ndarray): Predicted masks with shape (batch_size, H, W).

    Returns:
        float: Mean object IoU over the batch.
    """
    ious = []
    for i in range(y_true.shape[0]):
        gt = y_true[i]
        pred = y_pred[i]
        gt_object = gt > 0
        pred_object = pred > 0

        intersection = np.logical_and(gt_object, pred_object).sum()
        union = np.logical_or(gt_object, pred_object).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0  # Both empty = perfect match
        else:
            iou = intersection / union

        ious.append(iou)

    return float(np.mean(ious))


def background_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the background IoU score over a batch of predictions and ground truth masks.

    Parameters:
        y_true (np.ndarray): Ground truth masks with shape (B, H, W).
        y_pred (np.ndarray): Predicted masks with shape (B, H, W).

    Returns:
        float: Mean background IoU over the batch.
    """
    ious = []
    for i in range(y_true.shape[0]):
        gt = y_true[i]
        pred = y_pred[i]
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

def get_ious(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Given labels and prediction values, returns object IoU, background IoU, and mean IoU.
    Input Dim: (B, H, W)
    """
    obj_iou_val = object_iou(y_true, y_pred)
    bkg_iou_val = background_iou(y_true, y_pred)
    mean_iou_val = (obj_iou_val + bkg_iou_val) / 2.0
    return obj_iou_val, bkg_iou_val, mean_iou_val

def model_output_to_mask(output_tensor, threshold=0.5, apply_sigmoid=True):
    """
    Convert model output to binary mask (H x W numpy array)
    Differentiate between raw logit outputs and sigmoid outputs
    Input Dim: (B, 1, H, W)
    Output Dim: Returns a binary mask of shape (B, H, W) with values 0 or 1
    """
    # Output tensor has shape [B, C, H, W]
    output_tensor = output_tensor.squeeze(1) # now shape should be [B, H, W] since we have one output channel
    # convert to sigmoid if applicable
    if apply_sigmoid:
        # Apply sigmoid on tensor
        probs = torch.sigmoid(output_tensor)
    else:
        probs = output_tensor

    # Convert to numpy
    probs_np = probs.detach().cpu().numpy()

    # construct mask
    binary_mask = (probs_np >= threshold).astype(np.uint8)
    return binary_mask
