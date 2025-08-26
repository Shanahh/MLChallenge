import numpy as np
import torch

from src.grabcut_processing.grabcut import batch_prep_prob_mask_for_gc, batch_grabcut_with_mask


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

def model_output_to_mask(output_tensor, threshold=0.1, apply_sigmoid=True):
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
    binary_masks = (probs_np >= threshold).astype(np.uint8)
    return binary_masks

def output_to_mask_postprocess(output_tensor, input_tensor, apply_sigmoid=True):
    """
    Converts model output to prediction and applies postprocessing.
    """
    # Output tensor has shape [B, C, H, W]
    output_tensor = output_tensor.squeeze(1)  # now shape should be [B, H, W] since we have one output channel
    # convert to sigmoid if applicable
    if apply_sigmoid:
        # Apply sigmoid on tensor
        probs = torch.sigmoid(output_tensor)
    else:
        probs = output_tensor

    # Convert to numpy
    probs_np = probs.detach().cpu().numpy()
    rgb_images, _, scribbles = _reconstruct_from_inputs(input_tensor)

    masks_gc_ready = batch_prep_prob_mask_for_gc(probs_np, scribbles)
    binary_masks = batch_grabcut_with_mask(rgb_images, masks_gc_ready)

    return binary_masks

def _reconstruct_from_inputs(inputs):
    """
    Reconstruct batched RGB, grayscale, and scribble images from the input tensor.

    Args:
        inputs (torch.Tensor): Tensor of shape [B, 5, H, W]

    Returns:
        rgb_images: np.ndarray of shape (B, H, W, 3), uint8
        grayscale_images: np.ndarray of shape (B, H, W), uint8
        scribbles: np.ndarray of shape (B, H, W), uint8
    """
    inputs = inputs.detach().cpu().numpy()  # [B, 5, H, W]

    # Split
    rgb = inputs[:, 0:3]  # [B, 3, H, W]
    gray = inputs[:, 3]  # [B, H, W]
    scrib = inputs[:, 4]  # [B, H, W]

    # Reorder channels and rescale to [0,255]
    rgb = np.transpose(rgb, (0, 2, 3, 1)) * 255.0  # (B, H, W, 3)
    gray = gray * 255.0
    scrib = scrib * 255.0

    return rgb.astype(np.uint8), gray.astype(np.uint8), scrib.astype(np.uint8)
