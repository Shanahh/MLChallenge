import numpy as np
import cv2

def prep_bin_mask_for_gc(mask: np.ndarray, scribble_mask: np.ndarray) -> np.ndarray:
    """
    Prepare GrabCut mask from KNN mask and scribble mask.

    Args:
        mask: Binary mask (0 background, 1 foreground), shape (H, W).
        scribble_mask: Scribble annotations with:
            0 = definite background scribble
            1 = definite foreground scribble
            255 = no scribble (unknown)

    Returns:
        mask_gc: GrabCut mask with labels:
            cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD
    """
    mask_gc = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)  # default probable background

    # Assign definite foreground/background from scribbles
    mask_gc[scribble_mask == 1] = cv2.GC_FGD
    mask_gc[scribble_mask == 0] = cv2.GC_BGD

    # Assign probable foreground/background for non-scribbled pixels using mask
    no_scribble = (scribble_mask == 255)
    mask_gc[np.logical_and(no_scribble, mask == 1)] = cv2.GC_PR_FGD
    mask_gc[np.logical_and(no_scribble, mask == 0)] = cv2.GC_PR_BGD

    return mask_gc

def prep_prob_mask_for_gc(prob_mask: np.ndarray, scribble_mask: np.ndarray, threshold_lo=0.1, threshold_hi=0.9, threshold_normal=0.5) -> np.ndarray:
    """
    Prepare probability mask for grabCut algorithm postprocessing.
    Assign definite foreground if the sigmoid value is > threshold_hi.
    Assign definite background if the sigmoid value is < threshold_lo.
    Else assign probable foreground/ background using the standard threshold.
    """
    mask_gc = np.full(prob_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)  # default probable background

    # Assign definite foreground/background from scribbles
    mask_gc[scribble_mask == 1] = cv2.GC_FGD
    mask_gc[scribble_mask == 0] = cv2.GC_BGD

    no_scribble = (scribble_mask == 255)

    # Assign definite foreground/background from mask with thresholds
    mask_gc[np.logical_and(no_scribble, prob_mask >= threshold_hi)] = cv2.GC_FGD
    mask_gc[np.logical_and(no_scribble, prob_mask <= threshold_lo)] = cv2.GC_BGD

    not_definite = np.logical_and(mask_gc != cv2.GC_FGD, mask_gc != cv2.GC_BGD)

    # Assign probable foreground/background probabilities from mask with standard threshold
    mask_gc[np.logical_and(not_definite, prob_mask > threshold_normal)] = cv2.GC_PR_FGD
    mask_gc[np.logical_and(not_definite, prob_mask <= threshold_normal)] = cv2.GC_PR_BGD

    return mask_gc

def batch_prep_prob_mask_for_gc(
    prob_masks: np.ndarray,
    scribble_masks: np.ndarray,
    threshold_lo=0.1,
    threshold_hi=0.9,
    threshold_normal=0.5
) -> np.ndarray:
    """
    Apply prep_prob_mask_for_gc to a batch of probability masks + scribble masks.
    """
    batch_size = prob_masks.shape[0]
    processed = []

    for i in range(batch_size):
        mask_gc = prep_prob_mask_for_gc(
            prob_masks[i],
            scribble_masks[i],
            threshold_lo=threshold_lo,
            threshold_hi=threshold_hi,
            threshold_normal=threshold_normal
        )
        processed.append(mask_gc)

    return np.stack(processed, axis=0).astype(np.uint8)

def grabcut_with_mask(image: np.ndarray, mask_gc: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    Run GrabCut given an image and a preprocessed GrabCut mask.

    Args:
        image: RGB image (H, W, 3), dtype=np.uint8.
        mask_gc: GrabCut mask prepared by grabcut_preprocessing().
        iterations: number of GrabCut iterations.

    Returns:
        final_mask: binary mask (0 background, 255 foreground), dtype=np.uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")
    if mask_gc.shape != image.shape[:2]:
        raise ValueError("mask_gc must match image height and width")

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut with mask
    cv2.grabCut(image, mask_gc, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)

    # Return binary mask: foreground=1, background=0
    final_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    return final_mask

def batch_grabcut_with_mask(images: np.ndarray, masks_gc: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    Apply grabcut_with_mask to a batch of images + preprocessed gc masks.
    """
    batch_size = images.shape[0]
    processed = []

    for i in range(batch_size):
        mask_post_gc = grabcut_with_mask(
            images[i],
            masks_gc[i],
            iterations=iterations
        )
        processed.append(mask_post_gc)

    return np.stack(processed, axis=0).astype(np.uint8)