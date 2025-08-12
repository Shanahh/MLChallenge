import numpy as np
import cv2

def before_grabcut_mask_preprocessing(knn_mask: np.ndarray, scribble_mask: np.ndarray) -> np.ndarray:
    """
    Prepare GrabCut mask from KNN mask and scribble mask.

    Args:
        knn_mask: Binary mask (0 background, 1 foreground), shape (H, W).
        scribble_mask: Scribble annotations with:
            0 = definite background scribble
            1 = definite foreground scribble
            255 = no scribble (unknown)

    Returns:
        mask_gc: GrabCut mask with labels:
            cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD
    """
    mask_gc = np.full(knn_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)  # default probable background

    # Assign definite foreground/background from scribbles
    mask_gc[scribble_mask == 1] = cv2.GC_FGD
    mask_gc[scribble_mask == 0] = cv2.GC_BGD

    # Assign probable foreground/background for non-scribbled pixels using knn_mask
    no_scribble = (scribble_mask == 255)
    mask_gc[np.logical_and(no_scribble, knn_mask == 1)] = cv2.GC_PR_FGD
    mask_gc[np.logical_and(no_scribble, knn_mask == 0)] = cv2.GC_PR_BGD

    return mask_gc


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