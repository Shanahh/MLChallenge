import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from typing import Any


######### Methods for loading dataset

def _open_image(path, convert_to):
    if convert_to == "RGB":
        return Image.open(path).convert("RGB")
    if convert_to == "grayscale":
        return Image.open(path).convert("L")
    return np.array(Image.open(path))

def _get_file_names(folder):
    return sorted(
        [file for file in os.listdir(folder) if not file.startswith('.')]
    )

def _load_images(folder_path, folder_name, convert_to):
    image_dir_path = os.path.join(folder_path, folder_name)
    filenames = _get_file_names(image_dir_path)
    filepaths = [
        os.path.join(image_dir_path, filename) for filename in filenames
    ]
    return np.stack([_open_image(file, convert_to) for file in filepaths])

def _get_palette(folder_path, ground_truth_dir, filename):
    gt_dir_path = os.path.join(folder_path, ground_truth_dir)
    filepath = os.path.join(gt_dir_path, filename)
    return Image.open(filepath).getpalette()

def _get_filenames(folder_path, scribbles_dir):
    sc_dir_path = os.path.join(folder_path, scribbles_dir)
    filenames = _get_file_names(sc_dir_path)
    return filenames

def load_dataset_gray_twice(
    folder_path: str,
    gc_images_dir: str,
    scribbles_dir: str,
    knn_images_dir: str | None = None,
    ground_truth_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Load images, scribbles, and ground truth masks from a dataset folder.
    
    Args:
        gc_images_dir: folder name for gc images.
        folder_path (str): Path to the dataset folder (e.g., 'dataset/training').
        knn_images_dir (str): folder name for grayscale images.
        scribbles_dir (str): folder name for scribbles.
        ground_truth_dir (str): folder name for ground truth images.
        
    Returns:
        images (np.ndarray): Array of shape (N, H, W, 3) with gc images.
        scribbles (np.ndarray): Array of shape (N, H, W) with scribble labels.
        ground_truth (np.ndarray): Array of shape (N, H, W) with class labels.
        filenames (list[str]): List of filenames for storing predictions
        palette (_type_): _description_
    """
    # Load gc and scribbles
    gc_images = _load_images(folder_path, gc_images_dir, "grayscale")
    scribbles = _load_images(folder_path, scribbles_dir, "grayscale")
    filenames = _get_filenames(folder_path, scribbles_dir)

    # Optionally load grayscale
    knn_images = None
    if knn_images_dir:
        knn_images = _load_images(folder_path, knn_images_dir, "grayscale")

    # If no ground truth, return accordingly
    if ground_truth_dir is None:
        if knn_images is not None:
            return gc_images, knn_images, scribbles, filenames
        return gc_images, scribbles, filenames

    # If ground truth present
    ground_truth = _load_images(folder_path, ground_truth_dir, None)
    palette = _get_palette(folder_path, ground_truth_dir, filenames[0])

    if knn_images is not None:
        return gc_images, knn_images, scribbles, ground_truth, filenames, palette
    return gc_images, scribbles, ground_truth, filenames, palette


def load_dataset_gray_once(
        folder_path: str,
        grayscale_images_dir: str,
        scribbles_dir: str,
        ground_truth_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], Any]:
    """
    Load grayscale images, scribbles, and ground truth masks from a dataset folder.

    Args:
        folder_path (str): Path to the dataset folder (e.g., 'dataset/training').
        grayscale_images_dir (str): folder name for grayscale images.
        scribbles_dir (str): folder name for scribbles.
        ground_truth_dir (str, optional): folder name for ground truth images.

    Returns:
        grayscale_images (np.ndarray): Array of shape (N, H, W) with grayscale images.
        scribbles (np.ndarray): Array of shape (N, H, W) with scribble labels.
        ground_truth (np.ndarray | None): Array of shape (N, H, W) with class labels if ground_truth_dir is provided, else None.
        filenames (list[str]): List of filenames for storing predictions.
        palette (optional): Palette info if ground_truth_dir provided, else None.
    """
    grayscale_images = _load_images(folder_path, grayscale_images_dir, "grayscale")
    scribbles = _load_images(folder_path, scribbles_dir, "grayscale")
    filenames = _get_filenames(folder_path, scribbles_dir)

    ground_truth = None
    palette = None
    if ground_truth_dir is not None:
        ground_truth = _load_images(folder_path, ground_truth_dir, None)
        palette = _get_palette(folder_path, ground_truth_dir, filenames[0])

    return grayscale_images, scribbles, ground_truth, filenames, palette


def store_predictions(
    predictions: np.ndarray,
    folder_path: str,
    predictions_dir: str,
    filenames: list[str],
    palette: Any
):
    """Takes a stack of segmented images and stores them indvidually in the given folder.

    Args:
        predictions (np.ndarray): Array of shape (N, H, W) with predicted class labels.
        folder_path (str): Path to the dataset folder (e.g., 'dataset/training').
        predictions_dir (str): folder name for predictions.
        storage_info (Any): Useful info from load_dataset method for storing.
    """
    pred_dir_path = os.path.join(folder_path, predictions_dir)
    if not os.path.exists(pred_dir_path):
        os.makedirs(pred_dir_path)
    for filename, pred_array in zip(filenames, predictions):
        filepath = os.path.join(pred_dir_path, filename)
        pred_image = Image.fromarray(pred_array.astype(np.uint8), mode='P')
        pred_image.putpalette(palette)
        pred_image.save(filepath)


######### Methods for knn_baseline model

def segment_with_knn(
    image: np.ndarray,
    scribble: np.ndarray,
    k: int = 3
) -> np.ndarray:
    """
    Segment an image using K-Nearest Neighbors classifier based on RGB scribble.

    Parameters:
        image (np.ndarray): Color image of shape (H, W, 3).
        scribble (np.ndarray): Scribble mask of shape (H, W) with values:
                                0 (background), 1 (foreground), 255 (unmarked).
        k (int): Number of neighbors to use in KNN.

    Returns:
        np.ndarray: Predicted segmentation mask of shape (H, W) with values 0 or 1.
    """
    H, W, C = image.shape
    assert C == 3, "Image must be RGB."

    # Reshape image to (H*W, 3)
    image_flat = image.reshape(-1, 3)

    # Flatten scribble mask
    scribbles_flat = scribble.flatten()

    # Create mask for labeled and unlabeled pixels
    labeled_mask = (scribbles_flat != 255)
    unlabeled_mask = (scribbles_flat == 255)

    # Prepare training data
    X_train = image_flat[labeled_mask]
    y_train = scribbles_flat[labeled_mask]

    # Prepare test data
    X_test = image_flat[unlabeled_mask]

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on unlabeled pixels
    y_pred = knn.predict(X_test)

    # Reconstruct full prediction mask
    predicted_mask = np.zeros_like(scribbles_flat)
    predicted_mask[labeled_mask] = y_train
    predicted_mask[unlabeled_mask] = y_pred

    # Reshape to (H, W)
    return predicted_mask.reshape(H, W)

def safe_segment_with_knn(image, scribble, k):
    """Adjust k if there are fewer labeled pixels than k"""
    num_labeled = np.sum(scribble != 255)
    if num_labeled < k:
        print(f"Warning: Only {num_labeled} labeled pixels. Reducing k from {k} to {num_labeled}")
        k = num_labeled if num_labeled > 0 else 1
    return segment_with_knn(image, scribble, k=k)

######### Methods for visualization

def _overlay_scribbles(
    image, scribble, color_fg=(255, 0, 0), color_bg=(0, 0, 255), alpha=0.6
):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB")
    if scribble.shape != image.shape[:2]:
        raise ValueError("Scribble must match image spatial size")
    
    overlaid = image.copy().astype(np.float32)
    
    mask_fg = scribble == 1
    mask_bg = scribble == 0
    
    for mask, color in [(mask_fg, color_fg), (mask_bg, color_bg)]:
        for c in range(3):
            overlaid[..., c][mask] = (
                alpha * color[c] + (1 - alpha) * overlaid[..., c][mask]
            )
    
    return overlaid.astype(np.uint8)

def visualize(
    image: np.ndarray,
    scribbles: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float=0.6
):
    """
    Shows a 1x3 subplot of:
      1. Original image overlaid with scribbles
      2. Ground truth segmentation mask
      3. Prediction mask
      
      Blue = background, Red = foreground.

    Parameters:
        image (H, W, 3)        : RGB image
        scribbles (H, W)       : scribble mask with values {0, 1, 255}
        ground_truth (H, W)    : ground truth mask with values {0, 1}
        prediction (H, W)      : predicted mask with values {0, 1}
        alpha (float)          : alpha blending for overlay
    """
    
    image_with_scribbles = _overlay_scribbles(image, scribbles, alpha=alpha)
    
    cmap = plt.get_cmap('bwr')
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_with_scribbles)
    axes[0].set_title("Image + Scribbles")

    axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Model Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
