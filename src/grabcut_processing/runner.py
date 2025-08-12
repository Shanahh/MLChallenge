import numpy as np

from src.grabcut_processing.grabcut import grabcut_with_mask, before_grabcut_mask_preprocessing
from src.knn_baseline.util import load_dataset_grayscale_only, load_dataset_rgb_gray, store_predictions

SOURCE_PATH_TRAIN_RGB = "../../dataset/training"
SOURCE_PATH_TRAIN_MASKS = "../../dataset/training_knn_k_53"
SOURCE_PATH_TARGET_MASKS = "../../dataset/training_knn_k_53_gc"

rgb_images, scribbles, _, fnames_train, palette = load_dataset_rgb_gray(
    SOURCE_PATH_TRAIN_RGB, "images", "scribbles", ground_truth_dir="ground_truth"
)
knn_masks, *_ = load_dataset_grayscale_only(
    SOURCE_PATH_TRAIN_MASKS, "images", "scribbles", ground_truth_dir="ground_truth"
)

predictions_train = np.stack(
    [grabcut_with_mask(rgb_img, before_grabcut_mask_preprocessing(knn_mask, scrib)) for rgb_img, knn_mask, scrib in zip(rgb_images, knn_masks, scribbles)],
    axis=0,
)

store_predictions(
    predictions_train, SOURCE_PATH_TARGET_MASKS, "images", fnames_train, palette
)