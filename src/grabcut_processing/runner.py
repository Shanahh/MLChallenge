import numpy as np

from src.grabcut_processing.grabcut import grabcut_with_mask, prep_bin_mask_for_gc
from src.knn_baseline.util import load_dataset_grayscale_only, load_dataset_rgb_gray, store_predictions, visualize

SOURCE_PATH_TRAIN_RGB = "../../dataset/training"
SOURCE_PATH_TRAIN_MASKS = "../../dataset/training_knn_k_53"
SOURCE_PATH_TARGET_MASKS = "../../dataset/training_gc"

TEST_MODE = False
i = 10

rgb_images, scribbles, gt, fnames_train, palette = load_dataset_rgb_gray(
    SOURCE_PATH_TRAIN_RGB, "images", "scribbles", ground_truth_dir="ground_truth"
)
knn_masks, *_ = load_dataset_grayscale_only(
    SOURCE_PATH_TRAIN_MASKS, "scribbles", "scribbles", ground_truth_dir="ground_truth"
)

if not TEST_MODE:
    predictions_train = np.stack(
        [grabcut_with_mask(rgb_img, prep_bin_mask_for_gc(knn_mask, scrib)) for rgb_img, knn_mask, scrib in zip(rgb_images, knn_masks, scribbles)],
        axis=0,
    )

    store_predictions(
        predictions_train, SOURCE_PATH_TARGET_MASKS, "images", fnames_train, palette
    )
else:
    predictions_train = np.stack(
        #visualize(rgb_images[i], scribbles[i], gt[i], knn_masks[i])
        [grabcut_with_mask(rgb_images[i], prep_bin_mask_for_gc(knn_masks[i], scribbles[i]))],
        axis=0,
    )

    visualize(rgb_images[i], scribbles[i], gt[i], predictions_train[0])