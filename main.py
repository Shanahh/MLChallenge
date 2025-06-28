import numpy as np
from util import load_dataset
from util import store_predictions


images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/training", "images", "scribbles", "ground_truth"
)

test = np.stack(
        images_train[0],
        axis=0
    )

store_predictions(
    test, "dataset", "augmentations", fnames_train, palette
)