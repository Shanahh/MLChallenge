import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from typing import Any
from scipy.ndimage import distance_transform_edt

from util import load_dataset
from util import store_predictions

"""
approach:
Multi Output Logistic Regression
One data point is a list of pixels (the image) where each pixel carries the following information (features): 
1) r,g,b values
2) scribble values (either 0,1 or 255)
3) The distance to the closest foreground scribble
4) The distance to the closest background scribble
5) A value which somehow defines average color deviation from the neighbouring pixels 
"""
####################################
# Helper functions
####################################

def nearest_distance_to_scribble(scribble: np.ndarray, to_foreground: bool) -> np.ndarray:
    """
    Given a scribble mask, return a 2d vector that maps height, width to the distance of the nearest scribble
    The distance measure is Manhattan distance
    @param to_foreground: If true, return the distance to the foreground scribble, else return the distance to the background scribble
    """
    target_value = 1 if to_foreground else 0

    # Create a binary mask where 0s are target scribbles, and everything else is 1
    # This ensures distance_transform_edt computes distances to those 0s
    mask = (scribble != target_value)

    distances = distance_transform_edt(mask)
    return distances

def gradient_per_pixel(image: np.ndarray) -> np.ndarray:
    """
    Given an image, return a 2d vector that maps height, width (a pixel) to the average deviation of the neighbour pixel values
    """
    H, W, C = image.shape
    assert C == 3, "Expected image with 3 color channels (RGB)"

    image = image.astype(np.float32)

    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    center = padded[1:-1, 1:-1]

    # Gather all 8 neighbors
    neighbors = [
        padded[0:-2, 0:-2],  # top-left
        padded[0:-2, 1:-1],  # top
        padded[0:-2, 2:],  # top-right
        padded[1:-1, 0:-2],  # left
        padded[1:-1, 2:],  # right
        padded[2:, 0:-2],  # bottom-left
        padded[2:, 1:-1],  # bottom
        padded[2:, 2:],  # bottom-right
    ]

    diffs_sum = np.zeros((H, W, C), dtype=np.float32)

    for neighbor in neighbors:
        diffs_sum += np.abs(center - neighbor)

    # Average over 8 neighbors and 3 channels
    avg_gradient = diffs_sum.mean(axis=2) / 8

    return avg_gradient

def extract_features(image: np.ndarray, scribble: np.ndarray) -> np.ndarray:
    """
    Performs feature extraction on an image and corresponding scribble mask.
    The features are selected as defined above.
    Returns a 1d vector representing the feature vector for the image.
    """
    # Compute helper features
    dist_fg = nearest_distance_to_scribble(scribble, to_foreground=True)
    dist_bg = nearest_distance_to_scribble(scribble, to_foreground=False)
    gradient = gradient_per_pixel(image)

####################################
# Productive code
####################################

# get data
print("Loading data...")
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/training", "images", "scribbles", "ground_truth"
)

img = images_train[0]




