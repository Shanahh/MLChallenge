import os
import joblib  # for saving/loading models efficiently
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from scipy.ndimage import distance_transform_edt

from src.baseline.util import load_dataset_rgb_gray
from src.baseline.util import store_predictions

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
    The distance measure is euclidian distance
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

def feature_encode(image: np.ndarray, scribble: np.ndarray) -> np.ndarray:
    """
    Performs feature extraction on an image and corresponding scribble mask.
    The features are selected as defined above.
    Returns a 1d vector representing the feature vector for the image.
    """
    # Compute helper features
    dist_fg = nearest_distance_to_scribble(scribble, to_foreground=True)
    dist_bg = nearest_distance_to_scribble(scribble, to_foreground=False)
    gradient = gradient_per_pixel(image)

    # Encode
    H, W, C = image.shape
    num_pixels = H * W
    rgb_flat = image.reshape(num_pixels, C).astype(np.float32)
    scribble_flat = scribble.reshape(num_pixels, 1).astype(np.float32)
    dist_fg_flat = dist_fg.reshape(num_pixels, 1).astype(np.float32)
    dist_bg_flat = dist_bg.reshape(num_pixels, 1).astype(np.float32)
    gradient_flat = gradient.reshape(num_pixels, 1).astype(np.float32)

    feature_vec_2d = np.hstack([rgb_flat, scribble_flat, dist_fg_flat, dist_bg_flat, gradient_flat])
    feature_vec_1d = feature_vec_2d.reshape(1, -1)
    return feature_vec_1d

def encode_gt_picture(gt_image: np.ndarray) -> np.ndarray:
    """
    Brings the ground truth images in the correct form for learning
    """
    encoded_vec_1d = gt_image.reshape(1, -1).astype(np.float32)
    return encoded_vec_1d

def decode_pred_picture(prediction_vec: np.ndarray, h = 375, w = 500) -> np.ndarray:
    return prediction_vec.reshape(h, w)

####################################
# Productive code
####################################

# get data
print("Loading data...")
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset_rgb_gray(
    "../../dataset/training", "images", "scribbles", "ground_truth"
)

images_test, scrib_test, fnames_test = load_dataset_rgb_gray(
    "../../dataset/test", "images", "scribbles"
)

# construct feature and ground truth matrix
print("Construct training data...")
X_train = np.vstack([feature_encode(img, scrib) for img, scrib in zip(images_train, scrib_train)])
Y_train = np.vstack([encode_gt_picture(gt_img) for gt_img in gt_train])
print("Shape of feature matrix: " + str(X_train.shape))

# reduce features significantly using PCA
print("PCA feature reduction...")
n_components = 150 # maintains 95,9 % of variance information with this amount
pca = PCA(n_components=n_components)
X_train_reduced = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_.cumsum()
print("Explained variance by PCA: " + str(explained_variance))
print("Shape of reduced matrix: " + str(X_train_reduced.shape))

# train classifier
print("Starting training or retrieve existing model...")
base_lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    C=1.0,
    verbose=1
)

multi_lr = MultiOutputClassifier(base_lr, n_jobs=-1)

model_path = "multi_lr_model.pkl"

if os.path.exists(model_path):
    print("Loading existing trained model...")
    multi_lr = joblib.load(model_path)
else:
    print("Training model...")
    multi_lr.fit(X_train_reduced, Y_train)
    joblib.dump(multi_lr, model_path)
    print(f"Model saved to {model_path}")

# predictions
num_predictions = 3

print("Make prediction on training data...")
results_train = []
for i in range(num_predictions):
    x_train = feature_encode(images_train[i], scrib_train[i])
    x_train_reduced = pca.transform(x_train)
    pred_train = multi_lr.predict(x_train_reduced)
    pred_train_decoded = decode_pred_picture(pred_train)
    results_train.append(pred_train_decoded)
pred_train = np.stack(results_train, axis=0)

print("Store training predictions...")
store_predictions(
    pred_train, "../../dataset/training/predictions", "first_attempt", fnames_train, palette
)

print("Make prediction on test data...")
results_test = []
for i in range(num_predictions):
    x_test = feature_encode(images_test[i], scrib_test[i])
    x_test_reduced = pca.transform(x_test)
    pred_test = multi_lr.predict(x_test_reduced)
    pred_test_decoded = decode_pred_picture(pred_test)
    results_test.append(pred_test_decoded)

pred_test = np.stack(results_test, axis=0)
print("Store test predictions...")
store_predictions(
    pred_test, "../../dataset/test/predictions", "first_attempt", fnames_test, palette
)



