# Load important packages
import numpy as np

from util import load_dataset
from util import store_predictions
from util import segment_with_knn
from util import visualize

######### Training dataset

test_mode = True
image_number = 0
print("Test mode active: " + str(test_mode))

# Load training dataset
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/training", "images", "scribbles", "ground_truth"
)

# Inference
# Create a numpy array of size num_train x 375 x 500, a stack of all the
# segmented images. 1 = foreground, 0 = background.
print("Started predicting...")
if not test_mode:
    pred_train = np.stack(
        [segment_with_knn(image, scribble, k=3)
         for image, scribble in zip(images_train, scrib_train)],
        axis=0
    )
else:
    img = images_train[image_number]
    scrib = scrib_train[image_number]
    pred_train = np.stack(
        [segment_with_knn(img, scrib, k=3)],
        axis=0
    )

print("Stopped predicting...")

# Storing Predictions
store_predictions(
    pred_train, "dataset/training", "predictions", fnames_train, palette
)

# Visualizing model performance
if not test_mode:
    vis_index = np.random.randint(images_train.shape[0])
else:
    vis_index = image_number
visualize(
    images_train[vis_index], scrib_train[vis_index],
    gt_train[vis_index], pred_train[vis_index]
)


######### Test dataset

# Load test dataset
images_test, scrib_test, fnames_test = load_dataset(
    "dataset/test", "images", "scribbles"
)

# Inference
# Create a numpy array of size num_test x 375 x 500, a stack of all the 
# segmented images. 1 = foreground, 0 = background.
if not test_mode:
    pred_test = np.stack(
        [segment_with_knn(image, scribble, k=3)
         for image, scribble in zip(images_test, scrib_test)],
        axis=0
    )
else:
    img = images_test[image_number]
    scrib = scrib_test[image_number]
    pred_test = np.stack(
        [segment_with_knn(img, scrib, k=3)],
        axis=0
    )

print("Storing predictions...")
# Storing segmented images for test dataset.
store_predictions(
    pred_test, "dataset/test", "predictions", fnames_test, palette
)


