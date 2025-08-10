# Load important packages
import numpy as np

from util import load_dataset_rgb_gray
from util import store_predictions
from util import safe_segment_with_knn
from util import visualize

######### Just for testing purposes

test_mode = False
predict_on_train = False
predict_on_test = True
k_val = 53
image_number = "2010_000685"
print("Test mode active: " + str(test_mode))

######### Training dataset

# Load training dataset
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset_rgb_gray(
    "../../dataset/training", "images", "scribbles", ground_truth_dir="ground_truth"
)

if predict_on_train:
    # Inference
    # Create a numpy array of size num_train x 375 x 500, a stack of all the
    # segmented images. 1 = foreground, 0 = background.
    print("Start prediction train data...")
    if not test_mode:
        pred_train = np.stack(
            [safe_segment_with_knn(image, scribble, k=k_val)
             for image, scribble in zip(images_train, scrib_train)],
            axis=0
        )
    else: # test mode
        try:
            img_idx = fnames_train.index(image_number + ".png")
        except ValueError:
            raise ValueError(f"Image name {image_number}.png not found in training filenames.")

        img = images_train[img_idx]
        scrib = scrib_train[img_idx]
        ###
        print("Scribble annotations:")
        unique, counts = np.unique(scrib[scrib != 255], return_counts=True)
        print(dict(zip(unique, counts)))
        ###
        pred_train = np.stack(
            [safe_segment_with_knn(img, scrib, k=k_val)],
            axis=0
        )
        original_fname = fnames_train[img_idx]
        basename = original_fname.replace(".jpg", "").replace(".png", "")
        fnames_train = [f"{basename}_k_{k_val}.png"]

    print("Store prediction train data...")

    # Storing Predictions
    store_predictions(
        pred_train, "../../dataset/experiments", "images", fnames_train, palette
    )

    # Visualizing model performance
    if not test_mode:
        vis_index = np.random.randint(images_train.shape[0])
        visualize(
            images_train[vis_index], scrib_train[vis_index],
            gt_train[vis_index], pred_train[vis_index]
        )

######### Test dataset
if predict_on_test:
    # Load test dataset
    images_test, scrib_test, fnames_test = load_dataset_rgb_gray(
        "../../dataset/test", "images", "scribbles"
    )

    # Inference
    # Create a numpy array of size num_test x 375 x 500, a stack of all the
    # segmented images. 1 = foreground, 0 = background.
    print("Start prediction test data...")
    if not test_mode:
        pred_test = np.stack(
            [safe_segment_with_knn(image, scribble, k=k_val)
             for image, scribble in zip(images_test, scrib_test)],
            axis=0
        )
    else: # test mode
        try:
            img_idx = fnames_test.index(image_number + ".png")
        except ValueError:
            raise ValueError(f"Image name {image_number}.png not found in test filenames.")

        img = images_test[img_idx]
        scrib = scrib_test[img_idx]
        ###
        print("Scribble annotations:")
        unique, counts = np.unique(scrib[scrib != 255], return_counts=True)
        print(dict(zip(unique, counts)))
        ###
        pred_test = np.stack(
            [safe_segment_with_knn(img, scrib, k=k_val)],
            axis=0
        )
        original_fname = fnames_test[img_idx]
        basename = original_fname.replace(".jpg", "").replace(".png", "")
        fnames_test = [f"{basename}_k_{k_val}.png"]

    print("Store prediction test data...")

    # Storing segmented images for test dataset.
    store_predictions(
        pred_test, "../../dataset/test_knn_k_53", "masks", fnames_test, palette
    )


