from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from src.knn_baseline.util import load_dataset_gray_twice, load_dataset_gray_once
from src.cnn.data import train_test_split_dataset, augment_and_save_data, pad_and_save_data, \
    pad_and_return_data, SegmentationDatasetExt
from src.cnn.losses import WeightedBCEDiceLoss
from src.cnn.lr_finder import LRFinder
from src.cnn.models import UNet4
from src.cnn.trainer import train_model, predict_and_save

# steering cockpit
CREATE_NEW_AUGMENTATIONS = True
FIND_LR = False
DO_TRAIN = True
SAVE_STATISTICS_AND_MODEL = True
MAKE_PREDICTIONS_ON_VAL = False
MAKE_PREDICTIONS_ON_TEST = False

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_PATH_PRED_VAL = "../../dataset/augmentations/validation/predictions"
TARGET_PATH_PRED_TEST = "../../dataset/test_knn_k_53/predictions"
SOURCE_PATH_TRAIN_MASKS_GC = "../../dataset/training_gc"
SOURCE_PATH_TRAIN_MASKS_KNN = "../../dataset/training_knn_k_3"
SOURCE_PATH_AUG_TRAIN = "../../dataset/augmentations/train"
SOURCE_PATH_AUG_VAL = "../../dataset/augmentations/validation"
SOURCE_PATH_TEST = "../../dataset/test_knn_k_53"

HYPERPARAMS = {
    "regularization": {
        "weight_decay": 5e-7, # 1e-6
        "dropout_rate_model": 0.05
    },
    "training": {
        "learning_rate": 2.5e-2,
        "validation_set_size": 0.12, # only relevant if CREATE_NEW_AUGMENTATIONS is true
        "num_epochs": 50,
        "batch_size": 8,
        "loss_pos_weight": 2.0, # the higher, the more the model will be penalized for predicting too much background
        "loss_iou_weight": 1.0,
        "apply_sigmoid_in_model": False # leave False unless loss function without sigmoid application
    },
    "scheduler": {
        "one_cycle_scheduler": False,
        "max_lr": 2.5e-2, # only relevant if one_cycle_scheduler is True
        "scheduler_factor": 0.3, # only relevant if one_cycle_scheduler is False
        "scheduler_patience": 4, # only relevant if one_cycle_scheduler is False
    }
}

if CREATE_NEW_AUGMENTATIONS:
    print("Creating new augmentations...")
    # load original training data
    masks_gc, scrib, gt, fnames, palette = load_dataset_gray_once(
        SOURCE_PATH_TRAIN_MASKS_GC, "images", "scribbles", ground_truth_dir="ground_truth"
    )
    masks_knn, *_ = load_dataset_gray_once(
        SOURCE_PATH_TRAIN_MASKS_KNN, "images", "scribbles", ground_truth_dir="ground_truth"
    )
    # split data
    (train_images_gc, train_images_knn, train_scribbles, train_gt), (val_images_gc, val_images_knn, val_scribbles, val_gt) \
        = train_test_split_dataset(masks_gc, masks_knn, scrib, gt, validation_size=HYPERPARAMS["training"]["validation_set_size"])
    # augment
    augment_and_save_data(train_images_gc, train_images_knn, train_scribbles, train_gt, palette, SOURCE_PATH_AUG_TRAIN)
    pad_and_save_data(val_images_gc, val_images_knn, val_scribbles, val_gt, palette, SOURCE_PATH_AUG_VAL)

# load augmented training data
train_images_aug_gc, train_images_aug_knn, train_scrib_aug, train_gt_aug, train_fnames, palette2 = load_dataset_gray_twice(
    SOURCE_PATH_AUG_TRAIN, "images", "scribbles", "masks", "ground_truth"
)

# load validation data
val_images_gc, val_images_knn, val_scrib, val_gt, val_fnames, *_ = load_dataset_gray_twice(
    SOURCE_PATH_AUG_VAL, "images", "scribbles", "masks", "ground_truth"
)

# load test data and pad it
test_images_rgb, test_images_gray, test_scrib, test_fnames = load_dataset_gray_twice(
    SOURCE_PATH_TEST, "images", "scribbles", "masks"
)
dummy_test_gt = deepcopy(test_images_gray)
test_img_rgb_pad, test_img_gray_pad, test_scrib_pad, dummy_test_gt_pad = pad_and_return_data(test_images_rgb, test_images_gray, test_scrib, dummy_test_gt)

######## create instances

# model
model = UNet4(
    dropout_rate=HYPERPARAMS["regularization"]["dropout_rate_model"],
    apply_sigmoid=HYPERPARAMS["training"]["apply_sigmoid_in_model"]
)

# data loaders
train_dataset = SegmentationDatasetExt(train_images_aug_gc, train_images_aug_knn, train_scrib_aug, train_gt_aug)
val_dataset = SegmentationDatasetExt(val_images_gc, val_images_knn, val_scrib, val_gt)
test_dataset = SegmentationDatasetExt(test_img_rgb_pad, test_img_gray_pad, test_scrib_pad, dummy_test_gt_pad) # dummy s.t. data loader works in prediction

train_loader = DataLoader(
    train_dataset,
    batch_size=HYPERPARAMS["training"]["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=HYPERPARAMS["training"]["batch_size"],
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=HYPERPARAMS["training"]["batch_size"],
    shuffle=False
)

# loss and optimizing
criterion = WeightedBCEDiceLoss(
    pos_weight=HYPERPARAMS["training"]["loss_pos_weight"],
    dice_weight=HYPERPARAMS["training"]["loss_iou_weight"]
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=HYPERPARAMS["training"]["learning_rate"],
    weight_decay=HYPERPARAMS["regularization"]["weight_decay"]
)

# different schedulers
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=HYPERPARAMS["scheduler"]["scheduler_factor"],
    patience=HYPERPARAMS["scheduler"]["scheduler_patience"]
)

batches_per_epoch = len(train_loader)
scheduler_one_cycle = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=HYPERPARAMS["scheduler"]["max_lr"],
    steps_per_epoch=batches_per_epoch,
    epochs=HYPERPARAMS["training"]["num_epochs"]
)

# lr finder
if FIND_LR:
    print("Finding optimal learning rate...")
    model_copy = deepcopy(model)
    optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=1e-7)  # Start with a very small LR for the test

    lr_finder = LRFinder(model_copy, optimizer_copy, criterion, DEVICE)

    log_lrs, losses = lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1)
    lr_finder.plot()
    print("LR finder done. Inspect plot and set lr / max_lr accordingly before training.")

# train model
best_model_path = "../../models\model_2025-08-09_15-46-13.pth" # 78 avg. mIoU on validation set
if DO_TRAIN:
    best_model_path = train_model(
        model, DEVICE, train_loader, val_loader,
        criterion, optimizer,
        scheduler_one_cycle if HYPERPARAMS["scheduler"]["one_cycle_scheduler"] else scheduler_plateau,
        HYPERPARAMS["training"]["num_epochs"],
        SAVE_STATISTICS_AND_MODEL
    )

# make and save predictions on validation set
if MAKE_PREDICTIONS_ON_VAL:
    print("Making predictions on validation set...")
    predict_and_save(
        model, DEVICE, best_model_path,
        TARGET_PATH_PRED_VAL, val_loader, val_fnames, palette2,
        remove_padding=False
    )

if MAKE_PREDICTIONS_ON_TEST:
    print("Making predictions on test set...")
    predict_and_save(
        model, DEVICE, best_model_path,
        TARGET_PATH_PRED_TEST, test_loader, test_fnames, palette2,
        remove_padding=True
    )