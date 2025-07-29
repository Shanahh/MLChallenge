import torch
from torch.utils.data import DataLoader

from src.cnn.data import train_test_split_dataset, SegmentationDataset, augment_and_save_data, pad_and_save_data
from src.cnn.losses import WeightedBCEDiceLoss
from src.cnn.lr_finder import LRFinder
from src.cnn.models import UNet4
from src.cnn.trainer import train_model, predict_and_save
from src.baseline.util import load_dataset
from copy import deepcopy

# constants
CREATE_NEW_AUGMENTATIONS = False
SAVE_STATISTICS_AND_MODEL = True
SAVE_PREDICTIONS = False
FIND_LR = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH_PRED = "../../dataset/augmentations/validation/predictions"
SOURCE_PATH_TRAIN = "../../dataset/training"
SOURCE_PATH_AUG_TRAIN = "../../dataset/augmentations/train"
SOURCE_PATH_AUG_VAL = "../../dataset/augmentations/validation"

HYPERPARAMS = {
    "regularization": {
        "weight_decay": 1e-6, # 1e-5,  # e.g., 1e-6
        "dropout_rate_model": 0.05
    },
    "training": {
        "learning_rate": 6e-3,
        "validation_set_size": 0.12, # only relevant if CREATE_NEW_AUGMENTATIONS is true
        "num_epochs": 2,
        "batch_size": 8,
        "loss_pos_weight": 2, # the higher, the more the model will be penalized for predicting too much background
        "apply_sigmoid_in_model": False # leave false unless loss function without sigmoid application
    },
    "scheduler": {
        "one_cycle_scheduler": False,
        "max_lr": 6e-3, # only relevant if one_cycle_scheduler is True
        "scheduler_factor": 0.3, # only relevant if one_cycle_scheduler is False
        "scheduler_patience": 5, # only relevant if one_cycle_scheduler is False
    }
}

if CREATE_NEW_AUGMENTATIONS:
    print("Creating new augmentations...")
    # load original training data
    images, scrib, gt, fnames, palette = load_dataset(
        SOURCE_PATH_TRAIN, "images", "scribbles", "ground_truth"
    )
    # split data
    (train_images, train_scribbles, train_gt), (test_images, test_scribbles, test_gt) \
        = train_test_split_dataset(images, scrib, gt, test_size=HYPERPARAMS["training"]["validation_set_size"])
    # augment
    augment_and_save_data(train_images, train_scribbles, train_gt, palette, SOURCE_PATH_AUG_TRAIN)
    pad_and_save_data(test_images, test_scribbles, test_gt, palette, SOURCE_PATH_AUG_VAL)

# load augmented training data
train_images_aug, train_scrib_aug, train_gt_aug, fnames2, palette2 = load_dataset(
    SOURCE_PATH_AUG_TRAIN, "images", "scribbles", "ground_truth"
)

val_images, val_scrib, val_gt, *_ = load_dataset(
    SOURCE_PATH_AUG_VAL, "images", "scribbles", "ground_truth"
)

######## create instances

# model
model = UNet4(dropout_rate=HYPERPARAMS["regularization"]["dropout_rate_model"], apply_sigmoid=HYPERPARAMS["training"]["apply_sigmoid_in_model"])

# data loaders
train_dataset = SegmentationDataset(train_images_aug, train_scrib_aug, train_gt_aug)
val_dataset = SegmentationDataset(val_images, val_scrib, val_gt)
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

# loss and optimizing
criterion = WeightedBCEDiceLoss(pos_weight=HYPERPARAMS["training"]["loss_pos_weight"])

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
best_model_path = ""
if not FIND_LR:
    best_model_path = train_model(
        model, DEVICE, train_loader, val_loader,
        criterion, optimizer,
        scheduler_one_cycle if HYPERPARAMS["scheduler"]["one_cycle_scheduler"] else scheduler_plateau,
        HYPERPARAMS["training"]["num_epochs"],
        SAVE_STATISTICS_AND_MODEL
    )

# make and save predictions
if SAVE_PREDICTIONS:
    predict_and_save(
        model, DEVICE, best_model_path,
        SAVE_PATH_PRED, val_loader, fnames2, palette2
    )
