import torch
from torch.utils.data import DataLoader

from data import train_test_split_dataset, SegmentationDataset, augment_and_save_data, pad_and_save_data
from losses import ProbWeightedBCEDiceLoss, WeightedBCEDiceLoss, estimate_pos_loss_weight, WeightedBCEJaccardLoss
from models import UNet3
from trainer import train_model, predict_and_save
from util import load_dataset

# constants
CREATE_NEW_AUGMENTATIONS = False
SAVE_PREDICTIONS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH_PRED = "dataset/augmentations/validation/predictions"
SOURCE_PATH_TRAIN = "dataset/training"
SOURCE_PATH_AUG_TRAIN = "dataset/augmentations/train"
SOURCE_PATH_AUG_VAL = "dataset/augmentations/validation"

HYPERPARAMS = {
    "regularization": {
        "weight_decay": 5e-7,          # e.g., 1e-6
        "dropout_rate_model": 0.1
    },
    "training": {
        "learning_rate": 2e-3,
        "validation_set_size": 0.15,
        "num_epochs": 60,
        "scheduler_factor": 0.2,
        "scheduler_patience": 10,
        "batch_size": 8
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

# create instances

# model
model = UNet3(dropout_rate=HYPERPARAMS["regularization"]["dropout_rate_model"])

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
#est_pos_weight = estimate_pos_loss_weight(train_loader)
#print("Estimated positive weight for loss function: ", est_pos_weight)
criterion = WeightedBCEJaccardLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=HYPERPARAMS["training"]["learning_rate"],
    weight_decay=HYPERPARAMS["regularization"]["weight_decay"]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=HYPERPARAMS["training"]["scheduler_factor"],
    patience=HYPERPARAMS["training"]["scheduler_patience"]
)

# train model
best_model_path = train_model(
    model, DEVICE, train_loader, val_loader,
    criterion, optimizer, scheduler,
    HYPERPARAMS["training"]["num_epochs"]
)

# make and save predictions
if SAVE_PREDICTIONS:
    predict_and_save(
        model, DEVICE, best_model_path,
        SAVE_PATH_PRED, val_loader, fnames2, palette2
    )
