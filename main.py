import torch
from torch.utils.data import DataLoader

from data import train_test_split_dataset, SegmentationDataset, augment_and_save_data, pad_and_save_data
from losses import WeightedBCEDiceLoss
from trainer import train_model, predict_and_save
from util import load_dataset
from model import UNet3

# constants
CREATE_NEW_AUGMENTATIONS = True
SAVE_PREDICTIONS = True
SAVE_PATH_PRED = "dataset/augmentations/validation/predictions"
SOURCE_PATH_TRAIN = "dataset/training"
SOURCE_PATH_AUG_TRAIN = "dataset/augmentations/train"
SOURCE_PATH_AUG_VAL = "dataset/augmentations/validation"

# HYPERPARAMETERS:
#--------------
# model:
# number of filters / blocks in Unet -> Adapt in model.py
LEARNING_RATE = 1e-3

# training:
VALIDATION_SET_SIZE = 0.1
NUM_EPOCHS = 50
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 10 # compare with EPOCHS
BATCH_SIZE = 16
# loss type
#--------------

if CREATE_NEW_AUGMENTATIONS:
    print("Creating new augmentations...")
    # load original training data
    images, scrib, gt, fnames, palette = load_dataset(
        SOURCE_PATH_TRAIN, "images", "scribbles", "ground_truth"
    )
    # split data
    (train_images, train_scribbles, train_gt), (test_images, test_scribbles, test_gt) \
        = train_test_split_dataset(images, scrib, gt, VALIDATION_SET_SIZE)
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

# CREATE INSTANCES
#--------------
# model
model = UNet3()

# data loaders
train_dataset = SegmentationDataset(train_images_aug, train_scrib_aug, train_gt_aug)
val_dataset = SegmentationDataset(val_images, val_scrib, val_gt)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loss and optimizing
criterion = WeightedBCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
#--------------

# train model
best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS)
#best_model = "models/model_2025-07-06_17-37-02.pth"

# Make and save predictions
if SAVE_PREDICTIONS:
    predict_and_save(model, best_model_path, SAVE_PATH_PRED, val_loader, fnames2, palette2)