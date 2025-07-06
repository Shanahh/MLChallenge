import torch
from torch import nn
from torch.utils.data import DataLoader

from data import train_test_split_dataset, SegmentationDataset
from losses import WeightedSoftIoULoss, WeightedBCEDiceLoss
from trainer import train_model, predict_and_save
from util import load_dataset
from model import UNet3

# constants
SAVE_DIR_PATH = "dataset/augmentations"
SAVE_DIR = "predictions"

# HYPERPARAMETERS:
#-----------
# model:
# number of filters / blocks in Unet -> Adapt in model.py
LEARNING_RATE = 1e-3

# training:
DATA_PATH = "dataset/augmentations" # different augmentation versions
VALIDATION_SET_SIZE = 0.1
NUM_EPOCHS = 10
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 5 # compare with EPOCHS
BATCH_SIZE = 8
# loss type
#-----------

# load data
images_train, scrib_train, gt_train, fnames, palette = load_dataset(
    DATA_PATH, "images", "scribbles", "ground_truth"
)

# split data
(train_images, train_scribbles, train_gt), (test_images, test_scribbles, test_gt)\
= train_test_split_dataset(images_train, scrib_train, gt_train, VALIDATION_SET_SIZE)

# create instances

# model
model = UNet3()

# data loaders
train_dataset = SegmentationDataset(train_images, train_scribbles, train_gt)
val_dataset = SegmentationDataset(test_images, test_scribbles, test_gt)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loss and optimizing
criterion = WeightedBCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

# train model
best_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS)

# Make and save predictions
predict_and_save(model, best_model, SAVE_DIR_PATH, SAVE_DIR, val_loader, fnames, palette)