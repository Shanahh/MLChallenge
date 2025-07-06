import torch
from torch import nn
from torch.utils.data import DataLoader

from data import train_test_split_dataset, SegmentationDataset
from trainer import train_model
from util import load_dataset
from model import UNet

# HYPERPARAMETERS:
#-----------
# model:
# number of filters / blocks in Unet -> Adapt in model.py
# loss type
LEARNING_RATE = 1e-3

# training:
DATA_PATH = "dataset/training" # different augmentation versions
VALIDATION_SET_SIZE = 0.15
NUM_EPOCHS = 10
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 3 # compare with EPOCHS
BATCH_SIZE = 8
#-----------

# load data
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    DATA_PATH, "images", "scribbles", "ground_truth"
)

# split data
(train_images, train_scribbles, train_gt), (test_images, test_scribbles, test_gt)\
= train_test_split_dataset(images_train, scrib_train, gt_train, VALIDATION_SET_SIZE)

# create instances

# model
model = UNet()

# data loaders
train_dataset = SegmentationDataset(train_images, train_scribbles, train_gt)
val_dataset = SegmentationDataset(test_images, test_scribbles, test_gt)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loss and optimizing
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

# train model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS)