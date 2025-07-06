import os
import torch
from data import save_training_plots, save_model, remove_padding_gt
from evaluation import *
from model import UNet3
from util import store_predictions

# constants
TRAIN_LOCATION_GPU = "cuda"
TRAIN_LOCATION_CPU = "cpu"
MODEL_DIR_PATH = "models"
PLOTS_DIR_PATH = "training_plots"
NUM_BARS = 30

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """
    Trains the given model and saves the best model and all statistics collected on the validation set, such as training loss,
    validation loss, and IoU scores.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device(TRAIN_LOCATION_GPU if use_cuda else TRAIN_LOCATION_CPU)
    model = model.to(device)

    best_val_loss = float('inf')
    best_model_name = ""
    train_losses = []
    val_losses = []
    val_obj_ious = []
    val_bkg_ious = []
    val_mean_ious = []

    print("Starting training...")
    print("using cuda cores: " + str(use_cuda))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * NUM_BARS)
        # training
        epoch_train_loss = training_phase(model, train_loader, criterion, optimizer, device)
        train_losses.append(epoch_train_loss)

        # validation
        epoch_val_loss, epoch_val_obj_iou, epoch_val_bkg_iou, epoch_val_mean_iou = validation_phase(model, val_loader, criterion, device)
        val_losses.append(epoch_val_loss)
        val_obj_ious.append(epoch_val_obj_iou)
        val_bkg_ious.append(epoch_val_bkg_iou)
        val_mean_ious.append(epoch_val_mean_iou)

        # outputs
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Obj IoU: {epoch_val_obj_iou:.4f} | Bkg IoU: {epoch_val_bkg_iou:.4f} | Mean IoU: {epoch_val_mean_iou:.4f}")

        # save model if it achieves the current best results on the validation set in terms of validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_name = save_model(model, MODEL_DIR_PATH)

        # optimize learning rate if we have a scheduler
        if scheduler:
            scheduler.step(epoch_val_loss)

    print("Training finished - saving results...")
    save_training_plots(PLOTS_DIR_PATH, train_losses, val_losses, val_obj_ious, val_bkg_ious, val_mean_ious)

    return best_model_name

def training_phase(model, train_loader, criterion, optimizer, device):
    """
    Does one epoch of training and returns the training loss of this epoch
    """
    model.train()
    train_loss_sum = 0.0
    for inputs, masks in train_loader:
        # move inputs and masks to gpu or cpu accordingly
        inputs = inputs.to(device)
        masks = masks.to(device)
        # training step on the model with as many data points as defined in batch size of train_loader
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        # keep track of training loss
        batch_size = inputs.size(0)
        train_loss_sum += loss.item() * batch_size # loss.item() averages over the entire batch
    epoch_train_loss = train_loss_sum / len(train_loader.dataset)
    return epoch_train_loss

def validation_phase(model, val_loader, criterion, device):
    """
    Executes validation of the current epoch
    Returns the mean of each: validation_loss, object IoU, background IoU, mean IoU
    """
    model.eval()
    val_loss_sum = 0.0
    val_obj_iou = 0.0
    val_bkg_iou = 0.0
    val_mean_iou = 0.0
    with torch.no_grad():
        for inputs, masks in val_loader:
            # move inputs and masks to gpu or cpu accordingly
            inputs = inputs.to(device)
            masks = masks.to(device)
            # get model predictions and loss
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            # compute validation loss value
            batch_size = inputs.size(0)
            val_loss_sum += loss.item() * batch_size # mean value needs to be multiplied with batch size again
            # compute different IoU scores
            outputs_np = model_output_to_mask(outputs)
            masks_np = model_output_to_mask(masks)
            # revert padding to not get incorrect background IoU scores
            outputs_np_no_pad = remove_padding_gt(outputs_np)
            masks_np_no_pad = remove_padding_gt(masks_np)
            obj_io_batch, bkg_iou_batch, mean_iou_batch = get_ious(masks_np_no_pad, outputs_np_no_pad)
            # mean value needs to be multiplied with batch size again
            val_obj_iou += obj_io_batch * batch_size
            val_bkg_iou += bkg_iou_batch * batch_size
            val_mean_iou += mean_iou_batch * batch_size
    # compute statistics
    epoch_val_loss = val_loss_sum / len(val_loader.dataset)
    epoch_obj_iou = val_obj_iou / len(val_loader.dataset)
    epoch_bkg_iou = val_bkg_iou / len(val_loader.dataset)
    epoch_mean_iou = val_mean_iou / len(val_loader.dataset)
    return epoch_val_loss, epoch_obj_iou, epoch_bkg_iou, epoch_mean_iou


def predict_and_save(model, model_path, save_dir_path, save_dir, data_loader, fnames_test, palette):
    """
    Makes predictions for the data in the data loader and stores them in a folder in the given path.
    """
    device = TRAIN_LOCATION_GPU if torch.cuda.is_available() else TRAIN_LOCATION_CPU
    os.makedirs(save_dir, exist_ok=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}, running and saving prediction on {len(data_loader.dataset)} samples...")

    with torch.no_grad():
        prediction_list = []
        for inputs, _ in data_loader:
            inputs = inputs.to(device)  # shape: [B, 4, H, W]
            outputs = model(inputs)  # shape: [B, 1, H, W], model already outputs probabilities
            predictions = (outputs > 0.5).float()  # binarize

            # Save predictions (each sample)
            for i in range(predictions.size(0)):
                pred_mask = predictions[i, 0]  # shape: [H, W]
                mask_np = (pred_mask.cpu().numpy() * 255).astype(np.uint8)
                prediction_list.append(mask_np)

        pred_np = np.stack(prediction_list, axis=0)

        store_predictions(
            pred_np, save_dir_path, save_dir, fnames_test, palette
        )

    print("Predictions saved.")

