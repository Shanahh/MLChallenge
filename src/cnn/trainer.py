import copy
import os
from datetime import datetime

import matplotlib.pyplot as plt

from src.cnn.data import remove_padding_gt
from src.cnn.evaluation import *
from src.knn_baseline.util import store_predictions

# constants
MODEL_DIR_PATH = "../../models"
PLOTS_DIR_PATH = "../../training_plots_and_data"
NUM_BARS = 30

def train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_statistics_and_model):
    """
    Trains the given model and saves the best model and all statistics collected on the validation set, such as training loss,
    validation loss, and IoU scores.
    """
    # model and training information
    use_cuda = torch.cuda.is_available()
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    # scheduler information
    scheduler_is_one_cycle = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    # metrics
    best_epoch_val_mean_iou = float('inf')
    best_model = model
    train_losses = []
    val_losses = []
    val_obj_ious = []
    val_bkg_ious = []
    val_mean_ious = []

    print("Starting training...")
    print("using cuda cores: " + str(use_cuda))

    for epoch in range(num_epochs):
        print("-" * NUM_BARS)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # training
        epoch_train_loss = _training_phase(model, device, train_loader, criterion, optimizer, scheduler if scheduler_is_one_cycle else None)
        train_losses.append(epoch_train_loss)

        # validation
        epoch_val_loss, epoch_val_obj_iou, epoch_val_bkg_iou, epoch_val_mean_iou = _validation_phase(model, device, val_loader, criterion)
        val_losses.append(epoch_val_loss)
        val_obj_ious.append(epoch_val_obj_iou)
        val_bkg_ious.append(epoch_val_bkg_iou)
        val_mean_ious.append(epoch_val_mean_iou)

        # outputs
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Obj IoU: {epoch_val_obj_iou:.4f} | Bkg IoU: {epoch_val_bkg_iou:.4f} | Mean IoU: {epoch_val_mean_iou:.4f}")

        # save model if it achieves the current best results on the validation set in terms of mean IoU
        if epoch_val_mean_iou > best_epoch_val_mean_iou:
            best_epoch_val_mean_iou = epoch_val_mean_iou
            best_model = copy.deepcopy(model)

        # optimize learning rate if we have a scheduler which is not a one cycle scheduler
        if scheduler and not scheduler_is_one_cycle:
            scheduler.step(epoch_val_loss)

    # save results
    if save_statistics_and_model:
        best_model_path = _save_training_results(best_model, train_losses, val_losses, val_obj_ious, val_bkg_ious, val_mean_ious)
        return best_model_path

def _training_phase(model, device, train_loader, criterion, optimizer, one_cycle_scheduler):
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
        # scheduler step if applicable
        if one_cycle_scheduler:
            one_cycle_scheduler.step()
        # keep track of training loss
        batch_size = inputs.size(0)
        train_loss_sum += loss.item() * batch_size # loss.item() averages over the entire batch
    epoch_train_loss = train_loss_sum / len(train_loader.dataset)
    return epoch_train_loss

def _validation_phase(model, device, val_loader, criterion):
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
            inputs = inputs.to(device) # (B, 4, H, W)
            masks = masks.to(device) # (B, 1, H, W)
            # get model predictions and loss
            outputs = model(inputs) # (B, 1, H, W)
            loss = criterion(outputs, masks)
            # compute validation loss value
            batch_size = inputs.size(0)
            val_loss_sum += loss.item() * batch_size # mean value needs to be multiplied with batch size again
            # translate model output to np mask
            outputs_np = model_output_to_mask(outputs, apply_sigmoid=True) # (B, H, W)
            masks_np = model_output_to_mask(masks, apply_sigmoid=False) # (B, H, W)
            # revert padding to get correct IoU scores
            outputs_np_no_pad = remove_padding_gt(outputs_np) # (B, H, W)
            masks_np_no_pad = remove_padding_gt(masks_np) # (B, H, W)
            # compute different IoU scores
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

def _save_training_results(best_model, train_losses, val_losses, val_obj_ious, val_bkg_ious, val_mean_ious):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _save_training_plots(PLOTS_DIR_PATH, train_losses, val_losses, val_obj_ious, val_bkg_ious, val_mean_ious, timestamp)
    best_model_path = _store_model(MODEL_DIR_PATH, best_model, timestamp)
    return best_model_path

def _save_training_plots(save_dir_path, train_losses, val_losses, obj_ious, bkg_ious, mean_ious, timestamp):
    """
    Saves plots during model training for different statistics:
    training losses,
    validation losses,
    object ious,
    background ious,
    mean ious
    """
    os.makedirs(save_dir_path, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    full_path_loss_plot = os.path.join(save_dir_path, f"loss_plot_{timestamp}.pdf")
    plt.savefig(full_path_loss_plot)
    plt.close()

    print(f"Loss plot saved to {full_path_loss_plot}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, obj_ious, 'g-', label='Object IoU')
    plt.plot(epochs, bkg_ious, 'c-', label='Background IoU')
    plt.plot(epochs, mean_ious, 'm-', label='Mean IoU')
    plt.title('IoU scores over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    full_path_iou_plot = os.path.join(save_dir_path, f"iou_plot_{timestamp}.pdf")
    plt.savefig(full_path_iou_plot)
    plt.close()

    print(f"IoU plot saved to {full_path_iou_plot}")

def _store_model(save_dir_path, model, timestamp):
    """
    Saves the given model in the specified directory and returns the full model path.
    """
    os.makedirs(save_dir_path, exist_ok=True)

    # Construct filename with timestamp and extension
    filename = f"model_{timestamp}.pth"

    # Full path for saving
    full_path = os.path.join(save_dir_path, filename)

    # Save model state dict
    torch.save(model.state_dict(), full_path)

    print(f"Model saved to {full_path}")

    return full_path

def predict_and_save(model, device, model_path, save_dir_path, data_loader, fnames, palette, remove_padding, postprocess):
    """
    Makes predictions for the data in the data loader and stores them in a folder in the given path.
    """
    os.makedirs(save_dir_path, exist_ok=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}, running and saving prediction on {len(data_loader.dataset)} samples...")

    with torch.no_grad():
        prediction_list = []
        for inputs, _ in data_loader:
            inputs = inputs.to(device)  # shape: [B, 5, H, W]
            outputs = model(inputs)  # shape: [B, 1, H, W]
            predictions = output_to_mask_postprocess(outputs, inputs) if postprocess else model_output_to_mask(outputs)  # shape: [B, H, W]
            if remove_padding:
                predictions = remove_padding_gt(predictions)
            # Save predictions (each sample)
            for i in range(predictions.shape[0]):
                pred_mask = predictions[i]  # shape: [H, W]
                mask_np = pred_mask.astype(np.uint8)
                prediction_list.append(mask_np)

        pred_np = np.stack(prediction_list, axis=0)

        save_dir_path_hd, save_dir = os.path.split(save_dir_path)
        store_predictions(
            pred_np, save_dir_path_hd, save_dir, fnames, palette
        )

    print("Predictions saved")