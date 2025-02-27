# Run this code like python train.py --config example.yaml
# If you have a checkpoint, run this code like python train.py --config example.yaml --checkpoint /path/to/the/chechpoint.pth

import os
import time
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import UNet
from utils.dataset import load_data
from utils.cyclic_perceptual_loss import (
    CombinedLoss,
    update_plane_selection,
    calculate_cycle_epochs,
)
from utils.utils import (
    update_checkpoint_and_log,
    load_checkpoint,
    parse_arguments,
    load_config,
    seed_everything,
)


def train_one_epoch(
    model, train_loader, optimizer, loss_function, device, selection, grad_accum_steps
):
    """
    Train the model for one epoch on the given DataLoader and return the average loss.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): The DataLoader for the training set.
        optimizer (Optimizer): The optimizer.
        loss_function (callable): The loss function.
        device (torch.device): The device to use (CPU or GPU).
        selection (str): Which plane is currently selected ("axial", "coronal", or "sagittal").
        grad_accum_steps (int): Number of gradient accumulation steps.

    Returns:
        float: The average training loss for this epoch.
    """
    model.train()
    epoch_loss = 0.0

    for idx, batch_data in enumerate(train_loader):
        inputs = batch_data["image"].to(device)
        targets = batch_data["label"].to(device)

        outputs = model(inputs)
        loss = loss_function(outputs, targets, selection)

        # Gradient Accumulation
        (loss / grad_accum_steps).backward()
        if (idx + 1) % grad_accum_steps == 0 or idx == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

        # Clear memory for unused tensors
        del inputs, targets, outputs

    average_loss = epoch_loss / len(train_loader)
    return average_loss


def validate_one_epoch(model, val_loader, loss_function, device, selection):
    """
    Run validation on the given DataLoader for one epoch and return the average validation loss.

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): The DataLoader for the validation set.
        loss_function (callable): The loss function.
        device (torch.device): The device to use (CPU or GPU).
        selection (str): Which plane is currently selected ("axial", "coronal", or "sagittal").

    Returns:
        float: The average validation loss for this epoch.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_targets = val_data["label"].to(device)

            val_outputs = model(val_inputs)
            loss = loss_function(val_outputs, val_targets, selection)
            val_loss += loss.item()

            del val_inputs, val_targets, val_outputs

    average_loss = val_loss / len(val_loader)
    return average_loss


def train_model(train_loader, val_loader, config, device, checkpoint_path=None):
    """
    Train and validate the model on the provided data loaders. Manages checkpoints and logging.

    Args:
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader (DataLoader): The DataLoader for the validation set.
        config (dict): Configuration dictionary loaded from YAML.
        device (torch.device): The device to use for training.
        checkpoint_path (str, optional): Path to an existing checkpoint to resume from.
    """

    # Prepare directories
    log_dir = config["LOG_DIR"]
    model_dir = config["MODEL_DIR"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # File for logging
    log_file = os.path.join(log_dir, "log.txt")

    # Load settings
    max_epochs = config["MAX_EPOCHS"]
    learning_rate = float(config["LEARNING_RATE"])
    use_dropout = config["USE_DROPOUT"]
    dropout = config["DROPOUT"]
    early_stopping_patience = config["EARLY_STOPPING_PATIENCE"]
    channels_setting = tuple(config["CHANNELS_SETTING"])
    cycle_duration = config["CYCLE_DURATION"]
    cycle_factor = config["CYCLE_FACTOR"]
    grad_accum_steps = config["GRAD_ACCUM_STEPS"]

    # Model creation
    if use_dropout:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels_setting,
            strides=(2, 2, 2, 2),
            dropout=dropout,
        ).to(device)
    else:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels_setting,
            strides=(2, 2, 2, 2),
        ).to(device)

    loss_function = CombinedLoss(config=config)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=int((cycle_duration + 0.5) / 2), eta_min=0
    )

    # Calculate the plane-switching epochs (Algorithm 2 in the paper)
    ax_epochs, co_epochs, sa_epochs = calculate_cycle_epochs(
        max_epochs, cycle_duration, cycle_factor
    )
    print("ax: ", ax_epochs)
    print("co: ", co_epochs)
    print("sa: ", sa_epochs)
    print("Max Epoch: ", max_epochs)

    # Initialize checkpoint-related variables
    best_val_loss = float("inf")
    no_improvement_epochs = 0
    selection = "axial"  # Default plane
    start_epoch = 0
    early_stopping_trigger_epoch = (
        ax_epochs[2] if len(ax_epochs) > 2 else 0
    )  # Early stopping is activated in the late stages to prevent it from activating when the model is overfitted into a specific plane
    print(
        f"Early stopping will start to check after epoch: {early_stopping_trigger_epoch}"
    )
    best_checkpoint_path = None

    # Load checkpoint if provided
    if checkpoint_path:
        start_epoch, best_val_loss, selection, no_improvement_epochs = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
    else:
        print("No checkpoint provided.")

    log_message = ""
    epoch_time_start = time.time()

    # Main training loop
    for epoch in range(start_epoch, max_epochs):
        # Print elapsed time every 50 epochs
        if epoch > 0 and epoch % 50 == 0:
            epoch_time_end = time.time()
            elapsed = epoch_time_end - epoch_time_start
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            print(f"{epoch} epochs done: {hours}h {minutes}m {seconds}s elapsed.")
            epoch_time_start = time.time()

        # Check if we need to switch planes
        new_selection, reset_best = update_plane_selection(
            epoch, ax_epochs, co_epochs, sa_epochs, selection
        )

        if new_selection != selection:
            selection = new_selection
            if selection == "axial":
                print("Perceptual Loss is now based on axial slices.")
            elif selection == "coronal":
                print("Perceptual Loss is now based on coronal slices.")
            elif selection == "sagittal":
                print("Perceptual Loss is now based on sagittal slices.")

        if reset_best:
            best_val_loss = float("inf")

        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            selection=selection,
            grad_accum_steps=grad_accum_steps,
        )

        # Validate
        val_loss = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_function=loss_function,
            device=device,
            selection=selection,
        )

        print(
            f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )
        log_message += (
            f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}\n"
        )

        scheduler.step()

        # Update checkpoint and log
        best_val_loss, no_improvement_epochs, stop_training, best_checkpoint_path = (
            update_checkpoint_and_log(
                average_val_loss=val_loss,
                selection=selection,
                epoch=epoch,
                no_improvement_epochs=no_improvement_epochs,
                log_message=log_message,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                model_dir=model_dir,
                early_stopping_patience=early_stopping_patience,
                log_file=log_file,
                early_stopping_trigger_epoch=early_stopping_trigger_epoch,
                best_checkpoint_path=best_checkpoint_path,
            )
        )

        if stop_training:
            print(f"Early stopping at epoch {epoch+1}")
            break


def main():
    seed_everything(deterministic=True)  # Fix seed
    args = parse_arguments()
    config = load_config(args.config)
    if isinstance(config["CUDA_SETTING"], int):
        torch.cuda.set_device(config["CUDA_SETTING"])
    device = torch.device(
        config["CUDA_SETTING"] if torch.cuda.is_available() else "cpu"
    )
    train_loader, val_loader = load_data(config)
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
