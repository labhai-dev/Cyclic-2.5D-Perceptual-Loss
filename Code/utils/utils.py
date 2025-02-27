import os
import random
import yaml
import argparse
import torch
import numpy as np


def update_checkpoint_and_log(
    average_val_loss,
    selection,
    epoch,
    no_improvement_epochs,
    log_message,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    model_dir,
    early_stopping_patience,
    log_file,
    early_stopping_trigger_epoch,
    best_checkpoint_path=None,
):
    """
    Save checkpoints and log results after each epoch. If a new best validation
    loss is found, update the best checkpoint accordingly. Also check for early
    stopping conditions.

    Args:
        average_val_loss (float): Average validation loss of the current epoch.
        selection (str): Current plane selection.
        epoch (int): Current epoch index.
        no_improvement_epochs (int): Number of consecutive epochs without improvement.
        log_message (str): Message to log.
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_val_loss (float): The best (lowest) validation loss so far.
        model_dir (str): Directory to save models and checkpoints.
        early_stopping_patience (int): Patience value for early stopping.
        log_file (str): File path for saving logs.
        early_stopping_trigger_epoch (int): Epoch after which early stopping can be triggered.
        best_checkpoint_path (str, optional): Path to the previous best checkpoint.

    Returns:
        tuple:
            - best_val_loss (float): Updated best validation loss.
            - no_improvement_epochs (int): Updated number of consecutive epochs without improvement.
            - stopped (bool): Whether early stopping was triggered.
            - best_checkpoint_path (str or None): Updated path to the best checkpoint.
    """
    # 1) Always save the current epoch's model as 'final_model.pth'
    final_model_path = os.path.join(model_dir, "final_model.pth")
    final_checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "no_improvement_epochs": no_improvement_epochs,
        "selection": selection,
    }
    torch.save(final_checkpoint, final_model_path)

    # 2) Check if there's a new best model
    if average_val_loss < best_val_loss:
        # Remove old best checkpoint if it exists
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            os.remove(best_checkpoint_path)

        # Reset no_improvement_epochs
        no_improvement_epochs = 0

        # Update best_val_loss
        best_val_loss = average_val_loss

        # Save new best checkpoint
        new_best_checkpoint_filename = f"best_checkpoint_epoch_{epoch+1}.pth"
        new_best_checkpoint_path = os.path.join(model_dir, new_best_checkpoint_filename)

        best_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "no_improvement_epochs": no_improvement_epochs,
            "selection": selection,
        }
        torch.save(best_checkpoint, new_best_checkpoint_path)

        best_checkpoint_path = new_best_checkpoint_path
    else:
        # If it's not a new best, and the trigger epoch is passed, increment no_improvement_epochs
        if epoch > early_stopping_trigger_epoch:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stopping_patience:
                print("Early stopping due to no improvement.")
                with open(log_file, "w") as f:
                    f.write(
                        f"Training completed with best validation loss: {best_val_loss}\n\n{log_message}"
                    )
                return best_val_loss, no_improvement_epochs, True, best_checkpoint_path

    # Write log
    with open(log_file, "w") as f:
        f.write(
            f"Training completed with best validation loss: {best_val_loss}\n\n{log_message}"
        )

    return best_val_loss, no_improvement_epochs, False, best_checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """
    Load the model, optimizer, and scheduler states from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler instance.

    Returns:
        tuple:
            - start_epoch (int): Epoch number from the loaded checkpoint.
            - best_val_loss (float): Best validation loss recorded in the checkpoint.
            - selection (str): Saved plane selection from the checkpoint.
            - no_improvement_epochs (int): Number of consecutive epochs without improvement from the checkpoint.
    """
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint found! Epoch: {checkpoint['epoch']}")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        selection = checkpoint.get("selection", "axial")
        no_improvement_epochs = checkpoint.get("no_improvement_epochs", 0)
    else:
        print("No checkpoint provided.")
        start_epoch = 0
        best_val_loss = float("inf")
        selection = "axial"
        no_improvement_epochs = 0

    return start_epoch, best_val_loss, selection, no_improvement_epochs


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Namespace object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the model with parameters from a yaml file."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint file.")
    return parser.parse_args()


def load_config(yaml_path):
    """
    Load configuration from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary containing the loaded configuration.
    """
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def seed_everything(deterministic=False, seed=42):
    """
    Fix the random seed for reproducibility.

    Args:
        deterministic (bool): If True, set PyTorch to use deterministic algorithms.
        seed (int): The seed value to fix random number generation.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
