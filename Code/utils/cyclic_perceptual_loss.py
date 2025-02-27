import math
import fractions
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
from monai.losses import SSIMLoss


# Cyclic 2.5D perceptual loss function (Algorithm 1 in the paper)
class CyclicPerceptualLoss(nn.Module):
    """
    Implements a cyclic 2.5D perceptual loss using a pre-trained VGG16 network.
    This class extracts feature maps from slices of the input and target volumes
    along a specified axis ('axial', 'coronal', or 'sagittal').

    Args:
        config (dict): Configuration dictionary containing training settings.
                       Must include "CUDA_SETTING" to specify the device (e.g., "cuda:0").
    """

    def __init__(self, config):
        """
        Initialize the CyclicPerceptualLoss module.

        Loads the VGG16 network (up to the first 23 layers) and freezes its parameters.

        Args:
            config (dict): Configuration dictionary with at least "CUDA_SETTING" key.
        """
        super(CyclicPerceptualLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        ).features
        device = torch.device(
            config["CUDA_SETTING"] if torch.cuda.is_available() else "cpu"
        )
        self.vgg = nn.Sequential(*list(vgg_pretrained_features)[:23]).eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _process_slice(self, x_slice, y_slice):
        """
        Process a single slice (2D) from the input and target volumes by:
          1) Normalizing values to [0, 1].
          2) Repeating the channels to match the 3-channel input required by VGG16.
          3) Extracting features from VGG16.
          4) Computing the MSE loss between the extracted feature maps.

        Args:
            x_slice (torch.Tensor): A 2D slice from the input volume (batch x 1 x H x W).
            y_slice (torch.Tensor): A 2D slice from the target volume (batch x 1 x H x W).

        Returns:
            torch.Tensor: Scalar representing the MSE loss on the feature maps for this slice.
        """
        # Normalize to (0, 1)
        x_slice = (x_slice - x_slice.min()) / (x_slice.max() - x_slice.min() + 1e-8)
        y_slice = (y_slice - y_slice.min()) / (y_slice.max() - y_slice.min() + 1e-8)

        # Repeat channel to convert from 1 channel to 3 channels for VGG16 input
        x_slice = x_slice.repeat(1, 3, 1, 1)
        y_slice = y_slice.repeat(1, 3, 1, 1)

        # Extract feature maps using pre-trained VGG
        x_vgg = self.vgg(x_slice)
        y_vgg = self.vgg(y_slice)

        # Calculate MSE loss between feature maps
        return nn.functional.mse_loss(x_vgg, y_vgg)

    def forward(self, x, y, selection):
        """
        Compute the cyclic 2.5D perceptual loss between two volumes by unbinding
        along the specified axis ('axial', 'coronal', or 'sagittal').

        Args:
            x (torch.Tensor): Input volume (e.g., batch x channel x depth x height x width).
            y (torch.Tensor): Target volume (same shape as x).
            selection (str): Specifies which axis to slice along ('axial', 'coronal', 'sagittal').

        Returns:
            torch.Tensor: The average perceptual loss across all slices in the chosen axis.
        """
        axis_map = {"coronal": 3, "axial": 4, "sagittal": 2}
        if selection not in axis_map:
            raise ValueError(
                "Error! selection should be 'axial', 'coronal', or 'sagittal'."
            )

        axis = axis_map[selection]
        slices_x = torch.unbind(x, dim=axis)
        slices_y = torch.unbind(y, dim=axis)

        loss_sum = 0.0
        for x_slice, y_slice in zip(slices_x, slices_y):
            loss_sum += self._process_slice(x_slice, y_slice)

        return loss_sum / len(slices_x)


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions (MSE, SSIM, and a cyclic 2.5D perceptual loss)
    into a single loss value.

    Args:
        config (dict): Configuration dictionary containing at least:
                       - "ALPHA", "BETA", "GAMMA": weighting factors for MSE, SSIM, and perceptual loss.
                       - "CUDA_SETTING": device selection for the perceptual loss network.
    """

    def __init__(self, config):
        """
        Initialize the CombinedLoss module with MSE, SSIM, and CyclicPerceptualLoss.

        Args:
            config (dict): Configuration dictionary.
        """
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # MSE loss
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=2.0)  # SSIM loss
        self.perceptual_loss = CyclicPerceptualLoss(
            config=config
        )  # 2.5D perceptual loss

        # Hyperparameters
        self.alpha = config["ALPHA"]
        self.beta = config["BETA"]
        self.gamma = config["GAMMA"]

    def forward(self, output, target, selection):
        """
        Calculate the combined loss given model outputs and targets.

        The final loss is a weighted sum of:
          - MSE loss
          - SSIM loss
          - Cyclic 2.5D perceptual loss

        Args:
            output (torch.Tensor): Model output volume.
            target (torch.Tensor): Ground truth volume.
            selection (str): Axis selection for the perceptual loss ('axial', 'coronal', 'sagittal').

        Returns:
            torch.Tensor: Combined loss value (scalar).
        """
        mse = self.mse_loss(output, target)
        ssim = self.ssim_loss(output, target)
        p_loss = self.perceptual_loss(output, target, selection)
        return (self.alpha * mse) + (self.beta * ssim) + (self.gamma * p_loss)


def calculate_cycle_epochs(max_epochs, cycle_duration, cycle_factor):
    """
    Calculate the epoch lists where plane selection changes based on the given
    maximum number of epochs, cycle duration, and cycle factor.

    Args:
        max_epochs (int): Maximum number of epochs for training.
        cycle_duration (int): Initial duration of a cycle (in epochs).
        cycle_factor (float or Fraction): Factor by which the cycle duration is updated.

    Returns:
        tuple of lists:
            - ax_epochs (list): List of epochs for axial plane.
            - co_epochs (list): List of epochs for coronal plane.
            - sa_epochs (list): List of epochs for sagittal plane.
    """
    ax_epochs, co_epochs, sa_epochs = [], [], []
    cnt = 0
    cycle_factor = float(fractions.Fraction(cycle_factor))
    current_cycle = cycle_duration

    while cnt < max_epochs:
        ax_epochs.append(cnt)
        if cnt + current_cycle < max_epochs:
            co_epochs.append(cnt + current_cycle)
        if cnt + 2 * current_cycle < max_epochs:
            sa_epochs.append(cnt + 2 * current_cycle)
        cnt += 3 * current_cycle
        current_cycle = math.floor(current_cycle * cycle_factor)

    return ax_epochs, co_epochs, sa_epochs


def update_plane_selection(epoch, ax_epochs, co_epochs, sa_epochs, current_selection):
    """
    Check if the current epoch corresponds to the switching point for axial,
    coronal, or sagittal planes, and update the plane selection accordingly.

    Args:
        epoch (int): Current epoch.
        ax_epochs (list): Epoch list for axial plane.
        co_epochs (list): Epoch list for coronal plane.
        sa_epochs (list): Epoch list for sagittal plane.
        current_selection (str): Current plane selection (e.g., 'axial', 'coronal', 'sagittal').

    Returns:
        tuple:
            - new_selection (str): Updated plane selection.
            - reset_best (bool): Whether to reset the best validation loss upon switching.
    """
    if epoch in ax_epochs:
        return "axial", True
    elif epoch in co_epochs:
        return "coronal", True
    elif epoch in sa_epochs:
        return "sagittal", True
    else:
        return current_selection, False
