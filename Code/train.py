# Run this code like python train.py --config example.yaml
# If you have a checkpoint, run this code like python train.py --config example.yaml --checkpoint /path/to/the/chechpoint.pth

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import monai
from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord, Rand3DElasticd, RandAffined, RandFlipd, RandGaussianNoised
from monai.losses import SSIMLoss
import yaml
import argparse
import numpy as np
import time

# Read Config
def load_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

parser = argparse.ArgumentParser(description='Train the model with parameters from a yaml file.')
parser.add_argument('--config', required=True, help='Path to the yaml config file')
parser.add_argument('--checkpoint', default=None, help='Path to a checkpoint file')
args = parser.parse_args()
config = load_config(args.config)

# Configurations
TRAIN_DIR = 'dataset/Train' # Train dataset location
VAL_DIR = 'dataset/Val' # Validation dataset location

# Read values from a yaml file
LOG_DIR = config["LOG_DIR"]
MODEL_DIR = config["MODEL_DIR"]
LOG_FILE = os.path.join(LOG_DIR, 'log.txt')
BATCH_SIZE = config["BATCH_SIZE"]
NUM_WORKERS = config["NUM_WORKERS"]
MAX_EPOCHS = config["MAX_EPOCHS"]
LEARNING_RATE = float(config["LEARNING_RATE"])
USE_DROPOUT = config["USE_DROPOUT"]
DROPOUT = config["DROPOUT"]
USE_AUGMENTATION = config["USE_AUGMENTATION"]
EARLY_STOPPING_PATIENCE = config["EARLY_STOPPING_PATIENCE"]
ALPHA = config["ALPHA"]
BETA = config["BETA"]
GAMMA = config["GAMMA"]
CHANNELS_SETTING = tuple(config["CHANNELS_SETTING"])
ELASTIC_SIGMA_RANGE = tuple(config["ELASTIC_SIGMA_RANGE"])
ELASTIC_MAGNITUDE_RANGE = tuple(config["ELASTIC_MAGNITUDE_RANGE"])
USE_RANDELASTIC = config["USE_RANDELASTIC"]
USE_RANDAFFINED = config["USE_RANDAFFINED"]
USE_RANDFLIP = config["USE_RANDFLIP"]
USE_RANDGAUSSIAN = config["USE_RANDGAUSSIAN"]
ELACTIC_PROB = config["ELACTIC_PROB"]
RANDAFFINE_PROB = config["RANDAFFINE_PROB"]
RANDFLIP_PROB = config["RANDFLIP_PROB"]
RANDGAUSSIAN_PROB = config["RANDGAUSSIAN_PROB"]
CUDA_SETTING = config["CUDA_SETTING"]
INTERVAL = config["INTERVAL"]
GRAD_ACCUM_STEPS = config["GRAD_ACCUM_STEPS"]
EXPERIMENT_NAME = config["OUTPUT_SUBDIR"]
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Cyclic 2D perceptual loss function
class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        device = torch.device(CUDA_SETTING if torch.cuda.is_available() else "cpu")
        self.vgg = nn.Sequential(*list(vgg_pretrained_features)[:23]).eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, y, selection):
        loss = 0.0

        if selection == 'coronal':
            slide_counter = 0
            for d in range(x.shape[3]):
                slide_counter += 1
                x_slice = x[:, :, :, d, :] # Groundtruth PET coronal slide
                y_slice = y[:, :, :, d, :] # Generated PET coronal slide

                # Scale into (0, 1)
                x_slice = (x_slice-x_slice.min()) / (x_slice.max()-x_slice.min()+ 1e-8)
                y_slice = (y_slice-y_slice.min()) / (y_slice.max()-y_slice.min()+ 1e-8)

                # Pre-Trained VGG has RGB values, so change the original image from 1 channel to 3 channel by repeating
                x_slice = x_slice.repeat(1, 3, 1, 1)
                y_slice = y_slice.repeat(1, 3, 1, 1)

                # Get feature maps
                x_vgg, y_vgg = self.vgg(x_slice), self.vgg(y_slice)

                del x_slice, y_slice

                # Get MSE between the feature maps
                loss += nn.functional.mse_loss(x_vgg, y_vgg)
                
                del x_vgg, y_vgg
 
            return loss / slide_counter

        elif selection == 'axial':
            slide_counter = 0
            for h in range(x.shape[4]):
                slide_counter += 1
                x_slice = x[:, :, :, :, h] # Groundtruth PET axial slide
                y_slice = y[:, :, :, :, h] # Generated PET axial slide

                # Scale into 0, 1
                x_slice = (x_slice-x_slice.min()) / (x_slice.max()-x_slice.min()+ 1e-8)
                y_slice = (y_slice-y_slice.min()) / (y_slice.max()-y_slice.min()+ 1e-8)

                # Pre-Trained VGG has RGB values, so change the original image from 1 channel to 3 channel by repeating
                x_slice = x_slice.repeat(1, 3, 1, 1)
                y_slice = y_slice.repeat(1, 3, 1, 1)

                # Get feature maps
                x_vgg, y_vgg = self.vgg(x_slice), self.vgg(y_slice)

                del x_slice, y_slice

                # Get MSE between the feature maps
                loss += nn.functional.mse_loss(x_vgg, y_vgg)
                
                del x_vgg, y_vgg
            return loss / slide_counter

        elif selection == 'sagittal':
            slide_counter = 0
            for w in range(x.shape[2]):
                slide_counter += 1
                x_slice = x[:, :, w, :, :] # Groundtruth PET sagittal slide
                y_slice = y[:, :, w, :, :] # Generated PET sagittal slide 

                # Scale into 0, 1
                x_slice = (x_slice-x_slice.min()) / (x_slice.max()-x_slice.min()+ 1e-8)
                y_slice = (y_slice-y_slice.min()) / (y_slice.max()-y_slice.min()+ 1e-8)

                # Pre-Trained VGG has RGB values, so change the original image from 1 channel to 3 channel by repeating
                x_slice = x_slice.repeat(1, 3, 1, 1)
                y_slice = y_slice.repeat(1, 3, 1, 1)

                # Get feature maps
                x_vgg, y_vgg = self.vgg(x_slice), self.vgg(y_slice)

                del x_slice, y_slice

                # Get MSE between the feature maps
                loss += nn.functional.mse_loss(x_vgg, y_vgg)
                
                del x_vgg, y_vgg
            return loss / slide_counter

        # if selection is neither axial, sagittal, nor coronal
        else:
            print('Error! selection should be either \'axial\', \'coronal\', or \'sagittal\'!')
            exit()

# Combined loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss() # MSE loss
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=2.0) # SSIM loss
        self.perceptual_loss = PerceptualLoss() # Cyclic 2.5D perceptual loss

        # Hyperparameters
        self.alpha = ALPHA
        self.beta = BETA
        self.gamma = GAMMA

    # Return combined loss
    def forward(self, output, target, model, selection):
        mse = self.mse_loss(output, target)
        ssim = self.ssim_loss(output, target)
        p_loss = self.perceptual_loss(output, target, selection)
        return (self.alpha * mse) + (self.beta * ssim) + (self.gamma * p_loss)


def load_data():
    train_images = sorted(glob.glob(os.path.join(TRAIN_DIR, "MRI", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(TRAIN_DIR, "PET", "*.nii.gz")))
    val_images = sorted(glob.glob(os.path.join(VAL_DIR, "MRI", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(VAL_DIR, "PET", "*.nii.gz")))

    train_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
    val_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]

    keys = ["image", "label"]

    transforms_val = Compose([
        LoadImaged(keys),
        AddChanneld(keys),
        ToTensord(keys)
    ])

    # Data augmentation for Train dataset
    transforms_train_setting = [
        LoadImaged(keys),
        AddChanneld(keys)
    ]

    if USE_AUGMENTATION:
        if USE_RANDELASTIC:
            transforms_train_setting += [Rand3DElasticd(
                    keys,
                    mode=('trilinear', 'trilinear'),
                    prob=ELACTIC_PROB,
                    sigma_range=ELASTIC_SIGMA_RANGE,
                    magnitude_range=ELASTIC_MAGNITUDE_RANGE,
                    padding_mode = 'zeros'
                )
            ]
        if USE_RANDAFFINED:
            transforms_train_setting += [RandAffined(
                    keys,
                    prob=RANDAFFINE_PROB,
                    rotate_range=(np.pi/12, np.pi/12, np.pi/12),
                    scale_range=(0.1, 0.1, 0.1),
                    spatial_size=None
                )
            ]
        if USE_RANDFLIP:
            transforms_train_setting += [RandFlipd(keys, spatial_axis=[0, 1, 2], prob=RANDFLIP_PROB)]
        if USE_RANDGAUSSIAN:
            transforms_train_setting += [RandGaussianNoised(["image"], prob=RANDGAUSSIAN_PROB, mean=0, std=0.1)]

    transforms_train = Compose(transforms_train_setting + [ToTensord(keys)])

    train_dataset = Dataset(data=train_data_dicts, transform=transforms_train)
    val_dataset = Dataset(data=val_data_dicts, transform=transforms_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=pad_list_data_collate)

    return train_loader, val_loader

def train_model(train_loader, val_loader, checkpoint_path=None):
    start_time = time.time()
    log_message = ""
    device = torch.device(CUDA_SETTING if torch.cuda.is_available() else "cpu")

    # Model setting
    if USE_DROPOUT:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=CHANNELS_SETTING,
            strides=(2, 2, 2, 2),
            dropout=DROPOUT
        ).to(device)

    else:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=CHANNELS_SETTING,
            strides=(2, 2, 2, 2)
        ).to(device)
    
    loss_function = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE) # Adam optimiser
    scheduler = CosineAnnealingLR(optimizer, T_max=int((INTERVAL+0.5)/2), eta_min=0) # for odd numbers

    # best loss values for each plane
    best_val_loss_initial = float('inf')
    best_val_loss_first_change = float('inf')
    best_val_loss_second_change = float('inf')
    no_improvement_epochs = 0
    selection = 'axial'

    # Algorithm 2 in the paper
    ax = []
    co = []
    sa = []
    current_num = 0
    exit_condition = False
    interval = INTERVAL
    while (current_num < MAX_EPOCHS) and (exit_condition is False):
        ax.append(current_num)
        if (current_num + interval) < MAX_EPOCHS:
            co.append(current_num + interval)
            if (current_num + interval*2) < MAX_EPOCHS:
                sa.append(current_num + interval*2)
            else:
                exit_condition = True
        else:
            exit_condition = True
        current_num = current_num + interval*3
        if interval//1.5 > EARLY_STOPPING_PATIENCE:
            interval = interval // 1.5
    
    print('ax: ', ax)
    print('co: ', co)
    print('sa: ', sa)
    print('Max Epoch: ', MAX_EPOCHS)

    # read checkpoint file if given
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        print(f"checkpoint found!: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss_initial = checkpoint.get('best_val_loss_initial', float('inf'))
        best_val_loss_first_change = checkpoint.get('best_val_loss_first_change', float('inf'))
        best_val_loss_second_change = checkpoint.get('best_val_loss_second_change', float('inf'))
        selection = checkpoint.get('selection', 'axial')
        no_improvement_epochs = checkpoint.get('no_improvement_epochs', 0)

    else:
        print('no checkpoint')
        start_epoch = 0


    # Training
    for epoch in range(start_epoch, MAX_EPOCHS):
        
        # Show how much time passed for every 50 epochs
        if epoch % 50 == 0 and epoch != 0:
            end_time = time.time()
            print(f"50 epochs: {int((end_time - start_time)//3600)} hours {int(((end_time - start_time) % 3600) // 60)} minutes {(end_time - start_time) % 60} seconds")
            start_time = time.time()
            
        if epoch in ax:
            best_val_loss_initial = float('inf')
            selection = 'axial'
            print('Perceptual Loss is calculated based on axial')

        if epoch in co:
            best_val_loss_first_change = float('inf')
            selection = 'coronal'
            print('Perceptual Loss is now calculated based on coronal slices (previous: axial)')

        if epoch in sa:
            best_val_loss_second_change = float('inf')
            selection = 'sagittal'
            print('Perceptual Loss is now calculated based on sagittal slices (previous: coronal)')
        
        # Train
        model.train()
        epoch_loss = 0
        for idx, batch_data in enumerate(train_loader):
            inputs, targets = batch_data["image"].to(device), batch_data["label"].to(device)
            outputs = model(inputs)
            # print(type(outputs))
            loss = loss_function(outputs, targets, model, selection)
            (loss/GRAD_ACCUM_STEPS).backward()
            if (idx+1) % GRAD_ACCUM_STEPS == 0 or idx == len(train_loader)-1:
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            del inputs
            del targets
            del outputs
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_data in val_loader:
                val_inputs, val_targets = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_inputs)
                loss = loss_function(val_outputs, val_targets, model, selection)
                val_loss += loss.item()
                del val_inputs
                del val_targets
                del val_outputs

        average_train_loss = epoch_loss / len(train_loader)
        average_val_loss = val_loss / len(val_loader)

        # Show losses
        print(f"Epoch {epoch+1}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}")
        
        log_message += f"Epoch {epoch+1}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}\n"
        
        scheduler.step()

        # Save the checkpoint file and the log file
        if selection == 'axial':
            if average_val_loss < best_val_loss_initial:
                best_val_loss_initial = average_val_loss
                no_improvement_epochs = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss_initial': best_val_loss_initial,
                    'best_val_loss_first_change': best_val_loss_first_change,
                    'best_val_loss_second_change': best_val_loss_second_change,
                    'no_improvement_epochs': no_improvement_epochs,
                    'selection': selection
                }
                torch.save(checkpoint, os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
                torch.save(checkpoint, os.path.join(MODEL_DIR, 'final_model.pth'))
            else:
                # Early stopping is activated in the late stages to prevent it from activating when the model is overfitted into a specific plane
                if epoch > ax[3]:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= EARLY_STOPPING_PATIENCE and epoch > INTERVAL*3:
                        print("Early stopping due to no improvement")
                        break
            with open(LOG_FILE, 'w') as f:
                f.write("Training completed with best validation loss: " + str(best_val_loss_initial) + '\n' + '\n' + log_message)

        elif selection == 'coronal':
            if average_val_loss < best_val_loss_first_change:
                best_val_loss_first_change = average_val_loss
                no_improvement_epochs = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss_initial': best_val_loss_initial,
                    'best_val_loss_first_change': best_val_loss_first_change,
                    'best_val_loss_second_change': best_val_loss_second_change,
                    'no_improvement_epochs': no_improvement_epochs,
                    'selection': selection
                }
                torch.save(checkpoint, os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
                torch.save(checkpoint, os.path.join(MODEL_DIR, 'final_model.pth'))
            else:
                if epoch > ax[3]:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= EARLY_STOPPING_PATIENCE and epoch > INTERVAL*3:
                        print("Early stopping due to no improvement")
                        break
            with open(LOG_FILE, 'w') as f:
                f.write("Training completed with best validation loss: " + str(best_val_loss_first_change) + '\n' + '\n' + log_message)

        else:
            if average_val_loss < best_val_loss_second_change:
                best_val_loss_second_change = average_val_loss
                no_improvement_epochs = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss_initial': best_val_loss_initial,
                    'best_val_loss_first_change': best_val_loss_first_change,
                    'best_val_loss_second_change': best_val_loss_second_change,
                    'no_improvement_epochs': no_improvement_epochs,
                    'selection': selection
                }
                torch.save(checkpoint, os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
                torch.save(checkpoint, os.path.join(MODEL_DIR, 'final_model.pth'))
            else:
                if epoch > ax[3]:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= EARLY_STOPPING_PATIENCE and epoch > INTERVAL*3:
                        print("Early stopping due to no improvement")
                        break
            with open(LOG_FILE, 'w') as f:
                f.write("Training completed with best validation loss: " + str(best_val_loss_second_change) + '\n' + '\n' + log_message)
       
def main():
    torch.cuda.set_device(CUDA_SETTING)
    train_loader, val_loader = load_data()
    train_model(train_loader, val_loader, checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()