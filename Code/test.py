# Run this code like python test.py --config example.yaml
# The generated PET images are saved in dataset/{Train or Val or Test}/{OUTPUT_SUBDIR in the yaml file}

import os
import torch
import monai
import numpy as np
import nibabel as nib
from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord, DivisiblePadD
import yaml
import argparse

# Read Config
def load_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

parser = argparse.ArgumentParser(description='Use same yaml file as the one you used for training')
parser.add_argument('--config', required=True, help='Path to the yaml config file')
args = parser.parse_args()
config = load_config(args.config)

# Configurations
NUM_WORKERS = config["NUM_WORKERS"]
MODEL_DIR = config["MODEL_DIR"]
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.pth')
BASE_DIR = 'dataset/'
OUTPUT_SUBDIR = config["OUTPUT_SUBDIR"]
CUDA_SETTING = config["CUDA_SETTING"]
CHANNELS_SETTING = config["CHANNELS_SETTING"]

torch.cuda.set_device(CUDA_SETTING)

def generate_and_save_predictions(data_dir, output_message):
    output_dir = os.path.join(data_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    input_images = [os.path.join(data_dir, "MRI", f) for f in os.listdir(os.path.join(data_dir, "MRI")) if f.endswith('.nii.gz')]
    data_dicts = [{"image": img} for img in input_images]
    keys = ["image"]

    transforms = Compose([
        LoadImaged(keys),
        AddChanneld(keys),
        ToTensord(keys)
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=pad_list_data_collate)

    device = torch.device(CUDA_SETTING if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=tuple(CHANNELS_SETTING),
        strides=(2, 2, 2, 2)
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=CUDA_SETTING)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_ = data["image"].to(device)
            output = model(input_).cpu().squeeze().numpy()

            original_nii = nib.load(input_images[i])
            output_nii = nib.Nifti1Image(output, original_nii.affine, original_nii.header)

            input_file_name = os.path.basename(input_images[i])
            output_file_name = input_file_name.replace('mri', 'output')

            nib.save(output_nii, os.path.join(output_dir, output_file_name))

    print(output_message)


def main():
    generate_and_save_predictions(os.path.join(BASE_DIR, 'Train'), "Train NIFTI completed.")
    generate_and_save_predictions(os.path.join(BASE_DIR, 'Val'), "Val NIFTI completed.")
    generate_and_save_predictions(os.path.join(BASE_DIR, 'Test'), "Test NIFTI completed.")

if __name__ == "__main__":
    main()
