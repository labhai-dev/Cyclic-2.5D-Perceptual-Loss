# Run this code like python test.py --config example.yaml
# The generated PET images are saved in dataset/Test/{OUTPUT_SUBDIR in the yaml file}

import os
import glob
import torch
import nibabel as nib
from utils.utils import parse_arguments, load_config
from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
)


def generate_and_save_predictions(
    data_dir,
    output_subdir,
    model_path,
    channels_setting,
    device,
    num_workers=0,
    output_message="",
):
    """
    Generate predictions for all .nii.gz files under 'MRI' in data_dir,
    and save them to 'output_subdir' within the same data_dir.

    Args:
        data_dir (str): The base directory containing MRI input data.
        output_subdir (str): Subdirectory name to store the output NIfTI files.
        model_path (str): Path to the saved model checkpoint.
        channels_setting (tuple): Channel configuration for UNet.
        device (torch.device): Torch device ('cuda' or 'cpu').
        num_workers (int, optional): Number of workers for DataLoader. Default=0.
        output_message (str, optional): Message to print after processing. Default="".
    """

    # Prepare output directory
    output_dir = os.path.join(data_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Gather input images (assumes 'MRI' folder structure)
    mri_dir = os.path.join(data_dir, "MRI")
    input_files = sorted(glob.glob(os.path.join(mri_dir, "*.nii.gz")))

    # Build MONAI Dataset & DataLoader
    data_dicts = [{"image": file_path} for file_path in input_files]
    data_keys = ["image"]

    inference_transforms = Compose([
        LoadImaged(keys=data_keys),
        EnsureChannelFirstd(keys=data_keys),
        ToTensord(keys=data_keys),
    ])

    inference_dataset = Dataset(data=data_dicts, transform=inference_transforms)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate,
    )

    # Load model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels_setting,
        strides=(2, 2, 2, 2)
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference
    with torch.no_grad():
        for idx, batch_data in enumerate(inference_loader):
            input_tensor = batch_data["image"].to(device)
            output_tensor = model(input_tensor).cpu().squeeze().numpy()

            # Load original to retain affine and header
            original_nifti = nib.load(input_files[idx])
            output_nifti = nib.Nifti1Image(
                output_tensor, 
                affine=original_nifti.affine, 
                header=original_nifti.header
            )

            # Build output filename
            input_filename = os.path.basename(input_files[idx]) 
            output_filename = input_filename.replace("mri", "output")  # or any naming convention you prefer

            nib.save(output_nifti, os.path.join(output_dir, output_filename))

    print(output_message)


def main():
    """
    Main function to run inference on the Test set.
    """
    args = parse_arguments()
    config = load_config(args.config)

    # Extract config
    model_dir = config["MODEL_DIR"]
    model_path = os.path.join(model_dir, config.get("TEST_CHECKPOINT", "final_model.pth"))
    data_dir = config["DATA_DIR"]
    output_subdir = config["OUTPUT_SUBDIR"]
    cuda_setting = config["CUDA_SETTING"]
    channels_setting = tuple(config["CHANNELS_SETTING"])
    num_workers = config["NUM_WORKERS"]

    # Set device
    device = torch.device(cuda_setting if torch.cuda.is_available() else "cpu")
    if isinstance(cuda_setting, int):
        torch.cuda.set_device(cuda_setting)

    # Generate predictions for Test dataset
    generate_and_save_predictions(
        data_dir=os.path.join(data_dir, "Test"),
        output_subdir=output_subdir,
        model_path=model_path,
        channels_setting=channels_setting,
        device=device,
        num_workers=num_workers,
        output_message="Test NIFTI completed."
    )


if __name__ == "__main__":
    main()