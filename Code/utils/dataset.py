import os
import glob
import numpy as np
from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    Rand3DElasticd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    EnsureChannelFirstd,
)


def load_data(config):
    """
    Load training and validation data according to the given config,
    create transforms for data augmentation, and return DataLoaders.

    Args:
        config (dict): Configuration dictionary containing paths and transformation flags.

    Returns:
        (DataLoader, DataLoader): Training and validation DataLoaders.
    """

    # Directories
    data_dir = config["DATA_DIR"]
    train_dir = os.path.expanduser(os.path.join(data_dir, "Train"))
    val_dir = os.path.expanduser(os.path.join(data_dir, "Val"))

    # Transform usage flags
    use_augmentation = config["USE_AUGMENTATION"]
    use_elastic = config["USE_RANDELASTIC"]
    use_affine = config["USE_RANDAFFINED"]
    use_flip = config["USE_RANDFLIP"]
    use_gaussian = config["USE_RANDGAUSSIAN"]

    # Elastic, Affine, Flip, Gaussian noise params
    elastic_sigma_range = tuple(config["ELASTIC_SIGMA_RANGE"])
    elastic_magnitude_range = tuple(config["ELASTIC_MAGNITUDE_RANGE"])
    elastic_prob = config["ELACTIC_PROB"]
    affine_prob = config["RANDAFFINE_PROB"]
    flip_prob = config["RANDFLIP_PROB"]
    gaussian_prob = config["RANDGAUSSIAN_PROB"]

    # DataLoader params
    batch_size = config["BATCH_SIZE"]
    num_workers = config["NUM_WORKERS"]

    print(f"Dataset base directory: {data_dir}")
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")

    # Gather file paths
    train_image_paths = sorted(glob.glob(os.path.join(train_dir, "MRI", "*.nii.gz")))
    train_label_paths = sorted(glob.glob(os.path.join(train_dir, "PET", "*.nii.gz")))
    val_image_paths = sorted(glob.glob(os.path.join(val_dir, "MRI", "*.nii.gz")))
    val_label_paths = sorted(glob.glob(os.path.join(val_dir, "PET", "*.nii.gz")))

    # Create dictionaries for MONAI Dataset
    train_data_dicts = [
        {"image": img, "label": lbl}
        for img, lbl in zip(train_image_paths, train_label_paths)
    ]
    val_data_dicts = [
        {"image": img, "label": lbl}
        for img, lbl in zip(val_image_paths, val_label_paths)
    ]

    # Keys for transforms
    data_keys = ["image", "label"]

    # Validation transforms
    val_transforms = Compose(
        [
            LoadImaged(data_keys, image_only=True),
            EnsureChannelFirstd(data_keys),
            ToTensord(data_keys),
        ]
    )

    # Training transforms (basic loading + optional augmentations)
    train_transforms_list = [
        LoadImaged(data_keys, image_only=True),
        EnsureChannelFirstd(data_keys),
    ]

    if use_augmentation:
        if use_elastic:
            train_transforms_list.append(
                Rand3DElasticd(
                    keys=data_keys,
                    mode=("trilinear", "trilinear"),
                    prob=elastic_prob,
                    sigma_range=elastic_sigma_range,
                    magnitude_range=elastic_magnitude_range,
                    padding_mode="zeros",
                )
            )
        if use_affine:
            train_transforms_list.append(
                RandAffined(
                    keys=data_keys,
                    prob=affine_prob,
                    rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                    scale_range=(0.1, 0.1, 0.1),
                    spatial_size=None,
                )
            )
        if use_flip:
            train_transforms_list.append(
                RandFlipd(keys=data_keys, spatial_axis=[0, 1, 2], prob=flip_prob)
            )
        if use_gaussian:
            train_transforms_list.append(
                RandGaussianNoised(
                    keys=["image"], prob=gaussian_prob, mean=0.0, std=0.1
                )
            )

    train_transforms_list.append(ToTensord(data_keys))

    train_transforms = Compose(train_transforms_list)

    # Create Dataset objects
    train_dataset = Dataset(data=train_data_dicts, transform=train_transforms)
    val_dataset = Dataset(data=val_data_dicts, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate,
    )

    return train_loader, val_loader
