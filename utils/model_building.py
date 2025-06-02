import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from dataset.aimseg_dataset import AimSegDataset



def model_fn(device, norm_type="batch"):
    if norm_type == "batch":
        norm = Norm.BATCH
    elif norm_type == "group":
        norm = Norm.INSTANCE  # MONAI does not use GroupNorm directly, but INSTANCE works similarly for small batches
    else:
        raise ValueError("Unsupported norm type: choose 'batch' or 'group'")

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=5,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.25,
        norm=norm,
    ).to(device)

    return model

def get_datasets(
        train_tile_dir,
        val_tile_dir,
        train_transform,
        val_transform,
    ):
    train_dataset = AimSegDataset(
        tile_dir=train_tile_dir,
        transform=train_transform,
        cache=True
    )
    val_dataset = AimSegDataset(
        tile_dir=val_tile_dir,
        transform=val_transform,
        cache=True
    )
    return train_dataset, val_dataset

def worker_init_fn(worker_id):
    # Each worker gets a unique seed based on the initial seed and worker ID
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

def get_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader