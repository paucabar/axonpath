import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
import pkg_resources
from aimsegdl.transforms.custom_transforms import transforms_fn
from aimsegdl.utils import (
    model_fn, get_datasets, get_loaders, load_checkpoint, save_checkpoint,
    evaluate_fn, loss_plot_fn, loss_plot_log_fn, plot_segmentation_scores_fn
)
from aimsegdl.training.train_loop import train_loop
from aimsegdl.training.config import TrainingConfig


try:
    # Use pkg_resources only if available
    import pkg_resources
except ImportError:
    pkg_resources = None

def get_pretrained_path(weight_name_or_path: str) -> str:
    """
    Return the full path to the specified pretrained weights file.
    - If a direct file path is provided and exists, it's returned as-is.
    - Otherwise, it attempts to locate the file in the aimsegdl.weights package (for installed packages).
    - If not found, and running in a development environment, it checks aimsegdl/weights/ folder manually.
    """
    # If it's an existing full path, return it
    if os.path.isfile(weight_name_or_path):
        return weight_name_or_path

    # Try pkg_resources (for installed package)
    if pkg_resources:
        try:
            return pkg_resources.resource_filename("aimsegdl.weights", weight_name_or_path + ".pth")
        except Exception:
            pass  # Fall back to dev mode

    # Fallback: Check local dev path (e.g., aimsegdl/weights/)
    dev_weights_path = Path(__file__).resolve().parent.parent / "weights" / (weight_name_or_path + ".pth")
    if dev_weights_path.is_file():
        return str(dev_weights_path)

    # Not found
    raise FileNotFoundError(f"Pretrained weights '{weight_name_or_path}' not found in package or local dev path.")


def train(config: TrainingConfig):
    device = config.device

    train_tf, val_tf = transforms_fn(config.image_height, config.image_width)
    train_ds, val_ds = get_datasets(config.train_dir, config.val_dir, train_tf, val_tf)
    train_ds.populate_cache()
    val_ds.populate_cache()

    norm_type = "group" if config.batch_size < 8 else "batch"
    model = model_fn(device, norm_type=norm_type)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    loss_fns = [ce_loss, mse_loss]
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    train_loss, val_loss = [], []
    f1_fibre, f1_axon, dice_score = [], [], []
    balanced_seg_score, best_score = [], 0
    last_epoch = 0

    if config.pretrained_weights:
        pretrained_path = get_pretrained_path(config.pretrained_weights)
        print(f"Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path))
    elif config.load_checkpoint:
        last_epoch, train_loss, val_loss, f1_fibre, f1_axon, dice_score, balanced_seg_score, best_score = \
            load_checkpoint(torch.load("model_checkpoint.pth.tar"), model, optimizer)



    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        if config.load_checkpoint:
            print(f"Total epoch {last_epoch + epoch + 1}/{last_epoch + config.num_epochs}")

        train_loader, val_loader = get_loaders(train_ds, val_ds, config.batch_size, config.num_workers, config.pin_memory)

        t_loss = train_loop(train_loader, model, optimizer, loss_fns, scaler, device)
        train_loss.append(t_loss)

        show_results_epochs = [int(config.num_epochs * f) - 1 for f in [0.25, 0.5, 0.75, 1.0]]
        show_results_epochs = sorted(set(min(max(e, 0), config.num_epochs - 1) for e in show_results_epochs))
        show_results = epoch in show_results_epochs
        v_loss, f1_fib, f1_ax, dice = evaluate_fn(val_loader, model, loss_fns, device, show_results)
        val_loss.append(v_loss)
        f1_fibre.append(f1_fib)
        f1_axon.append(f1_ax)
        dice_score.append(dice)

        score = 0.4 * f1_fib + 0.4 * f1_ax + 0.2 * dice
        balanced_seg_score.append(score)

        print(f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | F1 Fibre: {f1_fib:.4f} | F1 Axon: {f1_ax:.4f} | Dice: {dice:.4f} | Balanced Segmentation Score: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_weights_model.pth")

        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1 + last_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_fibre": f1_fibre,
            "f1_axon": f1_axon,
            "dice_score": dice_score,
            "balanced_segmentation_score": balanced_seg_score,
            "best_score": best_score,
        })

    torch.save(model.state_dict(), "last_epoch_model.pth")
    loss_plot_fn(train_loss, val_loss)
    loss_plot_log_fn(train_loss, val_loss)
    plot_segmentation_scores_fn(f1_fibre, f1_axon, dice_score, balanced_seg_score)

    # Export logic
    export_model = model_fn(config.device)
    export_model.load_state_dict(torch.load(f"best_weights_model.pth"))
    if config.bioimageio:
        from aimsegdl.export_utils.model_export import export_bioimageio
        export_bioimageio(export_model, config.model_name + "_bioimageio", True, r"data_tem/test_images/P03B_Frame6_t0.tif")
    else:
        from aimsegdl.export_utils.model_export import export_torchscript_model
        export_torchscript_model(export_model, config.model_name + ".pt")
