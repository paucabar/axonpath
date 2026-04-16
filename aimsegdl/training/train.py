import torch
import torch.nn as nn
import torch.optim as optim
import os
import importlib.resources
from pathlib import Path
from aimsegdl.transforms.custom_transforms import transforms_fn
from aimsegdl.utils import (
    model_fn, get_datasets, get_loaders, load_checkpoint, save_checkpoint,
    evaluate, loss_plot_fn, loss_plot_log_fn, plot_segmentation_scores_fn
)
from aimsegdl.training.train_loop import train_loop
from aimsegdl.training.config import TrainingConfig
from aimsegdl.export_utils.model_export import export_torchscript_model


def get_pretrained_path(weight_name_or_path: str) -> str:
    """
    Return the full path to the specified pretrained weights file.
    - If a direct file path is provided and exists, it's returned as-is.
    - Otherwise, locates the file inside the installed aimsegdl.weights package.
    - Falls back to aimsegdl/weights/ when running from source.
    """
    if os.path.isfile(weight_name_or_path):
        return weight_name_or_path

    # Installed package: resolve via importlib.resources
    try:
        ref = importlib.resources.files("aimsegdl.weights").joinpath(weight_name_or_path + ".pth")
        return str(ref)
    except (TypeError, FileNotFoundError):
        pass

    # Source tree fallback
    dev_weights_path = Path(__file__).resolve().parent.parent / "weights" / (weight_name_or_path + ".pth")
    if dev_weights_path.is_file():
        return str(dev_weights_path)

    raise FileNotFoundError(f"Pretrained weights '{weight_name_or_path}' not found in package or local dev path.")


def train(config: TrainingConfig):
    device = config.device

    train_tf, val_tf = transforms_fn(config.image_height, config.image_width)
    train_ds, val_ds = get_datasets(config.train_dir, config.val_dir, train_tf, val_tf)
    train_ds.populate_cache()
    val_ds.populate_cache()

    norm_type = "instance" if config.batch_size < 8 else "batch"
    model = model_fn(device, norm_type=norm_type)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    loss_fns = [ce_loss, mse_loss]
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device)

    train_loss, val_loss = [], []
    f1_fibre, f1_axon, f1_inner_tongue = [], [], []
    balanced_seg_score, best_score = [], 0
    last_epoch = 0

    if config.pretrained_weights:
        pretrained_path = get_pretrained_path(config.pretrained_weights)
        print(f"Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
    elif config.load_checkpoint:
        last_epoch, train_loss, val_loss, f1_fibre, f1_axon, f1_inner_tongue, balanced_seg_score, best_score = \
            load_checkpoint(torch.load("model_checkpoint.pth.tar", weights_only=False), model, optimizer)

    train_loader, val_loader = get_loaders(train_ds, val_ds, config.batch_size, config.num_workers, config.pin_memory)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        if config.load_checkpoint:
            print(f"Total epoch {last_epoch + epoch + 1}/{last_epoch + config.num_epochs}")

        t_loss = train_loop(train_loader, model, optimizer, loss_fns, scaler, device)
        train_loss.append(t_loss)

        show_results_epochs = [int(config.num_epochs * f) - 1 for f in [0.25, 0.5, 0.75, 1.0]]
        show_results_epochs = sorted(set(min(max(e, 0), config.num_epochs - 1) for e in show_results_epochs))
        show_results = epoch in show_results_epochs
        v_loss, f1_fib, f1_ax, f1_in, _ = evaluate(val_loader, model, loss_fns, device, config.fibre_threshold, config.axon_threshold, config.min_diameter, show_results)
        val_loss.append(v_loss)
        f1_fibre.append(f1_fib)
        f1_axon.append(f1_ax)
        f1_inner_tongue.append(f1_in)

        score = (f1_fib + f1_ax + f1_in) / 3
        balanced_seg_score.append(score)

        print(f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | F1 Fibre: {f1_fib:.4f} | F1 Axon: {f1_ax:.4f} | F1 InTo: {f1_in:.4f} | Balanced Segmentation Score: {score:.4f}")

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
            "f1_inner_tongue": f1_inner_tongue,
            "balanced_segmentation_score": balanced_seg_score,
            "best_score": best_score,
        })

    torch.save(model.state_dict(), "last_epoch_model.pth")
    loss_plot_fn(train_loss, val_loss)
    loss_plot_log_fn(train_loss, val_loss)
    plot_segmentation_scores_fn(f1_fibre, f1_axon, f1_inner_tongue, balanced_seg_score)

    # Export torchscript model
    export_model = model_fn(config.device, norm_type=norm_type)
    export_model.load_state_dict(torch.load("best_weights_model.pth", map_location=config.device, weights_only=True))
    export_torchscript_model(export_model, config.model_name + ".pt")
