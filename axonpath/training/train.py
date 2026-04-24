import csv
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import importlib.resources
import numpy as np
from dataclasses import asdict
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from axonpath.transforms.custom_transforms import transforms_fn
from axonpath.utils import (
    model_fn, get_datasets, get_loaders, load_checkpoint, save_checkpoint,
    evaluate, loss_plot_fn, loss_plot_log_fn, plot_segmentation_scores_fn
)
from axonpath.training.train_loop import train_loop
from axonpath.training.config import TrainingConfig
from axonpath.export_utils.model_export import export_torchscript_model


def get_pretrained_path(weight_name_or_path: str) -> str:
    """
    Return the full path to the specified pretrained weights file.
    - If a direct file path is provided and exists, it's returned as-is.
    - Otherwise, locates the file inside the installed axonpath.weights package.
    - Falls back to axonpath/weights/ when running from source.
    """
    if os.path.isfile(weight_name_or_path):
        return weight_name_or_path

    # Installed package: resolve via importlib.resources
    try:
        ref = importlib.resources.files("axonpath.weights").joinpath(weight_name_or_path + ".pth")
        path = str(ref)
        if os.path.isfile(path):
            return path
    except (TypeError, FileNotFoundError):
        pass

    # Source tree fallback
    dev_weights_path = Path(__file__).resolve().parent.parent / "weights" / (weight_name_or_path + ".pth")
    if dev_weights_path.is_file():
        return str(dev_weights_path)

    raise FileNotFoundError(f"Pretrained weights '{weight_name_or_path}' not found in package or local dev path.")


def train(config: TrainingConfig):
    os.makedirs(config.output_dir, exist_ok=True)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    device = config.device
    norm_type = "instance" if config.batch_size < 8 else "batch"

    # If pretrained weights are provided, read their stored norm_type so the model
    # is built with matching normalisation layers before the state_dict is loaded.
    pretrained_path = None
    if config.pretrained_weights:
        pretrained_path = get_pretrained_path(config.pretrained_weights)
        ckpt_peek = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        stored_norm_type = ckpt_peek.get("norm_type", norm_type)
        if stored_norm_type != norm_type:
            print(f"Pretrained weights use norm_type='{stored_norm_type}' "
                  f"(batch_size={config.batch_size} would give '{norm_type}'). "
                  f"Using '{stored_norm_type}' to match pretrained weights.")
            norm_type = stored_norm_type

    config_dict = asdict(config)
    config_dict["norm_type"] = norm_type
    if config.load_checkpoint:
        # Avoid overwriting the original config; record resumed parameters separately
        config_filename = f"training_config_resumed_epoch{config_dict.get('num_epochs', 0)}.json"
    else:
        config_filename = "training_config.json"
    with open(os.path.join(config.output_dir, config_filename), "w") as f:
        json.dump(config_dict, f, indent=2)

    train_tf, val_tf = transforms_fn(config.image_height, config.image_width)
    train_ds, val_ds = get_datasets(config.train_dir, config.val_dir, train_tf, val_tf)
    train_ds.populate_cache()
    val_ds.populate_cache()

    model = model_fn(device, norm_type=norm_type)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    loss_fns = [ce_loss, mse_loss]
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device)

    scheduler = None
    if config.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-6)

    train_loss, val_loss = [], []
    f1_fibre, f1_axon, f1_inner_cylinder = [], [], []
    balanced_seg_score, best_score = [], 0
    last_epoch = 0
    _es_best = 0.0
    _es_counter = 0

    csv_path = os.path.join(config.output_dir, "training_log.csv")
    _csv_header_written = os.path.exists(csv_path) and config.load_checkpoint

    checkpoint_path = os.path.join(config.output_dir, "model_checkpoint.pth.tar")
    best_weights_path = os.path.join(config.output_dir, "best_weights_model.pth")
    last_weights_path = os.path.join(config.output_dir, "last_epoch_model.pth")

    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(ckpt_peek["state_dict"])
    elif config.load_checkpoint:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        last_epoch, train_loss, val_loss, f1_fibre, f1_axon, f1_inner_cylinder, balanced_seg_score, best_score = \
            load_checkpoint(checkpoint, model, optimizer)
        if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    train_loader, val_loader = get_loaders(train_ds, val_ds, config.batch_size, config.num_workers, config.pin_memory)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        if config.load_checkpoint:
            print(f"Total epoch {last_epoch + epoch + 1}/{last_epoch + config.num_epochs}")

        t_loss = train_loop(train_loader, model, optimizer, loss_fns, scaler, device, config.loss_weights)
        train_loss.append(t_loss)

        show_results_epochs = [int(config.num_epochs * f) - 1 for f in [0.25, 0.5, 0.75, 1.0]]
        show_results_epochs = sorted(set(min(max(e, 0), config.num_epochs - 1) for e in show_results_epochs))
        show_results = epoch in show_results_epochs
        v_loss, f1_fib, f1_ax, f1_in, _ = evaluate(val_loader, model, loss_fns, device, config.fibre_threshold, config.axon_threshold, config.min_diameter, show_results, output_dir=config.output_dir, loss_weights=config.loss_weights)
        val_loss.append(v_loss)
        f1_fibre.append(f1_fib)
        f1_axon.append(f1_ax)
        f1_inner_cylinder.append(f1_in)

        score = (f1_fib + f1_ax + f1_in) / 3
        balanced_seg_score.append(score)

        if scheduler is not None:
            scheduler.step(v_loss)

        print(f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | F1 Fibre: {f1_fib:.4f} | F1 Axon: {f1_ax:.4f} | F1 InTo: {f1_in:.4f} | Balanced Segmentation Score: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "norm_type": norm_type}, best_weights_path)

        # CSV logging
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not _csv_header_written:
                writer.writerow(["epoch", "train_loss", "val_loss", "f1_fibre", "f1_axon", "f1_inner_cylinder", "balanced_seg_score"])
                _csv_header_written = True
            writer.writerow([epoch + 1 + last_epoch, t_loss, v_loss, f1_fib, f1_ax, f1_in, score])

        # Early stopping
        if config.early_stopping_patience > 0:
            if score >= _es_best + config.early_stopping_min_delta:
                _es_best = score
                _es_counter = 0
            else:
                _es_counter += 1
            if _es_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1 + last_epoch}: "
                      f"no improvement > {config.early_stopping_min_delta} for "
                      f"{config.early_stopping_patience} consecutive epochs.")
                break

        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "norm_type": norm_type,
            "epoch": epoch + 1 + last_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_fibre": f1_fibre,
            "f1_axon": f1_axon,
            "f1_inner_cylinder": f1_inner_cylinder,
            "balanced_segmentation_score": balanced_seg_score,
            "best_score": best_score,
        }, filename=checkpoint_path)

    torch.save({"state_dict": model.state_dict(), "norm_type": norm_type}, last_weights_path)
    loss_plot_fn(train_loss, val_loss, output_dir=config.output_dir)
    loss_plot_log_fn(train_loss, val_loss, output_dir=config.output_dir)
    plot_segmentation_scores_fn(f1_fibre, f1_axon, f1_inner_cylinder, balanced_seg_score, output_dir=config.output_dir)

    # Export torchscript model
    export_model = model_fn(config.device, norm_type=norm_type)
    export_model.load_state_dict(torch.load(best_weights_path, map_location=config.device, weights_only=True)["state_dict"])
    export_torchscript_model(export_model, os.path.join(config.output_dir, config.model_name + ".pt"))
