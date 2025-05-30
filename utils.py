import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import monai
from monai.metrics import DiceMetric
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import colorcet as cc

from dataset import AimSegDataset
from evaluation2 import SegmentationEvaluator
from image_processing import (
    postprocessing_sdt,
    last_layer_fn
)

def get_glasbey_cmap():
    l = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
    l[0] = [0, 0, 0]
    return LinearSegmentedColormap.from_list('glasbey', l, N=256)

def apply_cmap(image, cmap):
    if cmap == "glasbey":
        glasbey = get_glasbey_cmap()
        rgba = glasbey(image / np.max(image))
        return np.uint8(rgba[:, :, :3] * 255), None  # RGB image, no colorbar
    else:
        return image, cmap  # Return as-is for matplotlib to handle

def get_glasbey_cmap():
    l = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
    l[0] = [0, 0, 0]
    return ListedColormap(l)

def show_images(*images, titles=None, cmaps=None, n_cols=3, figsize=(15, 10)):
    n_images = len(images)
    titles = titles or [f"Image {i}" for i in range(n_images)]
    cmaps = cmaps or ["gray"] * n_images

    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]

    for idx, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        ax = axes[idx]
        img_np = img.cpu().numpy() if hasattr(img, "cpu") else img
        img_np = np.squeeze(img_np)

        if cmap == "glasbey":
            cmap_obj = get_glasbey_cmap()
            im = ax.imshow(img_np, cmap=cmap_obj, interpolation="nearest", vmin=0, vmax=cmap_obj.N - 1)
            # Manually add colorbar
            norm = mpl.colors.Normalize(vmin=0, vmax=cmap_obj.N - 1)
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(img_np, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title)
        ax.axis("off")

    for ax in axes[n_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    last_epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss'] 
    val_loss = checkpoint['val_loss']
    f1_fibre = checkpoint['f1_fibre']
    f1_axon = checkpoint['f1_axon']
    dice_score = checkpoint['dice_score']
    balanced_segmentation_score = checkpoint['balanced_segmentation_score']
    best_score = checkpoint['best_score']
    print("Loading checkpoint")
    return last_epoch, train_loss, val_loss, f1_fibre, f1_axon, dice_score, balanced_segmentation_score, best_score

def model_fn(device):
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=5,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.25,
    ).to(device)
    return model

def get_datasets(
        train_dir,
        train_maskdir,
        train_labeldir,
        val_dir,
        val_maskdir,
        val_labeldir,
        train_transform,
        val_transform,
    ):
    train_dataset = AimSegDataset(
        image_dir=train_dir,
        masksem_dir=train_maskdir,
        maskins_dir=train_labeldir,
        transform=train_transform,
    )
    val_dataset = AimSegDataset(
        image_dir=val_dir,
        masksem_dir=val_maskdir,
        maskins_dir=val_labeldir,
        transform=val_transform,
    )
    return train_dataset, val_dataset

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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def evaluate_fn(loader, model, loss_fn, device="cuda", show_results=False):
    model.eval()
    val_loss = []
    f1_scores_fibre = []
    f1_scores_axon = []
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, num_classes = 3)

    with torch.no_grad():
        for x, y in loader:
            # Move input to device
            x = x.to(device)

            # Predict
            prediction = model(x)

            # Prepare target
            y = np.stack(y, axis=1)
            y = torch.tensor(y).to(device=device)

            # Compute loss
            val_crossentropy_loss = loss_fn[0](prediction[:, 0:3, :, :], y[:, 2, :, :].long()) # semantic
            val_mse_loss1 = loss_fn[1](prediction[:, 3, :, :], y[:, 3, :, :].float()) # fibre DT
            val_mse_loss2 = loss_fn[1](prediction[:, 4, :, :], y[:, 4, :, :].float()) # axon DT
            val_loss.append((val_crossentropy_loss + val_mse_loss1 + val_mse_loss2).item())

            # Dice Score
            sem_output = last_layer_fn(prediction[:, 0:3, :, :])
            y_sem = y[:, 2, :, :]  # only semantic GT
            y_onehot = nn.functional.one_hot(y_sem.long(), num_classes=3).permute(0, 3, 1, 2).float()
            pred_onehot = nn.functional.one_hot(sem_output.long(), num_classes=3).permute(0, 3, 1, 2).float()
            dice_metric(y_pred=pred_onehot, y=y_onehot)

            # F1 Scores
            for i in range(x.shape[0]):
                # Predicted instances
                pred_fibre = postprocessing_sdt(prediction[i:i+1, 3, :, :])
                pred_axon = postprocessing_sdt(prediction[i:i+1, 4, :, :])

                # Ground truth instances
                gt_fibre = y[i, 0, :, :].cpu().numpy().astype(np.int32)  # assuming channel 0 is fibre
                gt_axon = y[i, 1, :, :].cpu().numpy().astype(np.int32)  # assuming channel 1 is axon

                # Evaluate each
                evaluator_fibre = SegmentationEvaluator(gt_fibre, pred_fibre)
                evaluator_axon = SegmentationEvaluator(gt_axon, pred_axon)

                f1_fibre = evaluator_fibre.f1_mean(evaluator_fibre.evaluate_multiple_thresholds(f"sample_{i}_fibre"))
                f1_axon = evaluator_axon.f1_mean(evaluator_axon.evaluate_multiple_thresholds(f"sample_{i}_axon"))

                f1_scores_fibre.append(f1_fibre)
                f1_scores_axon.append(f1_axon)

        dice_score = dice_metric.aggregate().item()
        dice_metric.reset()

        if show_results:
            semantic = last_layer_fn(prediction[0:1, 0:3, :, :])
            labels_fibre = postprocessing_sdt(prediction[0:1, 3, :, :])
            labels_axon = postprocessing_sdt(prediction[0:1, 4, :, :])
            show_images(
                x[0].cpu(),
                y[0, 0, :, :].cpu(),
                y[0, 1, :, :].cpu(),
                y[0, 2, :, :].cpu(),
                prediction[0, 3, :, :].cpu(),
                prediction[0, 4, :, :].cpu(),
                semantic,
                labels_fibre,
                labels_axon,
                titles=[
                    "Image",  "Target Fibre", "Target Axon",
                    "Target Semantic", "Pred Fibre SDT", "Pred Axon SDT",
                    "Prediction Semantic", "Prediction Fibre Instance", "Prediction Axon Instance"
                ],
                cmaps=[
                    "gray", "glasbey", "glasbey",
                    "viridis", "magma", "magma",
                    "viridis", "glasbey", "glasbey"
                ],
                n_cols=3
            )

    model.train()

    val_loss_mean = sum(val_loss) / len(val_loss)
    f1_fibre = (sum(f1_scores_fibre)) / (len(f1_scores_fibre)) if (f1_scores_fibre) else 0.0
    f1_axon = (sum(f1_scores_axon)) / (len(f1_scores_axon)) if (f1_scores_axon) else 0.0

    return val_loss_mean, f1_fibre, f1_axon, dice_score

def loss_plot_fn(train_loss, val_loss):
    # plot train and val loss
    print("\n----------------------------------------------------------------------------")
    print("\nTraining and validation loss")
    
    fig_loss = plt.gcf()
    
    plt.plot(np.arange(1,len(train_loss)+1).tolist(), train_loss, label = "Training loss")
    plt.plot(np.arange(1,len(val_loss)+1).tolist(), val_loss, label = "Validation loss")
    plt.title('Training and validation loss vs epoch number (linear)')
    plt.ylabel("Loss")
    plt.xlabel("Epoch number")
    plt.xticks(ticks=np.arange(0, len(val_loss)+1, (len(val_loss))/4).tolist())
    plt.legend()
    
    fig_loss.set_facecolor('white')
    fig_loss.savefig('loss_plot.png', bbox_inches='tight', dpi=300)
    plt.show()

def loss_plot_log_fn(train_loss, val_loss):
    # Plot train and val loss in a log scale
    print("\n----------------------------------------------------------------------------")
    print("\nTraining and validation loss")
    
    fig_loss = plt.gcf()
    
    plt.plot(np.arange(1, len(train_loss) + 1).tolist(), train_loss, label="Training loss")
    plt.plot(np.arange(1, len(val_loss) + 1).tolist(), val_loss, label="Validation loss")
    
    plt.yscale('log')  # Apply a logarithmic scale to the y-axis
    plt.title('Training and validation loss vs epoch number (log)')
    plt.ylabel("Log Loss")
    plt.xlabel("Epoch number")
    plt.xticks(ticks=np.arange(0, len(val_loss)+1, (len(val_loss))/4).tolist())
    plt.legend()    
    
    fig_loss.set_facecolor('white')
    fig_loss.savefig('loss_plot_log.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_segmentation_scores_fn(f1_fibre, f1_axon, dice_score, balanced_segmentation_score):
    print("\n----------------------------------------------------------------------------")
    print("Segmentation metric trends over epochs")

    epochs = np.arange(1, len(f1_fibre) + 1)
    fig_scores = plt.figure(figsize=(10, 5))

    plt.plot(epochs, f1_fibre, label="F1 Fibre")
    plt.plot(epochs, f1_axon, label="F1 Axon")
    plt.plot(epochs, dice_score, label="Dice Score")
    plt.plot(epochs, balanced_segmentation_score, label="Balanced Segmentation Score")

    # Find best epoch (based on max balanced segmentation score)
    best_epoch = np.argmax(balanced_segmentation_score) + 1
    best_score = balanced_segmentation_score[best_epoch - 1]

    # Draw vertical line at best epoch
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.text(best_epoch + 0.5, 0.95, f'{best_score:.3f}', color='red')

    plt.title("Segmentation Scores per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(ticks=np.linspace(1, len(f1_fibre), 5).astype(int))
    plt.legend()
    plt.grid(True)
    fig_scores.set_facecolor('white')
    plt.tight_layout()

    plt.savefig("segmentation_scores_plot.png", bbox_inches='tight', dpi=300)
    plt.show()



