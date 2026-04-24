import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc


def get_glasbey_cmap():
    l = [list(c) for c in cc.cm.glasbey_bw_minc_20_minl_30_r.colors]
    l[0] = [0, 0, 0]
    return LinearSegmentedColormap.from_list('glasbey', l, N=2048)


def get_semantic_cmap():
    """Viridis colormap with black forced at index 0 for background (class 0)."""
    import matplotlib
    viridis = matplotlib.colormaps["viridis"].resampled(256)
    colors = viridis(np.linspace(0, 1, 256))
    colors[0] = [0, 0, 0, 1]
    return LinearSegmentedColormap.from_list("semantic", colors, N=256)


def get_sdt_cmap(base="plasma", background="black"):
    """Colormap for SDT maps: background color for <=0, base colormap gradient for >0.

    Args:
        base: any matplotlib colormap name ("inferno", "viridis", "turbo", "plasma", ...).
        background: "black" or "white".
    """
    import matplotlib
    bg_color = [0, 0, 0, 1] if background == "black" else [1, 1, 1, 1]
    base_cmap = matplotlib.colormaps[base].resampled(256)
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0] = bg_color
    return LinearSegmentedColormap.from_list(f"sdt_{base}_{background}", colors, N=256)


def apply_cmap(image, cmap):
    if cmap == "glasbey":
        glasbey = get_glasbey_cmap()
        rgba = glasbey(image / max(np.max(image), 1))
        return np.uint8(rgba[:, :, :3] * 255), None  # RGB image, no colorbar
    elif cmap == "sdt":
        clipped = np.clip(image, 0, None)
        img_norm = clipped / (clipped.max() + 1e-8)
        return img_norm, get_sdt_cmap()  # uses plasma/black defaults
    else:
        return image, cmap  # Return as-is for matplotlib to handle


def show_images(*images, titles=None, cmaps=None, n_cols=3, figsize=(15, 10), output_dir="."):
    """Display a grid of images with colorbars.

    Args:
        *images: arrays or tensors to display.
        titles: list of titles, one per image.
        cmaps: list of colormap specifiers, one per image. Recognised values:
            - any matplotlib colormap name (e.g. "gray", "viridis")
            - "glasbey": Glasbey categorical colormap, black background, range
              adapted to the actual number of labels in each image.
            - "semantic": modified viridis with black background for class 0.
            - "sdt": SDT colormap (plasma base, black background by default).
              Pass a tuple ("sdt", base, background) to customise, e.g.
              ("sdt", "viridis", "white").
        n_cols: number of columns in the grid.
        figsize: figure size passed to matplotlib.
    """
    n_images = len(images)
    titles = titles or [f"Image {i}" for i in range(n_images)]
    cmaps = cmaps or ["gray"] * n_images

    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        img_np = np.squeeze(img.cpu().numpy() if hasattr(img, "cpu") else img)

        if cmap == "glasbey":
            cmap_obj = get_glasbey_cmap()
            vmax = max(int(img_np.max()), 1)
            im = ax.imshow(img_np, cmap=cmap_obj, interpolation="nearest", vmin=0, vmax=vmax)
        elif cmap == "semantic":
            im = ax.imshow(img_np, cmap=get_semantic_cmap(), vmin=0, vmax=max(int(img_np.max()), 1))
        elif cmap == "sdt" or (isinstance(cmap, tuple) and cmap[0] == "sdt"):
            base = cmap[1] if isinstance(cmap, tuple) and len(cmap) > 1 else "plasma"
            bg   = cmap[2] if isinstance(cmap, tuple) and len(cmap) > 2 else "black"
            clipped = np.clip(img_np, 0, None)
            img_norm = clipped / (clipped.max() + 1e-8)
            im = ax.imshow(img_norm, cmap=get_sdt_cmap(base=base, background=bg), vmin=0, vmax=1)
        else:
            im = ax.imshow(img_np, cmap=cmap)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[n_images:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "prediction_example_plot.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def loss_plot_fn(train_loss, val_loss, output_dir="."):
    print("\n----------------------------------------------------------------------------")
    print("\nTraining and validation loss")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(train_loss) + 1), train_loss, label="Training loss")
    ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation loss")

    ax.set_title('Training and validation loss vs epoch number (linear)')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch number")
    ax.set_xticks(np.linspace(1, len(val_loss), 5).astype(int))
    ax.legend()

    fig.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_plot.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def loss_plot_log_fn(train_loss, val_loss, output_dir="."):
    print("\n----------------------------------------------------------------------------")
    print("\nTraining and validation loss (log)")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(train_loss) + 1), train_loss, label="Training loss")
    ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation loss")

    ax.set_yscale("log")
    ax.set_title("Training and validation loss vs epoch number (log)")
    ax.set_ylabel("Log Loss")
    ax.set_xlabel("Epoch number")
    ax.set_xticks(np.linspace(1, len(val_loss), 5).astype(int))
    ax.legend()

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_plot_log.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_segmentation_scores_fn(f1_fibre, f1_axon, f1_inner_cylinder, balanced_segmentation_score, output_dir="."):
    print("\n----------------------------------------------------------------------------")
    print("Segmentation metric trends over epochs")

    epochs = np.arange(1, len(f1_fibre) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, f1_fibre, label="F1 Fibre")
    ax.plot(epochs, f1_axon, label="F1 Axon")
    ax.plot(epochs, f1_inner_cylinder, label="F1 Inner Cylinder")
    ax.plot(epochs, balanced_segmentation_score, label="Balanced Segmentation Score")

    best_epoch = np.argmax(balanced_segmentation_score) + 1
    best_score = balanced_segmentation_score[best_epoch - 1]

    ax.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax.text(best_epoch + 0.5, 0.95, f'{best_score:.3f}', color='red')

    ax.set_title("Segmentation Scores per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.linspace(1, len(f1_fibre), 5).astype(int))
    ax.grid(True)
    ax.legend()

    fig.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "segmentation_scores_plot.pdf"), bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_iou_distributions(df, label, output_dir="."):
    """
    Plot mean F1 with confidence interval.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Image_Name', 'Threshold', 'F1'].
        label (str): Type of label (e.g., "Fibre" or "Axon").
        output_dir (str): Directory to save the plot PDF.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute mean and confidence interval
    grouped = df.groupby("Threshold")["F1"]
    mean_f1 = grouped.mean().astype(float)
    std_f1 = grouped.std().astype(float)
    n = df["Image_Name"].nunique()
    ci95 = 1.96 * std_f1 / np.sqrt(n)  # 95% CI

    # Plot mean line
    ax.plot(mean_f1.index, mean_f1.values, label="Mean F1", color="black", linewidth=2)

    # Plot shaded confidence interval
    ax.fill_between(mean_f1.index, mean_f1 - ci95, mean_f1 + ci95, color="blue", alpha=0.3, label="95% CI")

    # Styling
    ax.set_title(f"{label} F1 Across IoU Thresholds")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{label.lower()}_iou_distribution.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
