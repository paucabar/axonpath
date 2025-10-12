import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc

mpl.use("TkAgg")

def get_glasbey_cmap():
    l = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
    l[0] = [0, 0, 0]
    return LinearSegmentedColormap.from_list('glasbey', l, N=2048)

def apply_cmap(image, cmap):
    if cmap == "glasbey":
        glasbey = get_glasbey_cmap()
        rgba = glasbey(image / np.max(image))
        return np.uint8(rgba[:, :, :3] * 255), None  # RGB image, no colorbar
    else:
        return image, cmap  # Return as-is for matplotlib to handle


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
            norm = mpl.colors.Normalize(vmin=0, vmax=cmap_obj.N - 1)
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(img_np, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title)
        ax.axis("off")

    for ax in axes[n_images:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("prediction_example_plot.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    #plt.show(block=False)
    #plt.pause(0.01)


def loss_plot_fn(train_loss, val_loss):
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
    fig.savefig('loss_plot.pdf', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(0.01)


def loss_plot_log_fn(train_loss, val_loss):
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
    fig.savefig("loss_plot_log.pdf", bbox_inches="tight", dpi=300)
    plt.show(block=False)
    plt.pause(0.01)



def plot_segmentation_scores_fn(f1_fibre, f1_axon, f1_inner_tongue, balanced_segmentation_score):
    print("\n----------------------------------------------------------------------------")
    print("Segmentation metric trends over epochs")

    epochs = np.arange(1, len(f1_fibre) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, f1_fibre, label="F1 Fibre")
    ax.plot(epochs, f1_axon, label="F1 Axon")
    ax.plot(epochs, f1_inner_tongue, label="F1 Inner Tongue")
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
    fig.savefig("segmentation_scores_plot.pdf", bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(0.01)


def plot_iou_distributions(df, label):
    """
    Plot mean F1 with confidence interval.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Image_Name', 'Threshold', 'F1'].
        label (str): Type of label (e.g., "Fibre" or "Axon").
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute mean and confidence interval
    grouped = df.groupby("Threshold")["F1"]
    mean_f1 = grouped.mean()
    std_f1 = grouped.std()
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
    plt.savefig(f"{label.lower()}_iou_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.01)


def plot_iou_distributions_imagewise(df, label):
    """
    Plot IoU/F1 distributions for each image using matplotlib (no seaborn).

    Args:
        df (pd.DataFrame): DataFrame with columns ['Image_Name', 'Threshold', 'F1'].
        label (str): Type of label (e.g., "Fibre" or "Axon").
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Group by image name and plot each line
    grouped = df.groupby("Image_Name")
    for name, group in grouped:
        ax.plot(group["Threshold"], group["F1"], alpha=0.4, linewidth=1)

    # Plot mean F1 line across images at each threshold
    mean_f1 = df.groupby("Threshold")["F1"].mean()
    ax.plot(mean_f1.index, mean_f1.values, label="Mean F1", color="black", linewidth=2)

    ax.set_title(f"{label} F1 Scores Across IoU Thresholds")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.savefig(f"{label.lower()}_iou_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.01)