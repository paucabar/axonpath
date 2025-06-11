import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc


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


def plot_iou_distributions(df, label):
    """
    Plot IoU/F1 distributions for each image.

    Args:
        df (pd.DataFrame): DataFrame with metrics from evaluator.
        label (str): Name of target type (e.g., Fibre or Axon).
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df,
        x="Threshold",
        y="F1",
        )
    plt.title(f"{label} F1 Scores Across IoU Thresholds")
    plt.grid(False)
    plt.show()