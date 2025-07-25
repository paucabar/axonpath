import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
from tqdm import tqdm
from aimsegdl.evaluation.segmentation_evaluator import SegmentationEvaluator
from aimsegdl.inference.inference import load_model, run_inference
from aimsegdl.dataset.aimseg_dataset import AimSegDataset
from aimsegdl.utils.visualization import plot_iou_distributions

def plot_segmentation_comparison(gt_fibre, gt_axon, pred_fibre, pred_axon, figsize=(10, 8), title=None):
    """
    Plots ground truth and predicted segmentation masks for fibre and axon.

    Args:
        gt_fibre (np.ndarray): Ground truth fibre mask.
        gt_axon (np.ndarray): Ground truth axon mask.
        pred_fibre (np.ndarray): Predicted fibre mask.
        pred_axon (np.ndarray): Predicted axon mask.
        figsize (tuple): Size of the figure.
        title (str): Optional overall title for the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=16)

    axes[0, 0].imshow(gt_fibre, cmap='nipy_spectral', interpolation="nearest")
    axes[0, 0].set_title("GT Fibre")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_axon, cmap='nipy_spectral', interpolation="nearest")
    axes[0, 1].set_title("GT Axon")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(pred_fibre, cmap='nipy_spectral', interpolation="nearest")
    axes[1, 0].set_title("Predicted Fibre")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_axon, cmap='nipy_spectral', interpolation="nearest")
    axes[1, 1].set_title("Predicted Axon")
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
    plt.show()

def compute_imagewise_means(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean metrics per image, then averages over all images.

    Args:
        results_df (pd.DataFrame): DataFrame with 'Image_Name', 'F1', 'Precision', 'Recall', 'Jaccard', etc.

    Returns:
        pd.Series: Series with overall mean of each metric.
    """
    per_image = results_df.groupby("Image_Name").mean(numeric_only=True)
    overall_mean = per_image[["F1", "Precision", "Recall", "Jaccard"]].mean()
    return overall_mean

def plot_summary_bar(overall_metrics: pd.Series, title_tag: str):
    """
    Plots a bar chart of the average F1, Precision, Recall, and Jaccard.

    Args:
        overall_metrics (pd.Series): Series with metric names as index and their average values.
        title_tag (str): Target identifier, e.g., fibre, axon... (for title only)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(overall_metrics.index, overall_metrics.values, color=["steelblue", "seagreen", "goldenrod", "mediumpurple"])
    
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{title_tag} Average Segmentation Metrics")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.show()

def evaluate_model_on_testset(model_path, test_dir, device, output_csv=None, display_figure=False, show_plots=True):
    # Load model
    model = load_model(model_path, device=device)

    # Load test set
    dataset = AimSegDataset(test_dir)

    # Init result DataFrames with proper columns
    columns = ["Image_Name", "Threshold", "F1", "Precision", "Recall", "Jaccard", "TP", "FP", "FN"]
    fibre_results = pd.DataFrame(columns=columns)
    axon_results = pd.DataFrame(columns=columns)

    for path in tqdm(dataset.tile_paths, desc="Evaluating tiles"):
        tile_name = os.path.splitext(os.path.basename(path))[0]
        sample = dataset.__getitem__(dataset.tile_paths.index(path))
        image, masks = sample

        pred_fibre, pred_axon, pred_semantic = run_inference(image.numpy(), model, device)

        gt_fibre = masks[0].numpy()
        gt_axon = masks[1].numpy()

        fibre_eval = SegmentationEvaluator(gt_fibre, pred_fibre)
        axon_eval = SegmentationEvaluator(gt_axon, pred_axon)

        fibre_results = fibre_eval.evaluate_multiple_thresholds(tile_name, fibre_results)
        axon_results = axon_eval.evaluate_multiple_thresholds(tile_name, axon_results)

        if display_figure:
            # Filter rows by tile name
            fibre_f1_filtered = fibre_results["F1"][fibre_results["Image_Name"] == tile_name]
            axon_f1_filtered = axon_results["F1"][axon_results["Image_Name"] == tile_name]

            # Compute means
            mean_f1_fibre = fibre_f1_filtered.mean()
            mean_f1_axon = axon_f1_filtered.mean()

            # Create title
            title = f"{tile_name} | Mean F1 Fibre: {mean_f1_fibre:.3f} | Mean F1 Axon: {mean_f1_axon:.3f}"

            # Plot
            plot_segmentation_comparison(gt_fibre, gt_axon, pred_fibre, pred_axon, figsize=(10, 8), title=title)


    if output_csv:
        base = os.path.splitext(output_csv)[0]
        fibre_path = base + "_Fibre.tsv"
        axon_path = base + "_Axon.tsv"

        with open(fibre_path, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(fibre_results.columns)
            writer.writerows(fibre_results.values)

        with open(axon_path, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(axon_results.columns)
            writer.writerows(axon_results.values)

        print(f"\nSaved results to:\n- {fibre_path}\n- {axon_path}")

    if show_plots:
        plot_iou_distributions(fibre_results, label="Fibre")
        plot_iou_distributions(axon_results, label="Axon")

    return fibre_results, axon_results
