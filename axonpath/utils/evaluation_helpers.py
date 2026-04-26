import torch
import torch.nn as nn
import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects
from monai.metrics import DiceMetric
from axonpath.utils.visualization import show_images
from axonpath.utils.image_processing import (
    apply_semantic_segmentation_head,
    segment_instances_from_sdt,
    fill_labels,
    map_axon_labels_to_fibres,
)
from axonpath.evaluation.segmentation_evaluator import SegmentationEvaluator
from axonpath.utils.losses import compute_loss


def evaluate(
    loader,
    model,
    loss_fn,
    device="cuda",
    fibre_threshold: float = 0.5,
    axon_threshold: float = 0.5,
    min_diameter: float = 30.0,
    show_results: bool = False,
    output_dir: str = ".",
    loss_weights: tuple = (1.0, 1.0, 1.0),
):
    """
    Evaluate the model on the given loader with F1, Dice and loss metrics.

    Returns:
        Tuple: (mean_val_loss, mean_f1_fibre, mean_f1_axon, mean_f1_inner_cylinder, dice_score)
    """
    model.eval()
    val_losses = []
    f1_scores_fibre, f1_scores_axon, f1_scores_inner_cylinder = [], [], []
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, num_classes=3)
    axon_min_diameter = min_diameter / 2

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = torch.tensor(np.stack(y, axis=1)).to(device)

            with torch.amp.autocast(device_type=device):
                prediction = model(x)

            val_losses.append(compute_loss(prediction.float(), y, loss_fn, loss_weights).item())

            # Dice
            sem_pred = apply_semantic_segmentation_head(prediction[:, 0:3])
            y_sem = y[:, 2]
            y_onehot = nn.functional.one_hot(y_sem.long(), num_classes=3).permute(0, 3, 1, 2).float()
            pred_onehot = nn.functional.one_hot(sem_pred.long(), num_classes=3).permute(0, 3, 1, 2).float()
            dice_metric(y_pred=pred_onehot, y=y_onehot)

            # F1 per instance
            for i in range(x.shape[0]):
                scores = evaluate_instance_metrics(
                    prediction[i],
                    y[i],
                    fibre_threshold,
                    axon_threshold,
                    min_diameter,
                    axon_min_diameter,
                    i
                )
                f1_scores_fibre.append(scores[0])
                f1_scores_axon.append(scores[1])
                f1_scores_inner_cylinder.append(scores[2])

        dice_score = dice_metric.aggregate().item()
        dice_metric.reset()

        if show_results:
            plot_example(x[0], y[0], prediction[0], fibre_threshold, axon_threshold, min_diameter, axon_min_diameter, output_dir=output_dir)

    model.train()

    return (
        np.mean(val_losses),
        np.mean(f1_scores_fibre) if f1_scores_fibre else 0.0,
        np.mean(f1_scores_axon) if f1_scores_axon else 0.0,
        np.mean(f1_scores_inner_cylinder) if f1_scores_inner_cylinder else 0.0,
        dice_score
    )


def evaluate_instance_metrics(pred, target, fibre_threshold, axon_threshold, min_diameter, axon_min_diameter, index):
    # Predict instances
    pred_fibre = segment_instances_from_sdt(pred[3].unsqueeze(0), fibre_threshold, min_diameter)
    pred_axon = segment_instances_from_sdt(pred[4].unsqueeze(0), axon_threshold, axon_min_diameter)

    pred_sem = apply_semantic_segmentation_head(pred[0:3].unsqueeze(0))

    # Inner Cylinder postprocessing aligned with test eval and extension:
    # fill holes → remove small objects → map to fibres
    min_diameter_inner_cylinder = min_diameter / 2
    radius = min_diameter_inner_cylinder / 2
    min_area = int(np.pi * radius ** 2)

    pred_inner_cylinder = label((pred_sem.cpu().numpy().squeeze() == 2).astype(np.int32), connectivity=2)
    pred_inner_cylinder = fill_labels(pred_inner_cylinder)
    pred_inner_cylinder = label(
        remove_small_objects(pred_inner_cylinder > 0, max_size=max(0, min_area - 1), connectivity=2),
        connectivity=2
    )
    pred_inner_cylinder = map_axon_labels_to_fibres(pred_fibre, pred_inner_cylinder)

    # Ground truth
    gt_fibre = target[0].cpu().numpy().astype(np.int32)
    gt_axon = target[1].cpu().numpy().astype(np.int32)
    gt_sem = target[2].cpu().numpy().astype(np.int32)
    gt_inner_cylinder = label((gt_sem == 2).astype(np.int32), connectivity=2)

    fibre_eval = SegmentationEvaluator(gt_fibre, pred_fibre)
    f1_fibre = fibre_eval.f1_mean(fibre_eval.evaluate_multiple_thresholds(f"sample_{index}_fibre"))

    axon_eval = SegmentationEvaluator(gt_axon, pred_axon)
    f1_axon = axon_eval.f1_mean(axon_eval.evaluate_multiple_thresholds(f"sample_{index}_axon"))

    inner_cylinder_eval = SegmentationEvaluator(gt_inner_cylinder, pred_inner_cylinder)
    f1_inner_cylinder = inner_cylinder_eval.f1_mean(inner_cylinder_eval.evaluate_multiple_thresholds(f"sample_{index}_inner_cylinder"))

    return f1_fibre, f1_axon, f1_inner_cylinder


def plot_example(x, y, pred, fibre_threshold, axon_threshold, min_diameter, axon_min_diameter, output_dir="."):
    sem = apply_semantic_segmentation_head(pred[0:3].unsqueeze(0))
    labels_fibre = segment_instances_from_sdt(pred[3].unsqueeze(0), fibre_threshold, min_diameter)
    labels_axon = segment_instances_from_sdt(pred[4].unsqueeze(0), axon_threshold, axon_min_diameter)

    show_images(
        x.cpu(),
        y[0].cpu(), y[1].cpu(), y[2].cpu(),
        pred[3].cpu(), pred[4].cpu(),
        sem, labels_fibre, labels_axon,
        titles=[
            "Image",  "Target Fibre", "Target Axon",
            "Target Semantic", "Pred Fibre SDT", "Pred Axon SDT",
            "Prediction Semantic", "Prediction Fibre Instance", "Prediction Axon Instance"
        ],
        cmaps=[
            "gray", "glasbey", "glasbey",
            "semantic", "sdt", "sdt",
            "semantic", "glasbey", "glasbey"
        ],
        n_cols=3,
        output_dir=output_dir,
    )
