import torch
import torch.nn as nn
import numpy as np
from monai.metrics import DiceMetric
from utils.visualization import show_images
from utils.image_processing import last_layer_fn, postprocessing_sdt
from evaluation.segmentation_evaluator import SegmentationEvaluator


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