import os
import csv
import pandas as pd
from tqdm import tqdm
from aimsegdl.utils.image_processing import segment_instances_from_sdt
from aimsegdl.evaluation.segmentation_evaluator import SegmentationEvaluator
from aimsegdl.inference.inference import load_model, run_inference
from aimsegdl.dataset.aimseg_dataset import AimSegDataset
from aimsegdl.utils.visualization import plot_iou_distributions


def evaluate_model_on_testset(model_path, test_dir, device, output_csv=None, show_plots=True):
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
