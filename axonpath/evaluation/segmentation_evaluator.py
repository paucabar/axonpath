import numpy as np
from skimage.measure import label
import pandas as pd
from typing import Tuple


class SegmentationEvaluator:
    """
    Evaluates instance segmentation using object-wise IoU and F1 metrics.
    """

    def __init__(self, ground_truth: np.ndarray, prediction: np.ndarray):
        """
        Args:
            ground_truth (np.ndarray): Ground truth label image. Each instance should have a unique positive integer.
            prediction (np.ndarray): Predicted label image. Format same as ground_truth.
        """
        assert isinstance(ground_truth, np.ndarray), "ground_truth must be a numpy array"
        assert isinstance(prediction, np.ndarray), "prediction must be a numpy array"
        assert ground_truth.shape == prediction.shape, "Shape mismatch between ground truth and prediction"

        # connectivity=2 (8-connected) avoids splitting objects that are
        # 8-connected but not 4-connected, matching the extension's behaviour
        self.ground_truth = label(ground_truth, connectivity=2)
        self.prediction = label(prediction, connectivity=2)
        self.iou_matrix = self._compute_iou_matrix()

    def _compute_iou_matrix(self) -> np.ndarray:
        """
        Computes pairwise IoU between ground truth and predicted objects.

        Returns:
            np.ndarray: 2D array of shape (n_gt_objects, n_pred_objects).
        """
        gt_max = int(self.ground_truth.max())
        pred_max = int(self.prediction.max())

        if gt_max == 0 or pred_max == 0:
            return np.zeros((gt_max, pred_max), dtype=np.float64)

        gt_flat = self.ground_truth.flatten()
        pred_flat = self.prediction.flatten()

        # Integer-centred bin edges guarantee each label falls in exactly its own bin
        bins_gt = np.arange(-0.5, gt_max + 1.5)
        bins_pred = np.arange(-0.5, pred_max + 1.5)

        intersection, _, _ = np.histogram2d(gt_flat, pred_flat, bins=[bins_gt, bins_pred])
        area_true, _ = np.histogram(self.ground_truth, bins=bins_gt)
        area_pred, _ = np.histogram(self.prediction, bins=bins_pred)

        # Exclude background (label 0 → index 0)
        intersection = intersection[1:, 1:]
        area_true = np.expand_dims(area_true[1:], axis=-1)
        area_pred = np.expand_dims(area_pred[1:], axis=0)
        union = area_true + area_pred - intersection
        union[union == 0] = 1e-9

        return intersection / union

    def matches_at_threshold(self, threshold: float = 0.5) -> Tuple[dict, set, set]:
        """
        Greedy one-to-one matching between GT and predicted objects at a given IoU threshold.

        Args:
            threshold (float): IoU threshold for matching.

        Returns:
            Tuple:
                matches (dict): {gt_label: pred_label} for matched pairs (1-indexed label IDs).
                unmatched_gt (set): GT label IDs with no accepted match (FN).
                unmatched_pred (set): Predicted label IDs with no accepted match (FP).
        """
        n_gt, n_pred = self.iou_matrix.shape
        all_gt = set(range(1, n_gt + 1))
        all_pred = set(range(1, n_pred + 1))

        if n_gt == 0 or n_pred == 0:
            return {}, all_gt, all_pred

        matched_gt_idx = set()
        matched_pred_idx = set()
        matches = {}

        gt_indices, pred_indices = np.where(self.iou_matrix > threshold)
        if len(gt_indices) > 0:
            iou_values = self.iou_matrix[gt_indices, pred_indices]
            sort_order = np.argsort(-iou_values)
            for gi, pi in zip(gt_indices[sort_order], pred_indices[sort_order]):
                if gi not in matched_gt_idx and pi not in matched_pred_idx:
                    matched_gt_idx.add(gi)
                    matched_pred_idx.add(pi)
                    matches[gi + 1] = pi + 1  # back to 1-indexed label IDs

        unmatched_gt = all_gt - {gi + 1 for gi in matched_gt_idx}
        unmatched_pred = all_pred - {pi + 1 for pi in matched_pred_idx}

        return matches, unmatched_gt, unmatched_pred

    def _evaluate_at_threshold(self, threshold: float) -> Tuple[float, float, float, int, int, int]:
        """
        Computes evaluation metrics at a given IoU threshold using greedy one-to-one matching.

        Args:
            threshold (float): IoU threshold for matching.

        Returns:
            Tuple: (F1, Precision, Recall, TP, FP, FN)
        """
        n_gt, n_pred = self.iou_matrix.shape

        # Both empty: predicting nothing when there is nothing is perfect
        if n_gt == 0 and n_pred == 0:
            return 1.0, 1.0, 1.0, 0, 0, 0

        matches, unmatched_gt, unmatched_pred = self.matches_at_threshold(threshold)

        TP = len(matches)
        FP = len(unmatched_pred)
        FN = len(unmatched_gt)

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        f1 = 2 * TP / (2 * TP + FP + FN + 1e-9)

        return f1, precision, recall, TP, FP, FN

    def evaluate_multiple_thresholds(self, image_name: str, results_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Evaluates metrics at multiple IoU thresholds and appends results to a DataFrame.

        Args:
            image_name (str): Identifier for the evaluated image.
            results_df (pd.DataFrame): Existing results dataframe to append to.

        Returns:
            pd.DataFrame: Updated dataframe with results.
        """
        if results_df is None:
            results_df = pd.DataFrame(columns=["Image_Name", "Threshold", "F1", "Precision", "Recall", "TP", "FP", "FN"])

        for threshold in np.arange(0.5, 0.95, 0.05):
            f1, precision, recall, TP, FP, FN = self._evaluate_at_threshold(threshold)
            results_df.loc[len(results_df)] = {
                "Image_Name": image_name,
                "Threshold": threshold,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "TP": TP,
                "FP": FP,
                "FN": FN
            }

        return results_df

    @staticmethod
    def f1_mean(results_df: pd.DataFrame) -> float:
        """
        Computes mean F1 score across IoU thresholds from a per-sample results dataframe.

        Args:
            results_df (pd.DataFrame): Dataframe with an 'F1' column (one row per threshold).

        Returns:
            float: Mean F1 score across thresholds.
        """
        return results_df["F1"].mean()
