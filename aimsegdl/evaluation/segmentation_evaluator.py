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
        
        self.ground_truth = label(ground_truth)
        self.prediction = label(prediction)
        self.iou_matrix = self._compute_iou_matrix()

    def _compute_iou_matrix(self) -> np.ndarray:
        """
        Computes pairwise IoU between ground truth and predicted objects.
        
        Returns:
            np.ndarray: 2D array with IoU values between ground truth and predicted objects.
        """
        gt_flat = self.ground_truth.flatten()
        pred_flat = self.prediction.flatten()
        
        true_labels = np.unique(self.ground_truth)
        pred_labels = np.unique(self.prediction)
        
        n_gt = len(true_labels)
        n_pred = len(pred_labels)

        intersection, _, _ = np.histogram2d(gt_flat, pred_flat, bins=(n_gt, n_pred))
        area_true, _ = np.histogram(self.ground_truth, bins=n_gt)
        area_pred, _ = np.histogram(self.prediction, bins=n_pred)

        area_true = np.expand_dims(area_true, axis=-1)
        area_pred = np.expand_dims(area_pred, axis=0)
        union = area_true + area_pred - intersection

        # Exclude background (assumed label 0)
        intersection = intersection[1:, 1:]
        union = union[1:, 1:]
        union[union == 0] = 1e-9  # Avoid division by zero

        return intersection / union

    def _evaluate_at_threshold(self, threshold: float) -> Tuple[float, float, float, int, int, int]:
        """
        Computes evaluation metrics at a given IoU threshold.

        Args:
            threshold (float): IoU threshold for matching.

        Returns:
            Tuple: (F1, Precision, Recall, TP, FP, FN)
        """
        matches = self.iou_matrix > threshold
        true_positives = np.sum(matches, axis=1) == 1
        false_positives = np.sum(matches, axis=0) == 0
        false_negatives = np.sum(matches, axis=1) == 0

        TP = np.sum(true_positives)
        FP = np.sum(false_positives)
        FN = np.sum(false_negatives)

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
            results_df = pd.DataFrame(columns=["Image_Name", "Threshold", "F1", "Precision", "Recall", "Jaccard", "TP", "FP", "FN"])

        jaccard = np.max(self.iou_matrix, axis=0).mean() if self.iou_matrix.size > 0 else 0.0

        for threshold in np.arange(0.5, 1.0, 0.05):
            f1, precision, recall, TP, FP, FN = self._evaluate_at_threshold(threshold)
            results_df.loc[len(results_df)] = {
                "Image_Name": image_name,
                "Threshold": threshold,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Jaccard": jaccard,
                "TP": TP,
                "FP": FP,
                "FN": FN
            }

        return results_df

    @staticmethod
    def f1_mean(results_df: pd.DataFrame) -> float:
        """
        Computes mean F1 score from a results dataframe.

        Args:
            results_df (pd.DataFrame): Dataframe with an 'F1' column.

        Returns:
            float: Mean F1 score.
        """
        
        mean_per_image = results_df["F1"].groupby("Image_Name").mean(numeric_only=True)
        
        return mean_per_image.mean()