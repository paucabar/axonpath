# Modified from Caicedo et al 2019


import numpy as np
import pandas as pd
from skimage.measure import regionprops
import math


# Calculates object-based intersection over union (IoU) comparing a ground truth with a prediction image
# Intersection (I) is calculated as the elements (pixels) belonging to a target and a prediction (objects), i.e., the overlapping area
# Union (U) is calculated as the sum of the pixels that belong to either the target, the prediction or both
# Both input images must be label masks

def intersection_over_union(ground_truth, prediction):
    
    assert ground_truth.shape == prediction.shape
    ground_truth_collapsed = ground_truth.flatten() # returns a copy of the array collapsed into one dimension
    prediction_collapsed = prediction.flatten()
    assert len(ground_truth_collapsed) == len(prediction_collapsed)
    
    # Counts true (target) and predicted objects
    # NOTE that the count includes the background (label 0) as an object
    true_objects_count = len(np.unique(ground_truth)) # length of list of unique labels (objects)
    pred_objects_count = len(np.unique(prediction))
    
    # Computes a bi-dimensional histogram (returns H == per object intersection)
    # x coordinates = sum of unique prediction pixel values; y coordinates = sum of unique ground truth pixel values
    intersection, _xedges_, _yedges_ = np.histogram2d(ground_truth_collapsed, prediction_collapsed, bins=(true_objects_count, pred_objects_count))
    
    # Computes the area of each object (sums pixels with the same value) as a list
    area_true, _bin_edges_ = np.histogram(ground_truth, bins=true_objects_count)
    area_pred, _bin_edges_ = np.histogram(prediction, bins=pred_objects_count)
    
    # To operate with the intersection array (bi-dimensional), it is necessary to expand the dimensions of the area lists (unidimensional)
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Computes the union as the sum of the target and the prediction area minus the intersection (overlapping) area
    union = area_true + area_pred - intersection
    
    # Excludes background (pixels set as 0) from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9 # ensures that the scrip wil not crash computing the IoU if union == 0

    # Compute IoU
    IoU = intersection / union
    
    return IoU


# Uses the IoU array generated with the intersection_over_unio method to compute the count of
# true positives (TP), false positives (FP) and false negatives (FN). It requires to set an IoU
# threshold to determine when a pair of objects is considered a match.
# As a rule of thumb, threshold >= 0.5 to avoid multiple matches for one object

def evaluate_at_iou(threshold, IoU_array):
    
    # makes IoU array binary (1 == match; 0 == not match)
    matches = IoU_array > threshold
    
    # For matches array, axis 1 == y (target objects), axis 0 == x (predicted objects)
    true_positives = np.sum(matches, axis=1) == 1 # Correct objects: 1 match in a row means a target matches a prediction
    false_positives = np.sum(matches, axis=0) == 0 # Extra objects: 0 matches in a column means a prediction missing a matching target
    false_negatives = np.sum(matches, axis=1) == 0 # Missed objects: 0 matches in a row means a target missing a matching prediction
    
    # checks that all the elements in the arrays are <= 1
    assert np.all(np.less_equal(true_positives, 1)) # len(true_positives) == n rows in matches array
    assert np.all(np.less_equal(false_positives, 1)) # len(false_positives) == n columns in matches array
    assert np.all(np.less_equal(false_negatives, 1)) # len(false_negatives) == n rows in matches array
    
    # Calculates TP, FP and FN, which are then used to calculate the segmentation metrics (F1 score, precision and recall)
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    
    return f1, precision, recall, TP, FP, FN


# Compute segmentation metrics for a range of IoU thresholds
# Uses the intersection_over_union function the generate the IoU array and then applies the evaluate_at_iou function
# within a loop to assess the segmentation results setting different IoU thresholds.
# The method requires a pandas' DataFrame as input, where results will be stored adding rows.
# NOTE: The DataFrame can be either just a heading row or a bigger, pre-filled table.
# Returns a DataFrame containing the old and the new results

def multi_threshold_results(ground_truth, prediction, results, image_name):

    # Computes IoU
    IoU = intersection_over_union(ground_truth, prediction)
    
    # Calculates the average IoU (also known as Jaccard index)
    # For the sake of clarity, we will use only the term Jaccard index when calculating the mean of all the target objects
    # in order to provide a global metric for a whole image. Instead, previously we have used the term IoU on per object methods
    if IoU.shape[0] > 0:                        # if there are predictions:
        jaccard = np.max(IoU, axis=0).mean()    # Jaccard = average of the max IoU value for each target (axis=0 == x (predicted objects) == columns)
    else:                                       # else:
        jaccard = 0.0                           # NOTE: Jaccard = 0, even if there are no targets
    
    # Calculates segmentation metrics for all thresholds from 0.5 to 0.95 within a 0.05 interval 
    for t in np.arange(0.5, 0.95, 0.05):
        
        # Computes metrics at specific IoU
        f1, precision, recall, tp, fp, fn = evaluate_at_iou(t, IoU)
        
        # Stores results in a dictionary to add them at the bottom of the DataFrame
        res = {"Image_Name": image_name, "Threshold": t, "F1": f1, "Precision": precision, "Recall": recall, "Jaccard": jaccard, "TP": tp, "FP": fp, "FN": fn}
        row = len(results)
        results.loc[row] = res
        
    return results


# Identify False Negatives at specific IoU (default = 0.7) and extract features (shape descriptors)
# The results table enables to check how different features have an effect on segmentation accuracy
# The method requires a pandas' DataFrame as input, where results will be stored adding rows.
# NOTE: The DataFrame can be either just a heading row or a bigger, pre-filled table.
# Returns a DataFrame containing the old and the new results

def false_negative_features(ground_truth, prediction, results, threshold=0.7):

    # Compute IoU
    IoU = intersection_over_union(ground_truth, prediction)
    
    # Count the number of objects in ground truth
    true_objects = len(np.unique(ground_truth)) # NOTE that the count includes the background (label 0) as an object
    if true_objects <= 1:
        return results

    # Define additional features (not included in regionprops)
    def circularity(p):
        circ = 4 * math.pi * p['area'] / p['perimeter_crofton'] ** 2
        if circ > 1.0: circ = 1.0
        return circ
    def aspect_ratio(p):
        return p['axis_major_length'] / (p['axis_minor_length'] + 1e-9)
    
    # Get features of target objects as numpy arrays
    props = regionprops(ground_truth)
    area_true = np.asarray([p['area'] for p in props])
    solidity_true = np.asarray([p['solidity'] for p in props])
    circularity_true = np.asarray([circularity(p) for p in props])
    aspect_ratio_true = np.asarray([aspect_ratio(p) for p in props])
    
    # Identify False Negatives
    matches = IoU > threshold # makes IoU array binary (1 == match; 0 == not match)
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects: 0 matches in a row means a target missing a matching prediction

    # Prepare data to fill DataFrame 
    data = np.asarray([
        area_true.copy(), 
        solidity_true.copy(),
        circularity_true.copy(),
        aspect_ratio_true.copy(),
        np.array(false_negatives, dtype=np.int32)
    ])

    # Concatenate input DataFrame (results) and new data
    results = pd.concat([results, pd.DataFrame(data=data.T, columns=["Area", "Solidity", "Circularity", "Aspect_Ratio", "False_Negative"])], sort=False)
        
    return results


# Count the number of splits and merges based on the IoU
# Setting a low IoU threshold (< 0.5) enables objects (either target or predictions) to get multiple matches.
# A target matching multiple predictions is a split, whereas a prediction matching multiple targets is a merge.
# The method requires a pandas' DataFrame as input, where results will be stored adding rows.
# NOTE: The DataFrame can be either just a heading row or a bigger, pre-filled table.
# Returns a DataFrame containing the old and the new results

def split_merge_iou(ground_truth, prediction, results, image_name):

    # Compute IoU
    IoU = intersection_over_union(ground_truth, prediction)
    
    matches = IoU > 0.1 # Set a low IoU threshold
    # For matches array, axis 1 == y (target objects), axis 0 == x (predicted objects)
    merges = np.sum(matches, axis=0) > 1 # Merge = more than 1 match in a column, i.e., 1 prediction matches > 1 target
    splits = np.sum(matches, axis=1) > 1 # Split = more than 1 match in a row, i.e., 1 target matches > 1 prediction
    
    # Stores results in a dictionary to add them at the bottom of the DataFrame
    r = {"Image_Name":image_name, "Merges":np.sum(merges), "Splits":np.sum(splits)}
    results.loc[len(results)+1] = r
    
    return results


# Calculates object-based intersection over prediction (IoP) and intersection over target (IoT) comparing a ground truth with a prediction image
# Intersection (I) is calculated as the elements (pixels) belonging to a target and a prediction (objects), i.e., the overlapping area
# Both input images must be label masks

def iop_and_iot(ground_truth, prediction):
    
    # Counts true (target) and predicted objects
    # NOTE that the count includes the background (label 0) as an object
    true_objects_count = len(np.unique(ground_truth))# length of list of unique labels (objects)
    pred_objects_count = len(np.unique(prediction))
    
    # Computes a bi-dimensional histogram (returns H == per object intersection)
    # x coordinates = sum of unique prediction pixel values; y coordinates = sum of unique ground truth pixel values
    intersection, _xedges_, _yedges_ = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects_count,pred_objects_count))
    
    # Computes the area of each predicted  and target object (sums pixels with the same value) as a list
    area_pred, _bin_edges_ = np.histogram(prediction, bins=pred_objects_count)
    area_targ, _bin_edges_ = np.histogram(ground_truth, bins=true_objects_count)

    # To operate with the intersection array (bi-dimensional), it is necessary to expand the dimensions of the area lists (unidimensional)
    #area_pred = np.expand_dims(area_pred, 0)
    #area_targ = np.expand_dims(area_targ, 0)

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    area_pred = area_pred[1:]
    area_targ = area_targ[1:]

    # Transpose to make area target list vertical
    area_targ = np.array(area_targ).reshape(-1,1)

    # Compute IoP and IoT
    IoP = intersection / area_pred
    IoT = intersection / area_targ

    return IoP, IoT


# Count the number of splits based on IoP and merges based on the IoT
# These metrics are focused on intersection (overlap) and the specific object to be assesses.
# Thus, setting an intermediate value for the threshold still can generate multiple matches for one object.
# A target matching multiple predictions is a split, whereas a prediction matching multiple targets is a merge.
# The method requires a pandas' DataFrame as input, where results will be stored adding rows.
# NOTE: The DataFrame can be either just a heading row or a bigger, pre-filled table.
# Returns a DataFrame containing the old and the new results

def split_merge_iopiot(ground_truth, prediction, results, image_name):

    # Compute IoP and IoT
    IoP, IoT = iop_and_iot(ground_truth, prediction)
    
    # Set a intermediate thresholds
    matches_prediction = IoP > 0.2
    matches_target = IoT > 0.2
    # For matches array, axis 1 == y (target objects), axis 0 == x (predicted objects)
    splits = np.sum(matches_prediction, axis=1) > 1 # Merge = more than 1 match in a column, i.e., 1 prediction matches > 1 target
    merges = np.sum(matches_target, axis=0) > 1 # Split = more than 1 match in a row, i.e., 1 target matches > 1 prediction

    # Stores results in a dictionary to add them at the bottom of the DataFrame
    r = {"Image_Name":image_name, "Merges":np.sum(merges), "Splits":np.sum(splits)}
    results.loc[len(results)+1] = r
    
    return results


def targets_and_predictions(ground_truth, prediction, results, image_name):
        target_count = len(np.unique(ground_truth))
        prediction_count = len(np.unique(prediction))
        r = {"Image_Name":image_name, "Targets":np.sum(target_count), "Predictions":np.sum(prediction_count)}
        results.loc[len(results)+1] = r
        return results