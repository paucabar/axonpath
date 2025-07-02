import torch
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
import numpy as np
import torch.nn.functional as F


def normalize(image: np.ndarray, low_perc: float = 1, high_perc: float = 99) -> np.ndarray:
    """
    Normalize an image based on percentile clipping.

    Parameters:
        image (np.ndarray): Input image array.
        low_perc (float): Lower percentile for clipping (e.g., 1 for 1st percentile).
        high_perc (float): Upper percentile for clipping (e.g., 99 for 99th percentile).

    Returns:
        np.ndarray: Image normalized to [0, 1] after clipping.
    """
    # Compute intensity bounds from percentiles
    lower = np.percentile(image, low_perc)
    upper = np.percentile(image, high_perc)

    # Clip values to the computed range
    image_clipped = np.clip(image, lower, upper)

    # Normalize to [0, 1], safely handling zero division
    if upper > lower:
        normalized = (image_clipped - lower) / (upper - lower)
    else:
        normalized = np.zeros_like(image, dtype=np.float32)

    return normalized


def fill_border_holes(mask: np.ndarray) -> np.ndarray:
    """
    Python adaptation of the 'Fill_Border_Holes' ImageJ macro by G. Landini.
    Fills holes in binary masks, including those that touch at most two borders 
    (i.e., corners or edges), mimicking the macro's logic using edge manipulation.

    Original macro by G. Landini:
    https://sites.imagej.net/Landini/plugins/Morphology/Fill_Border_Holes.ijm-20140627120652

    Args:
        mask (np.ndarray): Binary mask (2D boolean or 0/1 array).

    Returns:
        np.ndarray: Mask with internal and edge-connected holes filled.
    """
    if mask.ndim != 2:
        raise ValueError("Only 2D arrays supported.")

    h, w = mask.shape
    padded = np.pad(mask.astype(bool), 1, mode='constant', constant_values=0)
    filled = padded.copy()

    # Step 1: Right and top borders set to 1
    filled[0, :] = True           # top row
    filled[:, -1] = True          # right column
    filled = binary_fill_holes(filled)

    # Step 2: Reset top to 0, bottom to 1
    filled[0, :] = False
    filled[-1, :] = True
    filled = binary_fill_holes(filled)

    # Step 3: Reset right to 0, left to 1
    filled[:, -1] = False
    filled[:, 0] = True
    filled = binary_fill_holes(filled)

    # Step 4: Reset top to 1, bottom to 0
    filled[0, :] = True
    filled[-1, :] = False
    filled = binary_fill_holes(filled)

    # Step 5: Remove padding
    result = filled[1:-1, 1:-1]

    return result.astype(bool)


def fill_labels(label_image: np.ndarray) -> np.ndarray:
    """
    Fill holes in each labeled region, including edge-touching enclosed holes,
    using bounding box cropping for efficiency.

    Args:
        label_image (np.ndarray): 2D labeled image with integer labels.

    Returns:
        np.ndarray: Labeled image with holes filled per instance.
    """
    if label_image.ndim != 2:
        raise ValueError("label_image must be a 2D array.")
    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError("label_image must contain integer labels.")

    filled_image = np.copy(label_image)
    height, width = label_image.shape

    regions = regionprops(label_image.astype(np.int32))

    for region in regions:
        label_id = region.label
        min_row, min_col, max_row, max_col = region.bbox

        # Add 1-pixel padding, but stay within image bounds
        pad_top = 1 if min_row > 0 else 0
        pad_bottom = 1 if max_row < height else 0
        pad_left = 1 if min_col > 0 else 0
        pad_right = 1 if max_col < width else 0

        pad_min_row = min_row - pad_top
        pad_max_row = max_row + pad_bottom
        pad_min_col = min_col - pad_left
        pad_max_col = max_col + pad_right

        cropped_label = label_image[pad_min_row:pad_max_row, pad_min_col:pad_max_col]
        region_mask = (cropped_label == label_id)

        # Fill internal and edge-touching holes
        filled_region_mask = fill_border_holes(region_mask)

        # Assign newly filled pixels back to output image
        filled_image[pad_min_row:pad_max_row, pad_min_col:pad_max_col][filled_region_mask] = label_id


    return filled_image


def apply_semantic_segmentation_head(pred: torch.Tensor):
    """Applies softmax and argmax to semantic predictions (for training/evaluation)."""
    m = torch.nn.Softmax(dim=1)
    return torch.argmax(m(pred), dim=1)


def apply_semantic_segmentation_head_scriptable(pred: torch.Tensor):
    """TorchScript-compatible version for semantic predictions."""
    return torch.argmax(F.softmax(pred, dim=1), dim=1)


def segment_instances_from_sdt(
    distancemap: torch.Tensor,
    threshold: float = 0.5,
    min_diameter: float = 15.0,
    compactness: float = 0.5,
    valid_mask: np.ndarray = None
) -> np.ndarray:
    """
    Segment instance regions (e.g., fibres or axons) from a skeleton-aware distance transform.

    Parameters:
        distancemap (torch.Tensor): Predicted distance map, shape (1, H, W) or (H, W).
        threshold (float): Threshold to define seed regions for watershed.
        min_diameter (float): Expected minimum object diameter (used to derive min_size for seeds).
        compactness (float): Compactness factor for the watershed algorithm.
        valid_mask (np.ndarray, optional): Optional binary mask specifying where to restrict watershed.

    Returns:
        np.ndarray: Postprocessed label image.
    """
    # Convert to NumPy
    if distancemap.ndim == 3:
        distancemap_np = distancemap[0].detach().cpu().numpy()
    elif distancemap.ndim == 2:
        distancemap_np = distancemap.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected distancemap shape: {distancemap.shape}")


    # Determine valid mask BEFORE clipping
    if valid_mask is None or not isinstance(valid_mask, np.ndarray):
        valid_mask = distancemap_np >= 0


    # Clip the map to [0, 1]
    distancemap_clipped = np.clip(distancemap_np, 0, 1)

    # Estimate seed area from min_diameter (30% of diameter radius)
    radius = 0.3 * min_diameter / 2
    min_area = int(np.pi * radius ** 2)

    # Generate seed mask
    seed_mask = np.logical_and(distancemap_clipped >= threshold, valid_mask)
    seeds = label(seed_mask)
    seeds = remove_small_objects(seeds, min_size=min_area, connectivity=1)
    seeds = label(seeds)

    # Watershed
    labels = watershed(-distancemap_clipped, markers=seeds, mask=valid_mask, connectivity=1, compactness=compactness)

    # Fill holes in final labels
    return fill_labels(labels)


def map_axon_labels_to_fibres(label_img1: np.ndarray, label_img2: np.ndarray) -> np.ndarray:
    """
    Merge label_img2 fragments by assigning each to the label_img1 object it overlaps with most.

    Parameters:
        label_img1 (np.ndarray): Reference label image (e.g., fibres).
        label_img2 (np.ndarray): Fragmented label image (e.g., axons).

    Returns:
        np.ndarray: New label image where label_img2 fragments are grouped by their best label_img1 match.
    """
    label_img1 = label_img1.astype(np.int32)
    label_img2 = label_img2.astype(np.int32)

    flat1 = label_img1.ravel()
    flat2 = label_img2.ravel()

    max_label1 = label_img1.max()
    max_label2 = label_img2.max()

    overlap_matrix, _, _ = np.histogram2d(flat1, flat2, bins=(max_label1 + 1, max_label2 + 1))

    merged = np.zeros_like(label_img2, dtype=np.int32)

    for l2 in range(1, max_label2 + 1):
        overlaps = overlap_matrix[1:, l2]
        if overlaps.sum() == 0:
            continue
        best_l1 = np.argmax(overlaps) + 1
        merged[label_img2 == l2] = best_l1

    return merged
