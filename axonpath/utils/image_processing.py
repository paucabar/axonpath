import torch
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import remove_small_objects
import numpy as np
import edt
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
    min_diameter: float = 30.0,
    compactness: float = 0.5,
    valid_mask: np.ndarray = None,
    seed_mask: np.ndarray = None
) -> np.ndarray:
    """
    Segment instance regions (e.g., fibres or axons) from a skeleton-aware distance transform.

    Parameters:
        distancemap (torch.Tensor): Predicted distance map, shape (1, H, W) or (H, W).
        threshold (float): Threshold to define seed regions for watershed.
        min_diameter (float): Expected minimum object diameter (used to derive min_size for seeds).
        compactness (float): Compactness factor for the watershed algorithm.
        valid_mask (np.ndarray, optional): Optional binary mask specifying where to restrict watershed.
        seed_mask (np.ndarray, optional): Optional binary mask specifying seeds.

    Returns:
        np.ndarray: Postprocessed label image.
    """
    # Convert to NumPy (cast to float32 first — autocast may produce bfloat16)
    if distancemap.ndim == 3:
        distancemap_np = distancemap[0].detach().cpu().float().numpy()
    elif distancemap.ndim == 2:
        distancemap_np = distancemap.detach().cpu().float().numpy()
    else:
        raise ValueError(f"Unexpected distancemap shape: {distancemap.shape}")


    # Determine valid mask
    if valid_mask is None or not isinstance(valid_mask, np.ndarray):
        valid_mask = distancemap_np >= 0

    # Clip the map to [0, 1]
    distancemap_clipped = np.clip(distancemap_np, 0, 1)

    # Estimate seed area from min_diameter (30% of diameter radius)
    radius = 0.3 * min_diameter / 2
    min_area = int(np.pi * radius ** 2)

    # Generate seed mask
    if seed_mask is None or not isinstance(seed_mask, np.ndarray):
        seed_mask = np.logical_and(distancemap_clipped >= threshold, valid_mask)
    
    # connectivity=2 (8-connected) matches the extension's implicit watershed behaviour
    # and avoids splitting diagonal seed blobs into multiple instances
    seeds = label(seed_mask, connectivity=2)
    seeds_mask = remove_small_objects(seeds > 0, min_size=max(1, min_area), connectivity=2)
    seeds = label(seeds_mask, connectivity=2)

    # Watershed
    labels = watershed(-distancemap_clipped, markers=seeds, mask=valid_mask, connectivity=2, compactness=compactness)

    # Fill holes in final labels
    return fill_labels(labels)


def map_axon_labels_to_fibres(
    label_img1: np.ndarray,
    label_img2: np.ndarray,
    min_overlap_frac: float = 0.9
) -> np.ndarray:
    """
    Assign each label_img2 fragment to the label_img1 object it overlaps most with,
    provided that overlap covers at least `min_overlap_frac` of the fragment (IoC metric).

    Matches the extension's hierarchy assignment: threshold = 0.9 (90% of inner cylinder
    must fall within the matched fibre). Accepted fragments are clipped to the fibre
    boundary — pixels outside the matched fibre are zeroed. Unmatched fragments are
    set to 0.

    Parameters:
        label_img1 (np.ndarray): Reference label image (e.g., fibres).
        label_img2 (np.ndarray): Fragment label image to map (e.g., inner tongue).
        min_overlap_frac (float): Minimum IoC for a mapping to be accepted. Default = 0.9.

    Returns:
        np.ndarray: Label image where each accepted fragment is assigned its fibre label
                    and clipped to the fibre boundary.
    """
    label_img1 = label_img1.astype(np.int32)
    label_img2 = label_img2.astype(np.int32)

    max_label1 = int(label_img1.max())
    max_label2 = int(label_img2.max())

    if max_label1 == 0 or max_label2 == 0:
        return np.zeros_like(label_img2, dtype=np.int32)

    flat1 = label_img1.ravel()
    flat2 = label_img2.ravel()

    # Integer-centred bin edges ensure each label falls in exactly its own bin
    bins1 = np.arange(-0.5, max_label1 + 1.5)
    bins2 = np.arange(-0.5, max_label2 + 1.5)
    overlap_matrix, _, _ = np.histogram2d(flat1, flat2, bins=[bins1, bins2])
    # overlap_matrix[i, j] = pixel count where label_img1==i and label_img2==j

    merged = np.zeros_like(label_img2, dtype=np.int32)

    for l2 in range(1, max_label2 + 1):
        overlaps = overlap_matrix[1:, l2]  # overlap with each fibre (labels 1..max_label1)
        if overlaps.sum() == 0:
            continue

        best_l1_idx = int(np.argmax(overlaps))
        best_l1 = best_l1_idx + 1
        overlap = overlaps[best_l1_idx]

        l2_area = overlap_matrix[:, l2].sum()  # total pixels of this fragment
        if l2_area == 0:
            continue

        if overlap / l2_area >= min_overlap_frac:
            # Clip to fibre boundary: only assign pixels inside the matched fibre
            merged[(label_img2 == l2) & (label_img1 == best_l1)] = best_l1

    return merged


def get_edge_touching_labels(label_img: np.ndarray) -> set:
    """
    Identify all instance labels that touch any edge of the image.

    Parameters
    ----------
    label_img : np.ndarray
        2D labeled image where 0 is background and positive integers represent instances.

    Returns
    -------
    set
        A set of integer label IDs that touch any image border.
    """
    labels_touching = set()
    rows, cols = label_img.shape

    # Check top, bottom, left, right edges
    for arr in [label_img[0, :], label_img[-1, :], label_img[:, 0], label_img[:, -1]]:
        edge_vals = np.unique(arr)
        labels_touching.update(edge_vals[edge_vals != 0])

    return labels_touching

def remove_edge_touching_labels(
    fibre_labels: np.ndarray,
    inner_tongue_labels: np.ndarray,
    axon_labels: np.ndarray = None
) -> tuple:
    """
    Remove all labels that touch the image border from the provided label images.

    Parameters
    ----------
    fibre_labels : np.ndarray
        Labeled image of fibre instances (2D).
    inner_tongue_labels : np.ndarray
        Labeled image of inner-tongue instances (2D).
    axon_labels : np.ndarray, optional
        Labeled image of axon instances (2D). If provided, these labels will also be cleaned.

    Returns
    -------
    tuple
        Cleaned (fibre_labels, inner_tongue_labels, axon_labels) as np.ndarrays.
        If `axon_labels` was not provided, returns only two arrays.
    """
    edge_labels = get_edge_touching_labels(fibre_labels)
    if not edge_labels:
        return (fibre_labels, inner_tongue_labels, axon_labels) if axon_labels is not None else (fibre_labels, inner_tongue_labels)

    # Remove edge-touching labels from all relevant maps
    fibre_labels = fibre_labels.copy()
    inner_tongue_labels = inner_tongue_labels.copy()
    fibre_labels[np.isin(fibre_labels, list(edge_labels))] = 0
    inner_tongue_labels[np.isin(inner_tongue_labels, list(edge_labels))] = 0

    if axon_labels is not None:
        axon_labels = axon_labels.copy()
        axon_labels[np.isin(axon_labels, list(edge_labels))] = 0
        return fibre_labels, inner_tongue_labels, axon_labels

    return fibre_labels, inner_tongue_labels

def remove_unmapped_labels(label_img: np.ndarray, mapped_img: np.ndarray) -> np.ndarray:
    """
    Remove labels from `label_img` that have no corresponding label in `mapped_img`.

    Parameters
    ----------
    label_img : np.ndarray
        Source label image whose unmapped labels should be removed.
    mapped_img : np.ndarray
        Reference label image that defines which instances are kept.

    Returns
    -------
    np.ndarray
        A copy of `label_img` where unmapped (unmatched) labels are set to 0.
    """
    label_img = label_img.copy()
    labels_in_img1 = np.unique(label_img)
    labels_in_img2 = np.unique(mapped_img)

    labels_in_img1 = labels_in_img1[labels_in_img1 != 0]  # exclude background
    unmapped_labels = [lab for lab in labels_in_img1 if lab not in labels_in_img2]

    if unmapped_labels:
        label_img[np.isin(label_img, unmapped_labels)] = 0

    return label_img


def merge_unmatched_fibres(
    fibre_labels: np.ndarray,
    mapped_axons: np.ndarray,
    padding: int = 2,
    max_merge_distance: float = 1.0
) -> np.ndarray:
    """
    Reassign unmatched fibre labels to their closest matched fibre using local EDT comparison
    within a padded bounding box.

    Parameters:
        fibre_labels (np.ndarray): Fibre instance label image.
        mapped_axons (np.ndarray): Mapped axon label image (with fibre label IDs).
        padding (int): Pixels to expand bounding box around unmatched fibre.
        max_merge_distance (float): Max allowed edge-to-edge distance for merging.

    Returns:
        np.ndarray: Updated fibre label image with unmatched fibres reassigned.
    """
    output = fibre_labels.copy()
    height, width = fibre_labels.shape

    matched_labels = np.unique(mapped_axons)
    matched_labels = matched_labels[matched_labels != 0]
    all_labels = np.unique(fibre_labels)
    unmatched_labels = [l for l in all_labels if l != 0 and l not in matched_labels]

    for uid in unmatched_labels:
        full_mask = fibre_labels == uid
        y_coords, x_coords = np.where(full_mask)
        y_min = max(0, y_coords.min() - padding)
        y_max = min(height, y_coords.max() + padding + 1)
        x_min = max(0, x_coords.min() - padding)
        x_max = min(width, x_coords.max() + padding + 1)

        # Crop region
        cropped_labels = fibre_labels[y_min:y_max, x_min:x_max]
        region_mask = cropped_labels == uid
        edge_unmatched = find_boundaries(region_mask, mode="outer")

        # EDT of unmatched edge
        edt_mask = np.ones_like(region_mask, dtype=bool)
        edt_mask[edge_unmatched] = False
        local_dist_map = edt.edt(edt_mask)

        # Nearby matched fibres in cropped region
        nearby_labels = np.unique(cropped_labels)
        candidate_labels = [l for l in nearby_labels if l in matched_labels and l != uid]

        best_label = None
        best_distance = np.inf

        for mid in candidate_labels:
            matched_mask = cropped_labels == mid
            matched_edge = find_boundaries(matched_mask, mode="outer")
            distances = local_dist_map[matched_edge]
            if distances.size == 0:
                continue
            min_dist = np.min(distances)
            if min_dist < best_distance:
                best_distance = min_dist
                best_label = mid

        if best_label is not None and best_distance <= max_merge_distance:
            output[y_min:y_max, x_min:x_max][region_mask] = best_label

    return output