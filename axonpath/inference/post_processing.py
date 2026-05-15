import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import label
from skimage.morphology import remove_small_objects
import edt
from axonpath.utils.image_processing import fill_labels


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
    seed_mask: np.ndarray = None,
    sdt_smooth_sigma: float = 1.5,
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
        sdt_smooth_sigma (float): Sigma for Gaussian smoothing applied to the SDT before seed
            extraction. Fills shallow SDT valleys caused by noisy predictions, preventing large
            fibres from being split into multiple instances. Set to 0 to disable.

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

    # Generate seed mask — use smoothed SDT so shallow prediction valleys don't
    # split a single large fibre into multiple seed blobs. The original (unsmoothed)
    # map is kept for the watershed gradient to preserve sharp instance boundaries.
    if seed_mask is None or not isinstance(seed_mask, np.ndarray):
        sdt_for_seeds = gaussian_filter(distancemap_clipped, sigma=sdt_smooth_sigma) if sdt_smooth_sigma > 0 else distancemap_clipped
        seed_mask = np.logical_and(sdt_for_seeds >= threshold, valid_mask)

    # connectivity=2 (8-connected) matches the extension's implicit watershed behaviour
    # and avoids splitting diagonal seed blobs into multiple instances
    seeds = label(seed_mask, connectivity=2)
    seeds_mask = remove_small_objects(seeds > 0, max_size=max(0, min_area - 1), connectivity=2)
    seeds = label(seeds_mask, connectivity=2)

    # Watershed
    labels = watershed(-distancemap_clipped, markers=seeds, mask=valid_mask, connectivity=2, compactness=compactness)

    # Fill holes in final labels
    return fill_labels(labels)


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
