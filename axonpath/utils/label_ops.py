import numpy as np


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
        label_img2 (np.ndarray): Fragment label image to map (e.g., Inner Cylinder).
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
    inner_cylinder_labels: np.ndarray,
    axon_labels: np.ndarray = None
) -> tuple:
    """
    Remove all labels that touch the image border from the provided label images.

    Parameters
    ----------
    fibre_labels : np.ndarray
        Labeled image of fibre instances (2D).
    inner_cylinder_labels : np.ndarray
        Labeled image of inner cylinder instances (2D).
    axon_labels : np.ndarray, optional
        Labeled image of axon instances (2D). If provided, these labels will also be cleaned.

    Returns
    -------
    tuple
        Cleaned (fibre_labels, inner_cylinder_labels, axon_labels) as np.ndarrays.
        If `axon_labels` was not provided, returns only two arrays.
    """
    edge_labels = get_edge_touching_labels(fibre_labels)
    if not edge_labels:
        return (fibre_labels, inner_cylinder_labels, axon_labels) if axon_labels is not None else (fibre_labels, inner_cylinder_labels)

    # Remove edge-touching labels from all relevant maps
    fibre_labels = fibre_labels.copy()
    inner_cylinder_labels = inner_cylinder_labels.copy()
    fibre_labels[np.isin(fibre_labels, list(edge_labels))] = 0
    inner_cylinder_labels[np.isin(inner_cylinder_labels, list(edge_labels))] = 0

    if axon_labels is not None:
        axon_labels = axon_labels.copy()
        axon_labels[np.isin(axon_labels, list(edge_labels))] = 0
        return fibre_labels, inner_cylinder_labels, axon_labels

    return fibre_labels, inner_cylinder_labels


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
