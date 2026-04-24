import numpy as np
import pandas as pd
from math import sqrt, pi
from skimage.measure import regionprops
import edt
from scipy.spatial import cKDTree


#  Helper functions

def compute_diameter_from_area(area):
    """Compute equivalent circular diameter from area."""
    return 2.0 * sqrt(area / pi) if area > 0 else np.nan


#  Single-label metric functions

def border_distance(label_img, label_id, pad=10):
    """
    Compute the minimum border-to-border distance from one object
    to its nearest neighbor.

    Parameters
    ----------
    label_img : np.ndarray
        Label image (0 = background).
    label_id : int
        Label ID to measure.
    pad : int
        Initial padding around the bounding box.

    Returns
    -------
    float
        Minimum border-to-border distance in pixels, or np.nan if none found.
    """
    H, W = label_img.shape
    props = [r for r in regionprops(label_img) if r.label == label_id]
    if not props:
        return np.nan
    region = props[0]

    minr, minc, maxr, maxc = region.bbox
    minr = max(minr - pad, 0)
    minc = max(minc - pad, 0)
    maxr = min(maxr + pad, H)
    maxc = min(maxc + pad, W)

    lbl = label_id
    while True:
        sub = label_img[minr:maxr, minc:maxc]
        mask_obj = (sub == lbl)
        mask_other = (sub != lbl) & (sub != 0)

        if mask_other.any():
            dist_map = edt.edt(~mask_other)
            return float(dist_map[mask_obj].min())

        # Expand window
        grow_r = maxr - minr
        grow_c = maxc - minc
        minr = max(minr - grow_r, 0)
        minc = max(minc - grow_c, 0)
        maxr = min(maxr + grow_r, H)
        maxc = min(maxc + grow_c, W)

        if minr == 0 and minc == 0 and maxr == H and maxc == W:
            return np.nan


def centroid_min_distance(label_img, label_id):
    """
    Compute minimum centroid-to-centroid distance for a given object.

    Parameters
    ----------
    label_img : np.ndarray
        Label image with integer labels (0 = background).
    label_id : int
        Label ID to measure.

    Returns
    -------
    float
        Minimum centroid distance (in pixels), or np.nan if no neighbors exist.
    """
    props = regionprops(label_img)
    if len(props) < 2:
        return np.nan

    centroids = np.array([p.centroid for p in props])
    labels = np.array([p.label for p in props])
    if label_id not in labels:
        return np.nan

    tree = cKDTree(centroids)
    distances, _ = tree.query(centroids, k=2)
    dist_dict = {lbl: dist for lbl, dist in zip(labels, distances[:, 1])}
    return dist_dict.get(label_id, np.nan)


def local_confluence(label_img, label_id, window_size_px):
    """
    Compute local confluence (fraction of fibre area in window).

    Parameters
    ----------
    label_img : np.ndarray
        Label image of fibres.
    label_id : int
        Label ID to evaluate.
    window_size_px : int
        Size of the square window in pixels.

    Returns
    -------
    float
        Local confluence (0–1).
    """
    props = [r for r in regionprops(label_img) if r.label == label_id]
    if not props:
        return np.nan

    region = props[0]
    cy, cx = map(int, np.round(region.centroid))
    H, W = label_img.shape
    half = window_size_px // 2

    r_min, r_max = max(cy - half, 0), min(cy + half + 1, H)
    c_min, c_max = max(cx - half, 0), min(cx + half + 1, W)
    window = label_img[r_min:r_max, c_min:c_max]

    fibre_pixels = np.sum(window > 0)
    total_pixels = window.size
    return fibre_pixels / total_pixels if total_pixels > 0 else np.nan


#  Main measurement table

def metrics_table(fibre_labels, inner_cylinder_labels, pixel_size_um,
                  axon_labels=None, window_size_px=128):
    """
    Compute morphometric and spatial measurements for labelled fibres.

    Parameters
    ----------
    fibre_labels : np.ndarray
        Label image of fibres (unique integer IDs).
    inner_cylinder_labels : np.ndarray
        Label image of mapped Inner Cylinders.
    pixel_size_um : float
        Pixel size in micrometers.
    axon_labels : np.ndarray, optional
        Label image of mapped axons, by default None.
    window_size_px : int, optional
        Window size for local confluence, by default 128.

    Returns
    -------
    pd.DataFrame
        Table of per-fibre measurements.
    """
    results = []
    px_um = pixel_size_um

    for region in regionprops(fibre_labels):
        lbl = region.label
        area_px = region.area
        area_um2 = area_px * (px_um ** 2)
        fibre_diam_um = compute_diameter_from_area(area_um2)

        # Inner Cylinder
        inner_area_px = np.sum(inner_cylinder_labels == lbl)
        inner_area_um2 = inner_area_px * (px_um ** 2)
        inner_diam_um = compute_diameter_from_area(inner_area_um2)
        g_ratio_inner = inner_diam_um / fibre_diam_um if fibre_diam_um > 0 else np.nan

        # Optional axon
        if axon_labels is not None:
            axon_area_px = np.sum(axon_labels == lbl)
            axon_area_um2 = axon_area_px * (px_um ** 2)
            axon_diam_um = compute_diameter_from_area(axon_area_um2)
            g_ratio_axon = axon_diam_um / fibre_diam_um if fibre_diam_um > 0 else np.nan
        else:
            axon_area_um2 = np.nan
            axon_diam_um = np.nan
            g_ratio_axon = np.nan

        # Morphometrics
        perimeter = getattr(region, "perimeter", np.nan)
        circularity = (4 * pi * area_px / (perimeter ** 2)) if perimeter and perimeter > 0 else np.nan
        circularity = min(circularity, 1)
        solidity = getattr(region, "solidity", np.nan)
        major_axis_length = getattr(region, "axis_major_length", np.nan) * px_um
        eccentricity = getattr(region, "eccentricity", np.nan)

        # Spatial metrics
        min_border_px = border_distance(fibre_labels, lbl)
        min_border_um = min_border_px * px_um if not np.isnan(min_border_px) else np.nan
        min_centroid_px = centroid_min_distance(fibre_labels, lbl)
        min_centroid_um = min_centroid_px * px_um if not np.isnan(min_centroid_px) else np.nan
        confluence = local_confluence(fibre_labels, lbl, window_size_px)

        results.append({
            "Label": lbl,
            "Fibre_Area_µm²": area_um2,
            "InnerTongue_Area_µm²": inner_area_um2,
            "Axon_Area_µm²": axon_area_um2,
            "Fibre_Diameter_µm": fibre_diam_um,
            "InnerTongue_Diameter_µm": inner_diam_um,
            "Axon_Diameter_µm": axon_diam_um,
            "g_ratio_myelin": g_ratio_inner,
            "g_ratio_axon": g_ratio_axon,
            "Circularity": circularity,
            "Solidity": solidity,
            "MajorAxisLength_µm": major_axis_length,
            "Eccentricity": eccentricity,
            "Min_Border_Dist_µm": min_border_um,
            "Min_Centroid_Dist_µm": min_centroid_um,
            "Local_Confluence": confluence,
        })

    return pd.DataFrame(results)


#  Single metric query

def single_metric(fibre_labels, inner_cylinder_labels, pixel_size_um,
                  metric_name, label_id=None, axon_labels=None, window_size_px=128):
    """
    Compute one specific metric, either for a given label or all labels.

    Parameters
    ----------
    fibre_labels : np.ndarray
        Label image of fibres.
    inner_cylinder_labels : np.ndarray
        Label image of mapped Inner Cylinders.
    pixel_size_um : float
        Pixel size in micrometers.
    metric_name : str
        Name of the metric (e.g. 'Circularity', 'g_ratio_axon').
    label_id : int, optional
        Specific label ID to measure. If None, returns all labels.
    axon_labels : np.ndarray, optional
        Label image of mapped axons, by default None.
    window_size_px : int, optional
        Window size for confluence, by default 128.

    Returns
    -------
    float or pd.Series
        Single value if label_id is given, otherwise a pandas Series for all fibres.
    """
    df = metrics_table(fibre_labels, inner_cylinder_labels, pixel_size_um,
                       axon_labels=axon_labels, window_size_px=window_size_px)

    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found. Available: {list(df.columns)}")

    if label_id is not None:
        row = df.loc[df["Label"] == label_id, metric_name]
        return row.iloc[0] if not row.empty else np.nan
    else:
        return df.set_index("Label")[metric_name]
