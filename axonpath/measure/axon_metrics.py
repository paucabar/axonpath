import numpy as np
import pandas as pd
from math import sqrt, pi
from typing import Optional
from skimage.measure import regionprops
import edt
from scipy.spatial import cKDTree


def compute_diameter_from_area(area: float) -> float:
    """Equivalent circular diameter from area (same units as area²)."""
    return 2.0 * sqrt(area / pi) if area > 0 else np.nan


def border_distance(label_img: np.ndarray, label_id: int, bbox: tuple, pad: int = 10) -> float:
    """
    Minimum border-to-border distance from one object to its nearest neighbour.

    Parameters
    ----------
    label_img : np.ndarray
        Label image (0 = background).
    label_id : int
        Label ID to measure.
    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col) from regionprops.
    pad : int
        Initial padding around the bounding box before expanding the search window.

    Returns
    -------
    float
        Minimum border-to-border distance in pixels, or np.nan if no neighbour found.
    """
    H, W = label_img.shape
    minr, minc, maxr, maxc = bbox
    minr = max(minr - pad, 0)
    minc = max(minc - pad, 0)
    maxr = min(maxr + pad, H)
    maxc = min(maxc + pad, W)

    while True:
        sub = label_img[minr:maxr, minc:maxc]
        mask_obj = sub == label_id
        mask_other = (sub != label_id) & (sub != 0)

        if mask_other.any():
            # EDT of ~mask_other: each pixel gets distance to nearest other-object pixel
            dist_map = edt.edt(~mask_other)
            return float(dist_map[mask_obj].min())

        grow_r = maxr - minr
        grow_c = maxc - minc
        minr = max(minr - grow_r, 0)
        minc = max(minc - grow_c, 0)
        maxr = min(maxr + grow_r, H)
        maxc = min(maxc + grow_c, W)

        if minr == 0 and minc == 0 and maxr == H and maxc == W:
            return np.nan


def local_confluence(label_img: np.ndarray, own_label: int, centroid: tuple, window_size_px: int) -> float:
    """
    Local confluence: fraction of the *non-self* area of the square window around
    `centroid` that is covered by *other* fibres. `own_label`'s own pixels are
    excluded from both the numerator and the denominator, so this measures
    surrounding packing density independently of the fibre's own size.

    (Earlier version counted the fibre's own pixels toward the numerator over a
    fixed denominator, which meant confluence was dominated by the fibre's own
    size rather than genuine neighbour density whenever the fibre was a large
    fraction of the window — verified against real data: corr(diameter,
    confluence) = 0.72, median self-fill-fraction ~47% of the (former) 128px
    window before counting any neighbours at all. Fixed together with widening
    window_size_px's default (see metrics_table) — excluding self alone isn't
    enough if the window is barely bigger than the fibre, since the "non-self"
    denominator would shrink toward zero for the largest fibres.)

    Normalised to the actual (clipped) window area, so edge fibres are not penalised
    for having a smaller observable window.

    Parameters
    ----------
    label_img : np.ndarray
        Label image of fibres.
    own_label : int
        Label ID of the fibre this confluence value is being computed for —
        excluded from both numerator and denominator.
    centroid : tuple
        (row, col) centroid of the fibre (from regionprops.centroid).
    window_size_px : int
        Side length of the square window in pixels.

    Returns
    -------
    float
        Fraction of non-self observed pixels occupied by another fibre label (0–1),
        or NaN if the window is entirely (or almost entirely) the fibre itself.
    """
    H, W = label_img.shape
    cy, cx = int(np.round(centroid[0])), int(np.round(centroid[1]))
    half = window_size_px // 2

    r_min = max(cy - half, 0)
    r_max = min(cy - half + window_size_px, H)
    c_min = max(cx - half, 0)
    c_max = min(cx - half + window_size_px, W)
    window = label_img[r_min:r_max, c_min:c_max]

    self_mask = window == own_label
    non_self_total = window.size - int(self_mask.sum())
    if non_self_total <= 0:
        return np.nan
    other_count = int(np.sum((window > 0) & ~self_mask))
    return float(other_count / non_self_total)


def _centroid_min_distances(props: list) -> dict:
    """
    Return {label: min_centroid_distance_px} for all labels in one KDTree query (O(N log N)).
    """
    if len(props) < 2:
        return {r.label: np.nan for r in props}
    centroids = np.array([r.centroid for r in props])
    labels = [r.label for r in props]
    tree = cKDTree(centroids)
    distances, _ = tree.query(centroids, k=2)
    return {lbl: float(d) for lbl, d in zip(labels, distances[:, 1])}


def metrics_table(
    fibre_labels: np.ndarray,
    inner_cylinder_labels: np.ndarray,
    pixel_size_um: Optional[float] = None,
    axon_labels: Optional[np.ndarray] = None,
    window_size_px: int = 512,
) -> pd.DataFrame:
    """
    Compute morphometric and spatial measurements for labelled fibres.

    Parameters
    ----------
    fibre_labels : np.ndarray
        Label image of fibres (unique integer IDs, 0 = background).
    inner_cylinder_labels : np.ndarray
        Label image of mapped inner cylinders (same label IDs as fibres).
    pixel_size_um : float, optional
        Pixel size in micrometres. When provided, area columns are labelled
        '(µm²)' and length columns '(µm)'; otherwise '(px²)' / '(px)'.
    axon_labels : np.ndarray, optional
        Label image of mapped axons (same label IDs as fibres).
    window_size_px : int
        Side length of the square window for Local Confluence, in pixels.

    Returns
    -------
    pd.DataFrame
        One row per fibre. Column names include a unit suffix that reflects
        whether pixel_size_um was supplied.
    """
    if pixel_size_um is not None:
        px2 = pixel_size_um ** 2
        area_sfx = "(µm²)"
        len_sfx = "(µm)"
        px_len = pixel_size_um
    else:
        px2 = 1.0
        area_sfx = "(px²)"
        len_sfx = "(px)"
        px_len = 1.0

    all_props = regionprops(fibre_labels)
    centroid_dist = _centroid_min_distances(all_props)

    results = []
    for region in all_props:
        lbl = region.label
        area_px = region.area

        inner_area_px = int(np.sum(inner_cylinder_labels == lbl))

        # g-ratios computed from px areas — scale cancels, dimensionless
        fibre_diam_px = compute_diameter_from_area(area_px)
        inner_diam_px = compute_diameter_from_area(inner_area_px)
        myelin_g = inner_diam_px / fibre_diam_px if fibre_diam_px > 0 else np.nan

        if axon_labels is not None:
            axon_area_px = int(np.sum(axon_labels == lbl))
            axon_diam_px = compute_diameter_from_area(axon_area_px)
            axon_g = axon_diam_px / fibre_diam_px if fibre_diam_px > 0 else np.nan
            axon_count = 1 if axon_area_px > 0 else 0
        else:
            axon_area_px = None
            axon_g = np.nan
            axon_count = 0

        perimeter = getattr(region, "perimeter", np.nan)
        circularity = (4 * pi * area_px / perimeter ** 2) if perimeter and perimeter > 0 else np.nan
        circularity = min(circularity, 1.0)

        min_border_px = border_distance(fibre_labels, lbl, region.bbox)
        min_centroid_px = centroid_dist.get(lbl, np.nan)

        results.append({
            "Label": lbl,
            f"Fibre Area {area_sfx}": area_px * px2,
            f"InnerCylinder Area {area_sfx}": inner_area_px * px2,
            f"Axon Area {area_sfx}": (axon_area_px * px2 if axon_area_px is not None else np.nan),
            "Myelin g-ratio": myelin_g,
            "Axon g-ratio": axon_g,
            "Fibre Circularity": circularity,
            "Fibre Solidity": getattr(region, "solidity", np.nan),
            "Axon Count": axon_count,
            f"Fibre MajorAxisLength {len_sfx}": getattr(region, "axis_major_length", np.nan) * px_len,
            "Fibre Eccentricity": getattr(region, "eccentricity", np.nan),
            f"Min Border Distance {len_sfx}": (min_border_px * px_len if not np.isnan(min_border_px) else np.nan),
            f"Min Centroid Distance {len_sfx}": (min_centroid_px * px_len if not np.isnan(min_centroid_px) else np.nan),
            "Local Confluence": local_confluence(fibre_labels, lbl, region.centroid, window_size_px),
        })

    return pd.DataFrame(results)


def single_metric(
    fibre_labels: np.ndarray,
    inner_cylinder_labels: np.ndarray,
    metric_name: str,
    pixel_size_um: Optional[float] = None,
    label_id: Optional[int] = None,
    axon_labels: Optional[np.ndarray] = None,
    window_size_px: int = 512,
):
    """
    Compute one specific metric, either for a given label or all labels.

    Parameters
    ----------
    fibre_labels : np.ndarray
        Label image of fibres.
    inner_cylinder_labels : np.ndarray
        Label image of mapped inner cylinders.
    metric_name : str
        Column name of the metric. Area and length columns include a unit
        suffix that depends on pixel_size_um (e.g. 'Fibre Area (µm²)' or
        'Fibre Area (px²)').
    pixel_size_um : float, optional
        Pixel size in micrometres. See metrics_table.
    label_id : int, optional
        Specific label ID. If None, returns values for all labels.
    axon_labels : np.ndarray, optional
        Label image of mapped axons.
    window_size_px : int
        Window size for Local Confluence, in pixels.

    Returns
    -------
    float or pd.Series
        Single value if label_id is given, otherwise a Series indexed by label.
    """
    df = metrics_table(
        fibre_labels, inner_cylinder_labels,
        pixel_size_um=pixel_size_um,
        axon_labels=axon_labels,
        window_size_px=window_size_px,
    )

    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found. Available: {list(df.columns)}")

    if label_id is not None:
        row = df.loc[df["Label"] == label_id, metric_name]
        return row.iloc[0] if not row.empty else np.nan
    return df.set_index("Label")[metric_name]
