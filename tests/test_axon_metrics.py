"""
Tests for axonpath.measure.axon_metrics.

Run from the axonpath conda environment:
    pytest tests/test_axon_metrics.py -v
"""

import numpy as np
import pytest
from math import sqrt, pi
from skimage.measure import regionprops

from axonpath.measure.axon_metrics import (
    metrics_table,
    border_distance,
    local_confluence,
    _centroid_min_distances,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_labels():
    """
    Two symmetric 30×30 fibres in a 128×128 image.
      Label 1: rows 10:40, cols 10:40  (900 px fibre, 400 px inner, 196 px axon)
      Label 2: rows 70:100, cols 70:100
    Centroids: (24.5, 24.5) and (84.5, 84.5) → distance = 60√2.
    """
    fibre = np.zeros((128, 128), dtype=np.int32)
    fibre[10:40, 10:40] = 1
    fibre[70:100, 70:100] = 2

    inner = np.zeros_like(fibre)
    inner[15:35, 15:35] = 1   # 20×20 = 400 px
    inner[75:95, 75:95] = 2

    axon = np.zeros_like(fibre)
    axon[18:32, 18:32] = 1    # 14×14 = 196 px
    axon[78:92, 78:92] = 2

    return fibre, inner, axon


# ---------------------------------------------------------------------------
# 1. Column suffixes switch with pixel_size_um
# ---------------------------------------------------------------------------

def test_column_suffix_um():
    fibre, inner, _ = _make_labels()
    df = metrics_table(fibre, inner, pixel_size_um=0.1)
    for col in ("Fibre Area (µm²)", "InnerCylinder Area (µm²)", "Axon Area (µm²)",
                "Fibre MajorAxisLength (µm)", "Min Border Distance (µm)",
                "Min Centroid Distance (µm)"):
        assert col in df.columns, f"Expected column '{col}'"


def test_column_suffix_px():
    fibre, inner, _ = _make_labels()
    df = metrics_table(fibre, inner)
    for col in ("Fibre Area (px²)", "InnerCylinder Area (px²)", "Axon Area (px²)",
                "Fibre MajorAxisLength (px)", "Min Border Distance (px)",
                "Min Centroid Distance (px)"):
        assert col in df.columns, f"Expected column '{col}'"


def test_dimensionless_columns_present():
    fibre, inner, axon = _make_labels()
    df = metrics_table(fibre, inner, axon_labels=axon)
    for col in ("Myelin g-ratio", "Axon g-ratio", "Fibre Circularity",
                "Fibre Solidity", "Fibre Eccentricity", "Axon Count", "Local Confluence"):
        assert col in df.columns, f"Expected column '{col}'"


# ---------------------------------------------------------------------------
# 2. Row count
# ---------------------------------------------------------------------------

def test_row_count():
    fibre, inner, axon = _make_labels()
    df = metrics_table(fibre, inner, axon_labels=axon)
    assert len(df) == 2
    assert set(df["Label"]) == {1, 2}


# ---------------------------------------------------------------------------
# 3. Area values
# ---------------------------------------------------------------------------

def test_area_values_px():
    fibre, inner, axon = _make_labels()
    df = metrics_table(fibre, inner, axon_labels=axon).set_index("Label")
    assert df.loc[1, "Fibre Area (px²)"] == pytest.approx(900)
    assert df.loc[1, "InnerCylinder Area (px²)"] == pytest.approx(400)
    assert df.loc[1, "Axon Area (px²)"] == pytest.approx(196)


def test_area_scales_with_pixel_size():
    fibre, inner, _ = _make_labels()
    px = 0.07
    df_px = metrics_table(fibre, inner).set_index("Label")
    df_um = metrics_table(fibre, inner, pixel_size_um=px).set_index("Label")
    assert df_um.loc[1, "Fibre Area (µm²)"] == pytest.approx(
        df_px.loc[1, "Fibre Area (px²)"] * px ** 2
    )


# ---------------------------------------------------------------------------
# 4. g-ratios
# ---------------------------------------------------------------------------

def test_g_ratio_dimensionless():
    """g-ratios must be identical regardless of pixel_size_um."""
    fibre, inner, axon = _make_labels()
    df_px = metrics_table(fibre, inner, axon_labels=axon).set_index("Label")
    df_um = metrics_table(fibre, inner, pixel_size_um=0.1, axon_labels=axon).set_index("Label")
    assert df_px.loc[1, "Myelin g-ratio"] == pytest.approx(df_um.loc[1, "Myelin g-ratio"])
    assert df_px.loc[1, "Axon g-ratio"] == pytest.approx(df_um.loc[1, "Axon g-ratio"])


def test_myelin_g_ratio_formula():
    """Myelin g-ratio = diam(InnerCylinder) / diam(Fibre) = sqrt(400/900) = 2/3."""
    fibre, inner, _ = _make_labels()
    df = metrics_table(fibre, inner).set_index("Label")
    expected = sqrt(400 / 900)   # = 2/3, since 2√(A/π) / 2√(B/π) = √(A/B)
    assert df.loc[1, "Myelin g-ratio"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Axon Count
# ---------------------------------------------------------------------------

def test_axon_count_with_axons():
    fibre, inner, axon = _make_labels()
    df = metrics_table(fibre, inner, axon_labels=axon).set_index("Label")
    assert df.loc[1, "Axon Count"] == 1
    assert df.loc[2, "Axon Count"] == 1


def test_axon_count_no_axon_arg():
    fibre, inner, _ = _make_labels()
    df = metrics_table(fibre, inner).set_index("Label")
    assert (df["Axon Count"] == 0).all()


def test_axon_fibre_without_axon():
    """Fibre without a corresponding axon label gets Axon Count 0."""
    fibre, inner, axon = _make_labels()
    axon[axon == 2] = 0   # remove axon for fibre 2
    df = metrics_table(fibre, inner, axon_labels=axon).set_index("Label")
    assert df.loc[1, "Axon Count"] == 1
    assert df.loc[2, "Axon Count"] == 0


def test_axon_area_nan_when_no_axon_arg():
    fibre, inner, _ = _make_labels()
    df = metrics_table(fibre, inner).set_index("Label")
    assert np.isnan(df.loc[1, "Axon Area (px²)"])


# ---------------------------------------------------------------------------
# 6. _centroid_min_distances
# ---------------------------------------------------------------------------

def test_centroid_min_distances_single():
    img = np.zeros((64, 64), dtype=np.int32)
    img[10:30, 10:30] = 1
    dist = _centroid_min_distances(regionprops(img))
    assert np.isnan(dist[1])


def test_centroid_min_distances_two_fibres():
    fibre, _, _ = _make_labels()
    dist = _centroid_min_distances(regionprops(fibre))
    # Centroid 1: (24.5, 24.5), centroid 2: (84.5, 84.5) → 60√2
    expected = 60 * sqrt(2)
    assert dist[1] == pytest.approx(expected, rel=1e-4)
    assert dist[2] == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# 7. border_distance
# ---------------------------------------------------------------------------

def test_border_distance_known_gap():
    """Right edge of label 1 at col 14, left edge of label 2 at col 20 → distance = 6."""
    img = np.zeros((20, 64), dtype=np.int32)
    img[:, 5:15] = 1    # cols 5–14
    img[:, 20:30] = 2   # cols 20–29
    props = {r.label: r for r in regionprops(img)}
    d = border_distance(img, 1, props[1].bbox)
    assert d == pytest.approx(6.0)


def test_border_distance_adjacent():
    """Two touching objects (no gap) → border distance = 1."""
    img = np.zeros((20, 30), dtype=np.int32)
    img[:, 5:15] = 1    # right edge at col 14
    img[:, 15:25] = 2   # left edge at col 15
    props = {r.label: r for r in regionprops(img)}
    d = border_distance(img, 1, props[1].bbox)
    assert d == pytest.approx(1.0)


def test_border_distance_no_neighbor():
    """Single label — no neighbour → NaN."""
    img = np.zeros((64, 64), dtype=np.int32)
    img[10:20, 10:20] = 1
    props = {r.label: r for r in regionprops(img)}
    d = border_distance(img, 1, props[1].bbox)
    assert np.isnan(d)


# ---------------------------------------------------------------------------
# 8. local_confluence
# ---------------------------------------------------------------------------

def test_local_confluence_full():
    """Self excluded from both num/denom; every non-self pixel in the window
    belongs to another fibre → confluence = 1."""
    img = np.full((64, 64), 2, dtype=np.int32)  # everything is a neighbouring fibre
    img[30:35, 30:35] = 1                        # small "self" region
    conf = local_confluence(img, own_label=1, centroid=(32, 32), window_size_px=20)
    assert conf == pytest.approx(1.0)


def test_local_confluence_isolated_self():
    """Self alone in an otherwise-empty window, no neighbours → confluence = 0."""
    img = np.zeros((64, 64), dtype=np.int32)
    img[32, 32] = 1  # subject's own single pixel, nothing else nearby
    conf = local_confluence(img, own_label=1, centroid=(32, 32), window_size_px=10)
    assert conf == pytest.approx(0.0)


def test_local_confluence_single_neighbor():
    """One neighbouring fibre pixel next to an isolated self pixel, in a 10x10
    window (100 px, 1 self px excluded) → confluence = 1/99."""
    img = np.zeros((64, 64), dtype=np.int32)
    img[32, 32] = 1  # self
    img[33, 33] = 2  # one neighbouring fibre pixel
    conf = local_confluence(img, own_label=1, centroid=(32, 32), window_size_px=10)
    assert conf == pytest.approx(1 / 99)


def test_local_confluence_clipped_window():
    """Window at image corner is clipped; normalised to the actual (smaller)
    non-self area, not the nominal window size."""
    img = np.full((64, 64), 2, dtype=np.int32)  # everything is a neighbouring fibre
    img[0, 0] = 1  # self at the very corner
    conf = local_confluence(img, own_label=1, centroid=(0, 0), window_size_px=10)
    assert conf == pytest.approx(1.0)  # all non-self pixels in the clipped window are "other"


def test_local_confluence_self_fills_window_returns_nan():
    """Self fills the entire window (no non-self area at all) → NaN, not a
    spurious 1.0 — this is the behaviour the exclude-self rewrite was for:
    a fibre's own bulk should never inflate its own confluence value."""
    img = np.ones((64, 64), dtype=np.int32)
    conf = local_confluence(img, own_label=1, centroid=(32, 32), window_size_px=20)
    assert np.isnan(conf)
