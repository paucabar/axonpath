"""
Smoke tests for core aimsegdl components.

Run from the axonpath conda environment:
    pytest tests/test_smoke.py -v

All tests run on CPU with small synthetic inputs.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Normalisation
# ---------------------------------------------------------------------------

def test_normalize_range_and_shape():
    """normalize() must output values in [0, 1] and preserve spatial shape."""
    from aimsegdl.utils.image_processing import normalize

    rng = np.random.default_rng(0)
    img = rng.integers(0, 4096, size=(512, 512), dtype=np.uint16).astype(np.float32)
    out = normalize(img)

    assert out.shape == img.shape, "normalize must preserve shape"
    assert out.min() >= 0.0, f"min {out.min()} < 0"
    assert out.max() <= 1.0, f"max {out.max()} > 1"


# ---------------------------------------------------------------------------
# 2. SDT creation
# ---------------------------------------------------------------------------

def test_sdt_non_zero_for_labelled_image():
    """SDT for an image with objects must contain positive interior values."""
    from aimsegdl.skeleton.skeleton_aware_distance_transform import LabelDistanceTransforms

    label_img = np.zeros((64, 64), dtype=np.int32)
    label_img[10:30, 10:30] = 1   # one square object
    label_img[40:55, 40:55] = 2   # a second object

    sdt, _, _ = LabelDistanceTransforms(label_img, alpha=0.3, signed_background=True).skeleton_aware_dist_trans()

    assert sdt.shape == label_img.shape, "SDT must preserve shape"
    assert np.any(sdt > 0), "SDT must have positive interior values for non-empty label image"
    assert np.any(sdt < 0), "SDT must have negative exterior values (background = -1)"


def test_sdt_all_background_for_empty_image():
    """SDT for an all-zero label image must be all ≤ 0 (no interior)."""
    from aimsegdl.skeleton.skeleton_aware_distance_transform import LabelDistanceTransforms

    empty = np.zeros((64, 64), dtype=np.int32)
    sdt, _, _ = LabelDistanceTransforms(empty, alpha=0.3).skeleton_aware_dist_trans()

    assert np.all(sdt <= 0), "SDT of empty label image must be all background (≤ 0)"


# ---------------------------------------------------------------------------
# 3. Model forward pass
# ---------------------------------------------------------------------------

def test_model_forward_output_shape():
    """UNet must return 5-channel output matching spatial input dimensions."""
    from aimsegdl.utils.model_building import model_fn

    model = model_fn("cpu", norm_type="batch")
    model.eval()

    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 5, 64, 64), (
        f"Expected (1, 5, 64, 64), got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# 4. Pipeline forward pass
# ---------------------------------------------------------------------------

def test_pipeline_preserves_spatial_dims():
    """Pipeline must return 3-channel output with the same H×W as the input."""
    from aimsegdl.pipeline import Pipeline
    from aimsegdl.utils.model_building import model_fn

    model = model_fn("cpu", norm_type="batch")
    pipeline = Pipeline(model)

    H, W = 64, 64
    x = torch.randn(1, 1, H, W)
    with torch.no_grad():
        out = pipeline(x)

    assert out.shape[0] == 1, f"Batch dim must be 1, got {out.shape[0]}"
    assert out.shape[1] == 3, f"Channel dim must be 3 (semantic logits), got {out.shape[1]}"
    assert out.shape[2] == H and out.shape[3] == W, (
        f"Spatial dims must be {H}×{W}, got {out.shape[2]}×{out.shape[3]}"
    )


# ---------------------------------------------------------------------------
# 5. Instance segmentation from SDT
# ---------------------------------------------------------------------------

def test_instance_segmentation_returns_integer_labels():
    """segment_instances_from_sdt must return a non-trivial integer label map."""
    from aimsegdl.utils.image_processing import segment_instances_from_sdt

    # Build a synthetic SDT: two blobs with positive interior, -1 elsewhere
    sdt = np.full((64, 64), -1.0, dtype=np.float32)
    sdt[10:25, 10:25] = 0.5
    sdt[40:55, 40:55] = 0.5

    sdt_tensor = torch.from_numpy(sdt).unsqueeze(0)  # shape (1, H, W)
    labels = segment_instances_from_sdt(sdt_tensor, threshold=0.3, min_diameter=3.0)

    assert labels.dtype in (np.int32, np.int64, np.int16, np.uint16, int), (
        f"Expected integer label array, got dtype {labels.dtype}"
    )
    assert labels.max() >= 1, "Expected at least one labelled instance"
