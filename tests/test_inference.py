"""
Tests for axonpath.inference.inference.

Run from the axonpath conda environment:
    pytest tests/test_inference.py -v
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. run_inference — output shapes and types
# ---------------------------------------------------------------------------

def test_run_inference_output_shapes():
    """run_inference must return correctly shaped arrays for all three outputs."""
    from axonpath.inference.inference import run_inference
    from axonpath.utils.model_building import model_fn

    H, W = 64, 64
    model = model_fn("cpu", norm_type="batch")
    image = np.random.default_rng(0).uniform(0, 1, (H, W)).astype(np.float32)

    labels_fibre, labels_axon, semantic = run_inference(
        image, model, device="cpu", roi_size=(64, 64), min_diameter=10.0
    )

    assert labels_fibre.shape == (H, W), f"labels_fibre shape {labels_fibre.shape} != {(H, W)}"
    assert semantic.shape == (H, W), f"semantic shape {semantic.shape} != {(H, W)}"
    assert semantic.dtype == np.uint8, f"semantic dtype {semantic.dtype} != uint8"
    assert np.issubdtype(labels_fibre.dtype, np.integer), "labels_fibre must be integer dtype"

    if labels_axon is not None:
        assert labels_axon.shape == (H, W), f"labels_axon shape {labels_axon.shape} != {(H, W)}"
        assert np.issubdtype(labels_axon.dtype, np.integer), "labels_axon must be integer dtype"


# ---------------------------------------------------------------------------
# 2. predict_inner_cylinder=False returns None
# ---------------------------------------------------------------------------

def test_run_inference_no_inner_cylinder():
    """predict_inner_cylinder=False must return None as the second element."""
    from axonpath.inference.inference import run_inference
    from axonpath.utils.model_building import model_fn

    model = model_fn("cpu", norm_type="batch")
    image = np.random.default_rng(1).uniform(0, 1, (64, 64)).astype(np.float32)

    _, labels_axon, _ = run_inference(
        image, model, device="cpu", roi_size=(64, 64),
        min_diameter=10.0, predict_inner_cylinder=False
    )

    assert labels_axon is None, "Expected None for labels_axon when predict_inner_cylinder=False"


# ---------------------------------------------------------------------------
# 3. Semantic valid_mask constrains instance segmentation
# ---------------------------------------------------------------------------

def test_semantic_valid_mask_constrains_instances():
    """Instances from segment_instances_from_sdt must not appear outside valid_mask.

    This directly tests the key behavioral change in run_inference: watershed is
    now restricted to the semantic prediction mask, matching the extension pipeline.
    """
    from axonpath.utils.image_processing import segment_instances_from_sdt

    H, W = 64, 64
    # Two SDT blobs: top-left and bottom-right, both with strong positive interior
    sdt = np.full((H, W), -1.0, dtype=np.float32)
    sdt[8:24, 8:24] = 0.8   # blob 1 — inside valid_mask
    sdt[40:56, 40:56] = 0.8  # blob 2 — outside valid_mask

    # Valid mask covers only the top-left region (blob 1)
    valid_mask = np.zeros((H, W), dtype=bool)
    valid_mask[4:28, 4:28] = True

    sdt_tensor = torch.from_numpy(sdt).unsqueeze(0)
    labels = segment_instances_from_sdt(
        sdt_tensor, threshold=0.5, min_diameter=5.0, valid_mask=valid_mask
    )

    outside = labels[~valid_mask]
    assert outside.max() == 0, (
        f"Instances found outside valid_mask (max label outside: {outside.max()}). "
        "Watershed is not being constrained by the semantic mask."
    )

    inside = labels[valid_mask]
    assert inside.max() >= 1, "No instances found inside valid_mask — check SDT values or threshold."


# ---------------------------------------------------------------------------
# 4. load_model reads norm_type from checkpoint
# ---------------------------------------------------------------------------

def test_load_model_reads_norm_type(tmp_path):
    """load_model must build the model with the norm_type stored in the checkpoint."""
    from axonpath.inference.inference import load_model
    from axonpath.utils.model_building import model_fn

    # Save a minimal checkpoint with instance norm
    model = model_fn("cpu", norm_type="instance")
    weights_path = str(tmp_path / "test_weights.pth")
    torch.save({"state_dict": model.state_dict(), "norm_type": "instance"}, weights_path)

    loaded = load_model(weights_path, device="cpu")
    loaded.eval()

    # Model must accept a forward pass without error
    with torch.no_grad():
        out = loaded(torch.randn(1, 1, 64, 64))
    assert out.shape == (1, 5, 64, 64), f"Unexpected output shape: {out.shape}"
