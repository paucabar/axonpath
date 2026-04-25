"""
Tests for axonpath.data_preparation.data_preparation.

Run from the axonpath conda environment:
    pytest tests/test_data_preparation.py -v
"""

import csv
import numpy as np
import pytest
from pathlib import Path
from skimage.draw import disk
from skimage import io


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_image(seed=0, n_fibres=2):
    """Return (image, mask_sem, label_img) arrays, 512x512, with n_fibres non-border fibres."""
    rng = np.random.default_rng(seed)
    H, W = 512, 512
    image = rng.uniform(0, 255, (H, W)).astype(np.float32)
    mask_sem = np.zeros((H, W), dtype=np.uint8)
    label_img = np.zeros((H, W), dtype=np.uint16)

    # Centres spread across image, well away from the 512x512 border
    centres = [(128 + 256 * (i % 2), 128 + 256 * (i // 2)) for i in range(n_fibres)]
    for fid, (cy, cx) in enumerate(centres, start=1):
        rr, cc = disk((cy, cx), 40, shape=(H, W))
        mask_sem[rr, cc] = 1   # fibre
        label_img[rr, cc] = fid
        rr_ax, cc_ax = disk((cy, cx), 15, shape=(H, W))
        mask_sem[rr_ax, cc_ax] = 3  # axon inside fibre

    return image, mask_sem, label_img


def _write_dataset(in_root: Path, ds_name: str, images):
    """Write (image, mask, label) triples to the expected input directory layout."""
    ds_path = in_root / ds_name
    for sub in ("images", "masks", "labels"):
        (ds_path / sub).mkdir(parents=True, exist_ok=True)
    for i, (img, mask, lbl) in enumerate(images):
        stem = f"img{i:02d}"
        io.imsave(str(ds_path / "images" / f"{stem}.tif"), img)
        io.imsave(str(ds_path / "masks" / f"{stem}.tif"), mask)
        io.imsave(str(ds_path / "labels" / f"{stem}.tif"), lbl)


# ---------------------------------------------------------------------------
# 1. Split-count attribution (regression for variable-shadowing bug)
# ---------------------------------------------------------------------------

def test_tile_split_counts_per_image(tmp_path):
    """Each image's split counts in dataset_summary.tsv must sum to its tile count.

    Regression test for the shadowing bug where 'base_name' / 'dataset' inside the
    tile-saving tuple unpacking clobbered the outer loop variables, causing split
    counts to accumulate only on the last image's entry.
    """
    from axonpath.data_preparation.data_preparation import split_dataset

    in_root = tmp_path / "in"
    out_root = tmp_path / "out"

    # Two images in one dataset -> 1 tile each -> 2 tiles total
    images = [_make_synthetic_image(seed=i) for i in range(2)]
    _write_dataset(in_root, "ds1", images)

    split_dataset(str(in_root), str(out_root), create_test_split=False, seed=42)

    summary_path = out_root / "dataset_summary.tsv"
    assert summary_path.exists(), "dataset_summary.tsv was not created"

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    assert len(rows) == 2, f"Expected 2 summary rows (one per image), got {len(rows)}"

    for row in rows:
        n_tiles = int(row["NumTiles"])
        total_splits = int(row["TrainSplit"]) + int(row["ValSplit"]) + int(row["TestSplit"])
        assert total_splits == n_tiles, (
            f"Image '{row['ImageName']}': split counts ({total_splits}) != NumTiles ({n_tiles}). "
            "Check for variable-shadowing regression in tile-saving loop."
        )


# ---------------------------------------------------------------------------
# 2. Annotation QC: orphaned axon
# ---------------------------------------------------------------------------

def test_annotation_qc_detects_orphaned_axon():
    """_annotation_qc must flag an axon pixel that lies outside every fibre label."""
    from axonpath.data_preparation.data_preparation import _annotation_qc

    H, W = 64, 64
    filled_label = np.zeros((H, W), dtype=np.int32)
    filled_label[10:30, 10:30] = 1  # one fibre

    mask_sem = np.zeros((H, W), dtype=np.uint8)
    mask_sem[10:30, 10:30] = 1   # fibre semantic
    mask_sem[15:25, 15:25] = 3   # axon inside fibre (clean)
    mask_sem[50, 50] = 3          # orphaned axon pixel outside fibre

    issues = _annotation_qc(filled_label, mask_sem)
    issue_types = [t for t, _, _ in issues]
    assert "orphaned_axon" in issue_types, f"Expected orphaned_axon, got {issue_types}"


# ---------------------------------------------------------------------------
# 3. Annotation QC: fibre without axon
# ---------------------------------------------------------------------------

def test_annotation_qc_detects_fibre_without_axon():
    """_annotation_qc must flag a fibre instance that contains no axon pixels."""
    from axonpath.data_preparation.data_preparation import _annotation_qc

    H, W = 64, 64
    filled_label = np.zeros((H, W), dtype=np.int32)
    filled_label[10:30, 10:30] = 1  # fibre 1 — has axon
    filled_label[35:55, 35:55] = 2  # fibre 2 — no axon

    mask_sem = np.zeros((H, W), dtype=np.uint8)
    mask_sem[10:30, 10:30] = 1
    mask_sem[15:25, 15:25] = 3  # axon only in fibre 1
    mask_sem[35:55, 35:55] = 1  # fibre 2 labelled as fibre but no axon

    issues = _annotation_qc(filled_label, mask_sem)
    issue_types = [t for t, _, _ in issues]
    assert "fibre_without_axon" in issue_types, f"Expected fibre_without_axon, got {issue_types}"


# ---------------------------------------------------------------------------
# 4. Annotation QC: clean image produces no issues
# ---------------------------------------------------------------------------

def test_annotation_qc_clean_image():
    """_annotation_qc must return an empty list for a correctly annotated image."""
    from axonpath.data_preparation.data_preparation import _annotation_qc

    H, W = 64, 64
    filled_label = np.zeros((H, W), dtype=np.int32)
    filled_label[10:54, 10:54] = 1  # one fibre

    mask_sem = np.zeros((H, W), dtype=np.uint8)
    mask_sem[10:54, 10:54] = 1   # fibre
    mask_sem[25:40, 25:40] = 3   # axon inside fibre

    issues = _annotation_qc(filled_label, mask_sem)
    assert issues == [], f"Expected no QC issues for clean image, got {issues}"


# ---------------------------------------------------------------------------
# 5. Annotation QC: edge-touching fibres without axon are not flagged
# ---------------------------------------------------------------------------

def test_annotation_qc_edge_fibre_not_flagged():
    """Edge-touching fibres with no axon must not produce a fibre_without_axon warning.

    Fibres cut by the image border are partially annotated by design — the annotator
    cannot label structure outside the image, so missing axon is expected.
    """
    from axonpath.data_preparation.data_preparation import _annotation_qc

    H, W = 64, 64
    filled_label = np.zeros((H, W), dtype=np.int32)
    filled_label[0:20, 10:30] = 1   # fibre 1 — touches top edge, no axon
    filled_label[25:45, 25:45] = 2  # fibre 2 — interior, no axon (should warn)

    mask_sem = np.zeros((H, W), dtype=np.uint8)
    mask_sem[0:20, 10:30] = 1   # fibre semantic for fibre 1
    mask_sem[25:45, 25:45] = 1  # fibre semantic for fibre 2

    issues = _annotation_qc(filled_label, mask_sem)
    issue_types = [t for t, _, _ in issues]

    # Interior fibre 2 (no axon) must still be caught
    assert "fibre_without_axon" in issue_types, (
        f"Expected fibre_without_axon for interior fibre, got {issue_types}"
    )

    # Edge fibre 1 must NOT inflate the count
    no_axon_counts = [count for t, count, _ in issues if t == "fibre_without_axon"]
    assert no_axon_counts == [1], (
        f"Expected exactly 1 fibre_without_axon (interior only), got counts {no_axon_counts}"
    )
