"""
Tests for axonpath.training.train.

Run from the axonpath conda environment:
    pytest tests/test_training.py -v
"""

import os

import numpy as np


def _write_tiles(tile_dir, n, size=512):
    """Write n minimal prepared tiles (the format written by split_dataset)."""
    os.makedirs(tile_dir)
    rng = np.random.default_rng(0)
    for i in range(n):
        tile = {
            "image": rng.uniform(0, 1, (size, size)).astype(np.float32),
            "mask_sem": np.zeros((size, size), dtype=np.uint8),
            "mask_fibre": np.zeros((size, size), dtype=np.int32),
            "mask_axon": np.zeros((size, size), dtype=np.int64),
            "sdt_fibre": np.full((size, size), -1.0),
            "sdt_axon": np.full((size, size), -1.0),
        }
        np.save(os.path.join(tile_dir, f"tile{i}.npy"), tile)


# ---------------------------------------------------------------------------
# 1. Best weights are saved even if the validation score never rises above 0
# ---------------------------------------------------------------------------

def test_train_saves_best_weights_when_score_stays_zero(tmp_path, monkeypatch):
    """A run whose balanced segmentation score is 0 in every epoch (e.g. a very short
    run) must still write best_weights_model.pth and finish, instead of failing when
    the TorchScript export loads the best weights."""
    import importlib
    from axonpath.training.config import TrainingConfig

    # The module, not the train() function that axonpath.training re-exports under the same name
    train_module = importlib.import_module("axonpath.training.train")

    train_dir = tmp_path / "train_tiles"
    val_dir = tmp_path / "val_tiles"
    _write_tiles(train_dir, 2)
    _write_tiles(val_dir, 1)

    # Validation always scores 0: (val_loss, f1_fibre, f1_axon, f1_inner_cylinder, dice)
    monkeypatch.setattr(train_module, "evaluate", lambda *args, **kwargs: (1.0, 0.0, 0.0, 0.0, 0.0))

    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        device="cpu",
        pin_memory=False,
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        output_dir=str(tmp_path / "out"),
        model_name="zero_score",
    )
    os.makedirs(config.output_dir)
    train_module.train(config)

    assert (tmp_path / "out" / "best_weights_model.pth").is_file()
    assert (tmp_path / "out" / "zero_score.pt").is_file()
