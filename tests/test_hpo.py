"""
Smoke tests for the HPO pipeline (axonpath/training/hpo.py).

Run from the axonpath conda environment with optuna installed:
    pytest tests/test_hpo.py -v

All tests run on CPU with synthetic 64x64 tiles (_skip_size_check=True bypasses
the 512x512 validation — tiles of any size are valid for wiring tests).
"""

import csv
import json
import os

import numpy as np
import pytest

optuna = pytest.importorskip("optuna", reason="optuna not installed")

from axonpath.training.hpo import (
    _build_config,
    make_objective,
    run_study,
    write_outputs,
    _SUMMARY_COLUMNS,
)
from axonpath.training.config import TrainingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TILE_SIZE = 64   # small enough to keep tests fast on CPU
N_TRAIN = 4
N_VAL = 2


def _make_tile(size=TILE_SIZE):
    """Return a dict matching the .npy tile format saved by data_preparation."""
    return {
        "image": np.random.rand(size, size).astype(np.float32),
        "mask_sem": np.zeros((size, size), dtype=np.uint8),
        "mask_fibre": np.zeros((size, size), dtype=np.int32),
        "mask_axon": np.zeros((size, size), dtype=np.int32),
        "sdt_fibre": np.full((size, size), -1.0, dtype=np.float32),
        "sdt_axon": np.full((size, size), -1.0, dtype=np.float32),
    }


@pytest.fixture()
def tile_dirs(tmp_path):
    """Synthetic train/val tile directories with 64x64 tiles."""
    for split, n in [("train_tiles", N_TRAIN), ("val_tiles", N_VAL)]:
        d = tmp_path / split
        d.mkdir()
        for i in range(n):
            np.save(str(d / f"tile_{i:02d}.npy"), _make_tile())
    return str(tmp_path / "train_tiles"), str(tmp_path / "val_tiles")


def _small_dataset_kwargs():
    return {"_skip_size_check": True}


def _run_tiny_study(tile_dirs, tmp_path, n_trials=2, proxy_epochs=2):
    """Helper: run a minimal study and return (study, output_dir)."""
    train_dir, val_dir = tile_dirs
    out = str(tmp_path / "hpo_out")
    study = run_study(
        train_dir=train_dir,
        val_dir=val_dir,
        n_trials=n_trials,
        proxy_epochs=proxy_epochs,
        min_diameter=10.0,
        output_dir=out,
        device="cpu",
        seed=0,
        image_height=TILE_SIZE,
        image_width=TILE_SIZE,
        _dataset_kwargs=_small_dataset_kwargs(),
    )
    return study, out


# ---------------------------------------------------------------------------
# Test 1: search space bounds and types
# ---------------------------------------------------------------------------

def test_search_space_bounds_and_types():
    """All sampled hyperparameters must be within declared bounds and correct type."""
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    observed = {"learning_rate": [], "batch_size": [],
                "loss_weight_mse_fibre": [], "loss_weight_mse_axon": []}

    def _mock_objective(trial):
        # Reuse _build_config to sample — we don't train, just record
        config, norm_type = _build_config(
            trial,
            train_dir=".", val_dir=".", proxy_epochs=10,
            min_diameter=10.0, device="cpu", trial_output_dir="."
        )
        observed["learning_rate"].append(config.learning_rate)
        observed["batch_size"].append(config.batch_size)
        observed["loss_weight_mse_fibre"].append(config.loss_weights[1])
        observed["loss_weight_mse_axon"].append(config.loss_weights[2])
        return 0.0

    study.optimize(_mock_objective, n_trials=20)

    for lr in observed["learning_rate"]:
        assert isinstance(lr, float), f"learning_rate must be float, got {type(lr)}"
        assert 1e-5 <= lr <= 1e-2, f"learning_rate {lr} out of [1e-5, 1e-2]"

    for bs in observed["batch_size"]:
        assert bs in (2, 4, 8), f"batch_size {bs} not in {{2, 4, 8}}"

    for w in observed["loss_weight_mse_fibre"]:
        assert isinstance(w, float)
        assert 0.5 <= w <= 3.0, f"loss_weight_mse_fibre {w} out of [0.5, 3.0]"

    for w in observed["loss_weight_mse_axon"]:
        assert isinstance(w, float)
        assert 0.5 <= w <= 3.0, f"loss_weight_mse_axon {w} out of [0.5, 3.0]"


# ---------------------------------------------------------------------------
# Test 2: norm_type follows batch_size
# ---------------------------------------------------------------------------

def test_norm_type_follows_batch_size():
    """batch_size < 8 must produce norm_type='instance'; >= 8 must give 'batch'."""
    sampler = optuna.samplers.GridSampler({
        "learning_rate": [1e-3],
        "batch_size": [2, 4, 8],
        "loss_weight_mse_fibre": [1.0],
        "loss_weight_mse_axon": [1.0],
    })
    study = optuna.create_study(direction="maximize", sampler=sampler)

    norm_by_bs = {}

    def _mock(trial):
        _, norm_type = _build_config(
            trial, ".", ".", 10, 10.0, "cpu", "."
        )
        norm_by_bs[trial.params["batch_size"]] = norm_type
        return 0.0

    study.optimize(_mock, n_trials=3)

    assert norm_by_bs[2] == "instance", "batch_size=2 must use instance norm"
    assert norm_by_bs[4] == "instance", "batch_size=4 must use instance norm"
    assert norm_by_bs[8] == "batch",    "batch_size=8 must use batch norm"


# ---------------------------------------------------------------------------
# Test 3: LR scheduler is always disabled inside objective
# ---------------------------------------------------------------------------

def test_lr_scheduler_disabled_in_trials():
    """Every trial config must have use_lr_scheduler=False regardless of inputs."""
    sampler = optuna.samplers.RandomSampler(seed=1)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    scheduler_flags = []

    def _mock(trial):
        config, _ = _build_config(trial, ".", ".", 10, 10.0, "cpu", ".")
        scheduler_flags.append(config.use_lr_scheduler)
        return 0.0

    study.optimize(_mock, n_trials=10)

    assert all(not flag for flag in scheduler_flags), (
        "use_lr_scheduler must be False in every trial config"
    )


# ---------------------------------------------------------------------------
# Test 4: pruning path is reachable (TrialPruned raised when pruner fires)
# ---------------------------------------------------------------------------

def test_pruning_raises_trial_pruned(tile_dirs, tmp_path):
    """When the pruner decides to prune, TrialPruned must be raised by objective."""
    train_dir, val_dir = tile_dirs
    out = str(tmp_path / "prune_test")

    # AlwaysPruner prunes every trial after the first intermediate report
    class AlwaysPruner(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return len(trial.intermediate_values) >= 1

    study = optuna.create_study(
        direction="maximize",
        pruner=AlwaysPruner(),
    )

    objective = make_objective(
        train_dir, val_dir,
        proxy_epochs=5,
        min_diameter=10.0,
        device="cpu",
        output_dir=out,
        image_height=TILE_SIZE,
        image_width=TILE_SIZE,
        _dataset_kwargs=_small_dataset_kwargs(),
    )

    study.optimize(objective, n_trials=3)

    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) >= 1, (
        f"Expected at least 1 pruned trial with AlwaysPruner, got {len(pruned)}"
    )


# ---------------------------------------------------------------------------
# Test 5: objective runs end-to-end and returns a valid F1 score
# ---------------------------------------------------------------------------

def test_objective_returns_valid_f1(tile_dirs, tmp_path):
    """
    Full objective run: 2 trials x 2 epochs on synthetic data.
    Result must be a float in [0, 1].
    """
    study, _ = _run_tiny_study(tile_dirs, tmp_path, n_trials=2, proxy_epochs=2)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) >= 1, "At least 1 trial must complete"

    for t in completed:
        assert isinstance(t.value, float), f"Objective must be float, got {type(t.value)}"
        assert 0.0 <= t.value <= 1.0, f"Objective {t.value} not in [0, 1]"


# ---------------------------------------------------------------------------
# Test 6: output files are written with correct structure
# ---------------------------------------------------------------------------

def test_output_files_written(tile_dirs, tmp_path):
    """After study completes, all expected output files must exist and be valid."""
    _, out = _run_tiny_study(tile_dirs, tmp_path, n_trials=2, proxy_epochs=2)

    # study.db
    assert os.path.isfile(os.path.join(out, "study.db")), "study.db must exist"

    # study_summary.tsv — check existence and columns
    summary_path = os.path.join(out, "study_summary.tsv")
    assert os.path.isfile(summary_path), "study_summary.tsv must exist"
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
    assert header == _SUMMARY_COLUMNS, (
        f"study_summary.tsv columns mismatch.\n"
        f"  Expected: {_SUMMARY_COLUMNS}\n"
        f"  Got:      {header}"
    )

    # best_config.json — must be valid JSON with required TrainingConfig keys
    best_path = os.path.join(out, "best_config.json")
    assert os.path.isfile(best_path), "best_config.json must exist"
    with open(best_path) as f:
        cfg = json.load(f)

    required_keys = {
        "learning_rate", "batch_size", "loss_weights",
        "min_diameter", "use_lr_scheduler", "norm_type",
    }
    missing = required_keys - set(cfg.keys())
    assert not missing, f"best_config.json missing keys: {missing}"

    # LR scheduler must be False in saved best config
    assert cfg["use_lr_scheduler"] is False, (
        "best_config.json must record use_lr_scheduler=False "
        "(scheduler disabled during HPO)"
    )


# ---------------------------------------------------------------------------
# Test 7: study is resumable from study.db
# ---------------------------------------------------------------------------

def test_study_resumability(tile_dirs, tmp_path):
    """
    Run 2 trials, then resume and run 1 more.
    The resumed study must have 3 trials total.
    """
    train_dir, val_dir = tile_dirs
    out = str(tmp_path / "resume_test")

    # First run: 2 trials
    run_study(
        train_dir=train_dir, val_dir=val_dir,
        n_trials=2, proxy_epochs=2, min_diameter=10.0,
        output_dir=out, device="cpu", seed=0,
        image_height=TILE_SIZE, image_width=TILE_SIZE,
        _dataset_kwargs=_small_dataset_kwargs(),
    )

    n_after_first = len(optuna.load_study(
        study_name="axonpath_hpo",
        storage=f"sqlite:///{os.path.join(out, 'study.db')}",
    ).trials)
    assert n_after_first == 2, f"Expected 2 trials after first run, got {n_after_first}"

    # Resumed run: 1 more trial
    run_study(
        train_dir=train_dir, val_dir=val_dir,
        n_trials=1, proxy_epochs=2, min_diameter=10.0,
        output_dir=out, device="cpu", seed=0,
        resume=True,
        image_height=TILE_SIZE, image_width=TILE_SIZE,
        _dataset_kwargs=_small_dataset_kwargs(),
    )

    n_after_resume = len(optuna.load_study(
        study_name="axonpath_hpo",
        storage=f"sqlite:///{os.path.join(out, 'study.db')}",
    ).trials)
    assert n_after_resume == 3, (
        f"Expected 3 trials after resume, got {n_after_resume}"
    )
