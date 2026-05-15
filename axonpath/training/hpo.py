"""
Hyperparameter optimisation using Optuna with ASHA pruning.

Requires: pip install optuna  (not in environment.yml — development use only)

Search space
------------
learning_rate         log-uniform [1e-5, 1e-2]
batch_size            categorical {2, 4, 8}   (norm_type follows automatically)
loss_weight_mse_fibre uniform     [0.5, 3.0]  (CE weight fixed at 1.0)
loss_weight_mse_axon  uniform     [0.5, 3.0]

Intermediate value reported to the pruner: val_loss (each epoch).
Trial objective (maximised): mean F1 across fibre / axon / inner_cylinder on val set.
"""

import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import optuna
except ImportError as e:
    raise ImportError(
        "optuna is required for HPO. Install it with: pip install optuna"
    ) from e

from axonpath.training.config import TrainingConfig
from axonpath.transforms.custom_transforms import transforms_fn
from axonpath.utils.model_building import model_fn, get_loaders
from axonpath.utils.losses import make_dice_ce
from axonpath.training.train_loop import train_loop
from axonpath.evaluation.helpers import evaluate
from axonpath.dataset.axonpath_dataset import AxonPathDataset


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_config(trial, train_dir, val_dir, proxy_epochs, min_diameter,
                  device, trial_output_dir, fixed_batch_size=None):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    if fixed_batch_size is not None:
        batch_size = fixed_batch_size
    else:
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    mse_fibre_w = trial.suggest_float("loss_weight_mse_fibre", 0.5, 3.0)
    mse_axon_w = trial.suggest_float("loss_weight_mse_axon", 0.5, 3.0)
    ce_weight_ic = trial.suggest_float("ce_weight_ic", 1.0, 4.0, log=True)

    norm_type = "instance" if batch_size < 8 else "batch"

    return TrainingConfig(
        learning_rate=lr,
        batch_size=batch_size,
        num_epochs=proxy_epochs,
        num_workers=0,
        pin_memory=(device == "cuda"),
        device=device,
        min_diameter=min_diameter,
        train_dir=train_dir,
        val_dir=val_dir,
        use_lr_scheduler=False,
        loss_weights=(1.0, mse_fibre_w, mse_axon_w),
        ce_weight_ic=ce_weight_ic,
        output_dir=trial_output_dir,
    ), norm_type


def make_objective(train_dir, val_dir, proxy_epochs, min_diameter, device,
                   output_dir, image_height=512, image_width=512,
                   fixed_batch_size=None, _dataset_kwargs=None):
    """
    Return a closure that Optuna calls for each trial.

    _dataset_kwargs: extra kwargs forwarded to AxonPathDataset (e.g. _skip_size_check=True
                     for unit tests using sub-512 synthetic tiles).
    """
    if _dataset_kwargs is None:
        _dataset_kwargs = {}

    def objective(trial):
        config, norm_type = _build_config(
            trial, train_dir, val_dir, proxy_epochs, min_diameter,
            device, output_dir, fixed_batch_size=fixed_batch_size
        )

        # Store config so _write_outputs can read it later
        trial.set_user_attr("config_json", json.dumps({
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "loss_weight_mse_fibre": config.loss_weights[1],
            "loss_weight_mse_axon": config.loss_weights[2],
            "ce_weight_ic": config.ce_weight_ic,
            "norm_type": norm_type,
            "use_lr_scheduler": config.use_lr_scheduler,
        }))

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        model = model_fn(device, norm_type=norm_type)
        mse_loss = nn.MSELoss()
        dice_ce = make_dice_ce(config.ce_weight_ic, device)
        loss_fns = [mse_loss, dice_ce]
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scaler = (torch.amp.GradScaler("cuda") if device == "cuda"
                  else torch.amp.GradScaler("cpu"))

        train_tf, val_tf = transforms_fn(image_height, image_width)

        train_ds = AxonPathDataset(train_dir, transform=train_tf, cache=True, **_dataset_kwargs)
        val_ds = AxonPathDataset(val_dir, transform=val_tf, cache=True, **_dataset_kwargs)
        train_ds.populate_cache()
        val_ds.populate_cache()

        train_loader, val_loader = get_loaders(
            train_ds, val_ds, config.batch_size,
            num_workers=0, pin_memory=config.pin_memory
        )

        val_loss_final = float("inf")
        f1_fibre = f1_axon = f1_inner_cylinder = 0.0

        for epoch in range(config.num_epochs):
            train_loop(train_loader, model, optimizer, loss_fns, scaler,
                       device, config.loss_weights)

            val_loss_final, f1_fibre, f1_axon, f1_inner_cylinder, _ = evaluate(
                val_loader, model, loss_fns, device,
                min_diameter=config.min_diameter,
                loss_weights=config.loss_weights,
            )

            mean_f1 = (f1_fibre + f1_axon + f1_inner_cylinder) / 3.0
            trial.report(mean_f1, epoch)
            if trial.should_prune():
                trial.set_user_attr("epochs_run", epoch + 1)
                raise optuna.TrialPruned()

        trial.set_user_attr("f1_fibre", round(f1_fibre, 6))
        trial.set_user_attr("f1_axon", round(f1_axon, 6))
        trial.set_user_attr("f1_inner_cylinder", round(f1_inner_cylinder, 6))
        trial.set_user_attr("val_loss_final", round(val_loss_final, 6))
        trial.set_user_attr("epochs_run", config.num_epochs)

        return (f1_fibre + f1_axon + f1_inner_cylinder) / 3.0

    return objective


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    "trial_id", "state", "objective_f1", "val_loss_final", "epochs_run",
    "learning_rate", "batch_size", "loss_weight_mse_fibre", "loss_weight_mse_axon", "ce_weight_ic",
    "f1_fibre", "f1_axon", "f1_inner_cylinder",
]


def _trial_row(trial):
    params = trial.params
    ua = trial.user_attrs
    obj = trial.value if trial.value is not None else float("nan")
    return [
        trial.number,
        trial.state.name,
        round(obj, 6) if not np.isnan(obj) else "nan",
        ua.get("val_loss_final", "nan"),
        ua.get("epochs_run", "nan"),
        params.get("learning_rate", "nan"),
        params.get("batch_size", "nan"),
        params.get("loss_weight_mse_fibre", "nan"),
        params.get("loss_weight_mse_axon", "nan"),
        params.get("ce_weight_ic", "nan"),
        ua.get("f1_fibre", "nan"),
        ua.get("f1_axon", "nan"),
        ua.get("f1_inner_cylinder", "nan"),
    ]


def _write_summary(study, output_dir):
    completed_and_pruned = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    completed_and_pruned.sort(
        key=lambda t: t.value if t.value is not None else -1.0,
        reverse=True
    )

    path = os.path.join(output_dir, "study_summary.tsv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(_SUMMARY_COLUMNS)
        for t in completed_and_pruned:
            writer.writerow(_trial_row(t))


def _write_best_config(study, output_dir, proxy_epochs, min_diameter, device,
                       image_height, image_width, fixed_batch_size=None):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    best = max(completed, key=lambda t: t.value)
    batch_size = best.params.get("batch_size", fixed_batch_size)
    norm_type = "instance" if batch_size < 8 else "batch"

    config = TrainingConfig(
        learning_rate=best.params["learning_rate"],
        batch_size=batch_size,
        loss_weights=(1.0, best.params["loss_weight_mse_fibre"],
                      best.params["loss_weight_mse_axon"]),
        ce_weight_ic=best.params["ce_weight_ic"],
        min_diameter=min_diameter,
        device=device,
        image_height=image_height,
        image_width=image_width,
        output_dir=output_dir,
        use_lr_scheduler=False,
    )

    payload = {
        **{k: v for k, v in config.__dict__.items()},
        "norm_type": norm_type,
        "_hpo_trial_number": best.number,
        "_hpo_objective_f1": round(best.value, 6),
    }
    path = os.path.join(output_dir, "best_config.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_importance(study, output_dir):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        return  # fANOVA requires scikit-learn; skip silently if unavailable

    path = os.path.join(output_dir, "parameter_importance.tsv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["parameter", "importance_score"])
        for param, score in importance.items():
            writer.writerow([param, round(score, 6)])


def _write_plots(study, output_dir):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return

    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_parallel_coordinate,
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        plt.sca(axes[0])
        plot_optimization_history(study, target_name="mean F1")

        plt.sca(axes[1])
        plot_parallel_coordinate(study, target_name="mean F1")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "optimization_history.pdf"))
        plt.close(fig)
    except Exception:
        pass  # plotting is best-effort; never fail the study


def write_outputs(study, output_dir, proxy_epochs, min_diameter, device,
                  image_height=512, image_width=512, fixed_batch_size=None):
    """Write all result files. Safe to call after every trial."""
    os.makedirs(output_dir, exist_ok=True)
    _write_summary(study, output_dir)
    _write_best_config(study, output_dir, proxy_epochs, min_diameter, device,
                       image_height, image_width, fixed_batch_size=fixed_batch_size)
    _write_importance(study, output_dir)
    _write_plots(study, output_dir)


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(
    train_dir,
    val_dir,
    n_trials,
    proxy_epochs,
    min_diameter,
    output_dir,
    device=None,
    resume=False,
    seed=None,
    image_height=512,
    image_width=512,
    fixed_batch_size=None,
    _dataset_kwargs=None,
):
    """
    Run an Optuna HPO study.

    Args:
        train_dir: Path to prepared train_tiles directory.
        val_dir: Path to prepared val_tiles directory.
        n_trials: Total number of trials to run.
        proxy_epochs: Epochs per trial (all trials; ASHA prunes losers early).
        min_diameter: Fixed minimum object diameter (pixels) passed to evaluate().
        output_dir: Directory for all output files and per-trial model weights.
        device: 'cuda' or 'cpu'. Auto-detected if None.
        resume: If True, load existing study.db from output_dir and continue.
        seed: Seed for the Optuna sampler (reproducible trial sampling).
        image_height: Tile height (must match prepared data).
        image_width: Tile width (must match prepared data).
        _dataset_kwargs: Extra kwargs for AxonPathDataset (internal / test use).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "study.db")
    storage = f"sqlite:///{db_path}"
    study_name = "axonpath_hpo"

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=max(1, proxy_epochs // 3),   # start pruning after ~1/3 of proxy epochs
        reduction_factor=3,
        min_early_stopping_rate=0,
    )

    if resume:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )
        print(f"Resuming study '{study_name}' — "
              f"{len(study.trials)} trials already recorded.")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False,
        )

    objective = make_objective(
        train_dir, val_dir, proxy_epochs, min_diameter, device,
        output_dir, image_height, image_width, fixed_batch_size, _dataset_kwargs,
    )

    def _callback(study, trial):
        write_outputs(study, output_dir, proxy_epochs, min_diameter, device,
                      image_height, image_width, fixed_batch_size=fixed_batch_size)

    study.optimize(objective, n_trials=n_trials, callbacks=[_callback])

    # Final write (covers the last trial if callback already ran, idempotent)
    write_outputs(study, output_dir, proxy_epochs, min_diameter, device,
                  image_height, image_width, fixed_batch_size=fixed_batch_size)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        batch_size_str = str(best.params.get("batch_size", fixed_batch_size))
        print(f"\nBest trial #{best.number}: mean F1 = {best.value:.4f}")
        print(f"  learning_rate          = {best.params['learning_rate']:.2e}")
        print(f"  batch_size             = {batch_size_str}")
        print(f"  loss_weight_mse_fibre  = {best.params['loss_weight_mse_fibre']:.3f}")
        print(f"  loss_weight_mse_axon   = {best.params['loss_weight_mse_axon']:.3f}")
    else:
        print("\nNo trials completed (all pruned or failed).")

    return study
