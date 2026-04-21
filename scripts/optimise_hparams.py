"""
Hyperparameter optimisation for aimsegdl models.

Requires: pip install optuna  (development tool — not in environment.yml)

Usage examples
--------------
EM dataset:
    python scripts/optimise_hparams.py \
        --train_dir data/em/train_tiles \
        --val_dir   data/em/val_tiles   \
        --min_diameter 30               \
        --output_dir   hpo_em           \
        --n_trials 40 --proxy_epochs 100

BF dataset:
    python scripts/optimise_hparams.py \
        --train_dir data/bf/train_tiles \
        --val_dir   data/bf/val_tiles   \
        --min_diameter 10               \
        --output_dir   hpo_bf           \
        --n_trials 40 --proxy_epochs 100

Combined EM+BF model (use the lower min_diameter — BF default):
    python scripts/optimise_hparams.py \
        --train_dir data/combined/train_tiles \
        --val_dir   data/combined/val_tiles   \
        --output_dir hpo_combined             \
        --n_trials 40 --proxy_epochs 100

Resume an interrupted study:
    python scripts/optimise_hparams.py ... --resume

Output files (all in --output_dir)
-----------------------------------
study_summary.tsv        — all trials sorted by F1 (open in Excel / Sheets)
best_config.json         — TrainingConfig for the best trial, ready to use
parameter_importance.tsv — fANOVA importance scores (requires scikit-learn)
optimization_history.pdf — objective history + parallel coordinates plot
study.db                 — SQLite study file (resumable with --resume)
"""

import argparse
import torch
from aimsegdl.training.hpo import run_study


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimisation for aimsegdl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--train_dir", required=True,
                        help="Path to prepared train_tiles directory")
    parser.add_argument("--val_dir", required=True,
                        help="Path to prepared val_tiles directory")

    # Segmentation
    parser.add_argument("--min_diameter", type=float, default=10.0,
                        help="Minimum object diameter in pixels. Use the lower "
                             "value when running a combined EM+BF study "
                             "(BF default: 10.0, EM typically 30.0)")

    # Study
    parser.add_argument("--n_trials", type=int, default=40,
                        help="Total number of trials")
    parser.add_argument("--proxy_epochs", type=int, default=100,
                        help="Epochs per trial. ASHA prunes poor trials early; "
                             "only the best reach this ceiling.")
    parser.add_argument("--output_dir", default="hpo_results",
                        help="Directory for all output files and per-trial artefacts")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing study.db in --output_dir")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the Optuna sampler (reproducible trial order)")

    # Hardware
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu'. Auto-detected if omitted.")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_study(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        n_trials=args.n_trials,
        proxy_epochs=args.proxy_epochs,
        min_diameter=args.min_diameter,
        output_dir=args.output_dir,
        device=device,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
