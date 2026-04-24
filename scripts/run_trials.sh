#!/bin/bash
# Sequential runner for the 6 long-run trials.
# Each trial's verbose output goes to long_runs/<name>/stdout.log.
# Only START/DONE/FAILED lines are printed to stdout (monitored by Claude).
# Git-excluded: one-off script, not part of the package.

PYTHON="$HOME/anaconda3/envs/axonpath/python.exe"
SCRIPT="scripts/long_run_training.py"

run_trial() {
    local name=$1; shift
    mkdir -p "long_runs/${name}"
    echo "=== START: $name ==="
    PYTHONPATH=. "$PYTHON" "$SCRIPT" "$@" > "long_runs/${name}/stdout.log" 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "=== DONE: $name ==="
    else
        echo "=== FAILED: $name (exit $exit_code) ==="
    fi
}

run_trial em_trial12 \
    --study_db hpo_em/study.db --trial_number 12 \
    --train_dir prepared_data_em/train_tiles --val_dir prepared_data_em/val_tiles \
    --run_name em_trial12 --output_dir long_runs --min_diameter 30

run_trial em_trial9 \
    --study_db hpo_em/study.db --trial_number 9 \
    --train_dir prepared_data_em/train_tiles --val_dir prepared_data_em/val_tiles \
    --run_name em_trial9 --output_dir long_runs --min_diameter 30

run_trial em_trial33 \
    --study_db hpo_em/study.db --trial_number 33 \
    --train_dir prepared_data_em/train_tiles --val_dir prepared_data_em/val_tiles \
    --run_name em_trial33 --output_dir long_runs --min_diameter 30

run_trial bf_trial12 \
    --study_db hpo_bf/study.db --trial_number 12 \
    --train_dir prepared_data_bf/train_tiles --val_dir prepared_data_bf/val_tiles \
    --run_name bf_trial12 --output_dir long_runs --min_diameter 10

run_trial bf_trial38 \
    --study_db hpo_bf/study.db --trial_number 38 \
    --train_dir prepared_data_bf/train_tiles --val_dir prepared_data_bf/val_tiles \
    --run_name bf_trial38 --output_dir long_runs --min_diameter 10

run_trial bf_trial9 \
    --study_db hpo_bf/study.db --trial_number 9 \
    --train_dir prepared_data_bf/train_tiles --val_dir prepared_data_bf/val_tiles \
    --run_name bf_trial9 --output_dir long_runs --min_diameter 10

echo "=== ALL TRIALS COMPLETE ==="
