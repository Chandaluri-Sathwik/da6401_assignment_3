"""
W&B experiment runner for Assignment 3 Part 2.

Run from the repository root:
    python part2_wandb/run_experiments.py --experiment all
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train import run_training_experiment


BASE_CONFIG = {
    "use_wandb": True,
    "batch_size": 64,
    "num_epochs": 10,
    "d_model": 512,
    "N": 4,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "warmup_steps": 4000,
    "min_freq": 2,
    "max_len": 100,
    "log_every": 100,
    "grad_log_steps": 1000,
}


def with_base(**overrides):
    config = dict(BASE_CONFIG)
    config.update(overrides)
    return config


EXPERIMENTS = {
    "noam_vs_fixed": [
        (
            "noam_scheduler",
            with_base(
                use_noam=True,
                checkpoint_path="checkpoints/noam_scheduler.pt",
            ),
        ),
        (
            "fixed_lr",
            with_base(
                use_noam=False,
                fixed_lr=1e-4,
                checkpoint_path="checkpoints/fixed_lr.pt",
            ),
        ),
    ],
    "scaling_ablation": [
        (
            "attention_scaled",
            with_base(
                use_attention_scaling=True,
                checkpoint_path="checkpoints/attention_scaled.pt",
            ),
        ),
        (
            "attention_unscaled",
            with_base(
                use_attention_scaling=False,
                checkpoint_path="checkpoints/attention_unscaled.pt",
            ),
        ),
    ],
    "attention_heatmaps": [
        (
            "attention_heatmaps",
            with_base(
                log_attention_heatmaps=True,
                checkpoint_path="checkpoints/attention_heatmaps.pt",
            ),
        ),
    ],
    "positional_ablation": [
        (
            "sinusoidal_positional_encoding",
            with_base(
                positional_encoding_type="sinusoidal",
                checkpoint_path="checkpoints/sinusoidal_positional_encoding.pt",
            ),
        ),
        (
            "learned_positional_encoding",
            with_base(
                positional_encoding_type="learned",
                checkpoint_path="checkpoints/learned_positional_encoding.pt",
            ),
        ),
    ],
    "label_smoothing": [
        (
            "label_smoothing_0_1",
            with_base(
                label_smoothing=0.1,
                checkpoint_path="checkpoints/label_smoothing_0_1.pt",
            ),
        ),
        (
            "label_smoothing_0_0",
            with_base(
                label_smoothing=0.0,
                checkpoint_path="checkpoints/label_smoothing_0_0.pt",
            ),
        ),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["all", *EXPERIMENTS.keys()],
        default="all",
        help="Which assignment Part 2 experiment to run.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help=(
            "Optional single run name inside the selected experiment, e.g. "
            "fixed_lr, attention_unscaled, learned_positional_encoding."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiment/run names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for experiment_name, runs in EXPERIMENTS.items():
            print(experiment_name)
            for run_name, _ in runs:
                print(f"  {run_name}")
        return

    selected = EXPERIMENTS.keys() if args.experiment == "all" else [args.experiment]

    for experiment_name in selected:
        for run_name, config in EXPERIMENTS[experiment_name]:
            if args.run is not None and args.run != run_name:
                continue
            print(f"Running {experiment_name}: {run_name}")
            run_training_experiment(
                config_overrides=config,
                run_name=run_name,
                group=experiment_name,
            )


if __name__ == "__main__":
    main()
