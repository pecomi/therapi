"""Create loss-curve plots restricted to the first ten ascent epochs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plot_unlearning_results import (
    _find_runs,
    _parse_experiment,
    _plot_loss_components,
    _plot_near_zero_losses,
    _safe_name,
)


def _plot_components_by_assignment(history: pd.DataFrame, path: Path, title: str) -> None:
    """Plot every raw component for both forget and retain in the early window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = history.loc[history["epoch"] == 0].iloc[0]
    curve = history[history["run"] == history["run"].iloc[0]].sort_values("epoch")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, component in zip(axes.ravel(), ("recon", "emb_class", "exp_class", "center")):
        for assignment, color in (("forget", "tab:blue"), ("retain", "tab:orange")):
            axis.plot(
                curve["epoch"], curve[f"{assignment}_{component}"], color=color,
                label=f"{assignment}",
            )
            axis.axhline(
                reference[f"baseline_{assignment}_{component}"], color=color,
                linestyle=":", label=f"baseline {assignment}",
            )
            axis.axhline(
                reference[f"retrained_{assignment}_{component}"], color=color,
                linestyle="--", label=f"retrained {assignment}",
            )
        axis.set(title=f"{component} (raw)", xlabel="ascent epoch", ylabel="loss")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle(f"{title}: forget and retain raw loss components", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment_label, experiment_root in args.experiment:
        if not experiment_root.is_dir():
            raise FileNotFoundError(f"experiment directory does not exist: {experiment_root}")
        _, histories = _find_runs(experiment_label, experiment_root, args.unit)
        if not histories:
            print(f"[warning] no history.csv files found under {experiment_root}")
            continue

        history = pd.concat(histories, ignore_index=True)
        for columns, group in history.groupby(
            ["step_mode", "learning_rate", "train_center_weight", "unlearn_seed"],
            sort=True,
            dropna=False,
        ):
            if not isinstance(columns, tuple):
                columns = (columns,)
            shown = group[group["epoch"] <= args.max_epoch].copy()
            if shown.empty:
                continue
            if int(group["epoch"].max()) < args.max_epoch:
                print(
                    f"[warning] {experiment_label} has only {int(group['epoch'].max())} epochs; "
                    f"plotting all available rows"
                )
            tag = "_".join(
                f"{name}_{value:g}" if isinstance(value, float) else f"{name}_{value}"
                for name, value in zip(
                    ["step_mode", "learning_rate", "train_center_weight", "unlearn_seed"], columns
                )
            )
            title = f"{experiment_label}: {tag} (epochs 0–{args.max_epoch})"
            prefix = f"first_{args.max_epoch}_epochs_{_safe_name(experiment_label)}_{_safe_name(tag)}"
            _plot_loss_components(shown, output_dir / f"{prefix}.png", title, args.loss_scale)
            _plot_near_zero_losses(
                shown,
                output_dir / f"{prefix}_near_zero.png",
                title,
            )
            _plot_components_by_assignment(
                shown,
                output_dir / f"{prefix}_components_forget_retain.png",
                title,
            )

    print(f"[done] first-{args.max_epoch}-epoch plots -> {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=_parse_experiment, action="append", required=True,
        help="LABEL=PATH; repeat once per result directory",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--unit", choices=("patient", "sample"), default="patient")
    parser.add_argument("--loss-scale", choices=("linear", "log", "symlog"), default="log")
    parsed = parser.parse_args()
    if parsed.max_epoch < 0:
        parser.error("--max-epoch must be non-negative")
    main(parsed)
