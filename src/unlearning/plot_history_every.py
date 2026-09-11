"""Plot a sparse, report-friendly view of an unlearning history CSV.

The training scripts continue to write every epoch to ``history.csv``.  This
utility selects epoch 0, every Nth epoch, and the final epoch, then writes
forget- and retain-component curves without changing the original curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = (
    "epoch",
    "forget_task",
    "retain_task",
)
COMPONENTS = ("recon", "emb_class", "exp_class", "center")


def _select_epochs(history: pd.DataFrame, every: int) -> pd.DataFrame:
    """Keep epoch 0, each requested interval, and the last completed epoch."""
    if every <= 0:
        raise ValueError("--every must be positive")
    if history.empty:
        raise ValueError("history CSV contains no rows")

    history = history.sort_values("epoch").drop_duplicates("epoch", keep="last")
    last_epoch = int(history["epoch"].iloc[-1])
    selected = history[
        (history["epoch"] == 0)
        | (history["epoch"] % every == 0)
        | (history["epoch"] == last_epoch)
    ]
    return selected.copy()


def _plot(
    history: pd.DataFrame,
    output_path: Path,
    *,
    component_assignment: str,
    loss_scale: str,
    every: int,
) -> None:
    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["forget_task"], marker="o", label="forget")
    axes[0].plot(epochs, history["retain_task"], marker="o", label="retain")
    axes[0].set(
        title=f"Mean target alignment loss (every {every} epochs)",
        xlabel="epoch",
        ylabel="loss",
    )
    axes[0].legend()

    for component in COMPONENTS:
        column = f"{component_assignment}_{component}"
        if column in history:
            axes[1].plot(epochs, history[column], marker="o", label=component)
    axes[1].set(
        title=f"{component_assignment.capitalize()} loss components",
        xlabel="epoch",
        ylabel="loss",
    )
    axes[1].legend()

    for axis in axes:
        if loss_scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-2)
        else:
            axis.set_yscale(loss_scale)
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_history_every(args: argparse.Namespace) -> None:
    history_path = Path(args.history)
    output_dir = Path(args.output_dir) if args.output_dir else history_path.parent
    history = pd.read_csv(history_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in history]
    if missing:
        raise ValueError(f"history CSV is missing required columns: {missing}")

    selected = _select_epochs(history, args.every)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"every{args.every}"
    _plot(
        selected,
        output_dir / f"loss_curve_{suffix}.png",
        component_assignment="forget",
        loss_scale=args.loss_scale,
        every=args.every,
    )
    _plot(
        selected,
        output_dir / f"retain_loss_curve_{suffix}.png",
        component_assignment="retain",
        loss_scale=args.loss_scale,
        every=args.every,
    )
    print(f"[data] selected_epochs={selected['epoch'].tolist()}")
    print(f"[done] {output_dir / f'loss_curve_{suffix}.png'}")
    print(f"[done] {output_dir / f'retain_loss_curve_{suffix}.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, help="path to history.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for plots; defaults to the history.csv directory",
    )
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument(
        "--loss-scale", choices=("linear", "log", "symlog"), default="log"
    )
    plot_history_every(parser.parse_args())
