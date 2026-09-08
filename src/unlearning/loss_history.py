"""Shared full-set forget/retain loss-history output."""

from __future__ import annotations

import csv
from pathlib import Path


def split_metrics_row(epoch: int, forget: dict, retain: dict, **extra) -> dict:
    """Build one standardized post-update (or epoch-zero) history row."""
    return {
        "epoch": epoch,
        **extra,
        **{f"forget_{name}": value for name, value in forget.items()},
        **{f"retain_{name}": value for name, value in retain.items()},
    }


def write_history(history: list[dict], path: Path) -> None:
    """Write heterogeneous history rows without dropping later columns."""
    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(
    history: list[dict],
    path: Path,
    loss_scale: str,
    *,
    epoch_label: str,
) -> None:
    """Plot the same full-set target losses used by gradient_ascent.py."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["forget_task"] for row in history], label="forget")
    axes[0].plot(epochs, [row["retain_task"] for row in history], label="retain")
    axes[0].set(
        title="Mean target alignment loss",
        xlabel=epoch_label,
        ylabel="loss",
    )
    axes[0].legend()
    for name in ("recon", "emb_class", "exp_class", "center"):
        axes[1].plot(
            epochs,
            [row[f"forget_{name}"] for row in history],
            label=name,
        )
    axes[1].set(
        title="Forget target loss components",
        xlabel=epoch_label,
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
    figure.savefig(path, dpi=180)
    plt.close(figure)
