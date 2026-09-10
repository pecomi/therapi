"""Shared full-set forget/retain loss-history output."""

from __future__ import annotations

import csv
import math
from pathlib import Path


_CORE_HISTORY_FIELDS = (
    "epoch",
    "train_objective",
    "evaluation_objective",
    "gradient_norm",
)
_PAIRED_METRICS = (
    "task",
    "recon",
    "emb_class",
    "exp_class",
    "center",
    "emb_accuracy",
    "exp_accuracy",
    "n_samples",
)


def split_metrics_row(
    epoch: int,
    forget: dict | None,
    retain: dict | None,
    *,
    train_objective: float | None = None,
    evaluation_objective: float | None = None,
    gradient_norm: float | None = None,
    **extra,
) -> dict:
    """Build the shared history row used by every aligner experiment.

    ``train_objective`` is the arithmetic mean of the scalar objectives passed
    to ``backward`` in this epoch. ``evaluation_objective`` is the same method
    objective recomputed from sample means over the complete, fixed
    forget/retain sets. Fields that do not apply to a method are left empty.
    """
    row = {
        "epoch": epoch,
        **extra,
        **({f"forget_{name}": value for name, value in forget.items()} if forget else {}),
        **({f"retain_{name}": value for name, value in retain.items()} if retain else {}),
    }
    for name, value in (
        ("train_objective", train_objective),
        ("evaluation_objective", evaluation_objective),
        ("gradient_norm", gradient_norm),
    ):
        if value is not None:
            row[name] = value
    return row


def format_epoch_log(row: dict, total_epochs: int) -> str:
    """Return one compact, method-independent console log line."""
    parts = [
        f"[epoch {int(row['epoch']):03d}/{total_epochs:03d}]",
    ]
    for key in (
        "train_objective",
        "evaluation_objective",
        "forget_task",
        "retain_task",
        "source_task",
        "gradient_norm",
    ):
        value = row.get(key)
        if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
            parts.append(f"{key}={value:.6f}")
    return " ".join(parts)


def write_history(history: list[dict], path: Path) -> None:
    """Write heterogeneous history rows without dropping later columns."""
    observed = list(dict.fromkeys(key for row in history for key in row))
    paired_fields = [
        f"{assignment}_{metric}"
        for metric in _PAIRED_METRICS
        for assignment in ("forget", "retain")
    ]
    fieldnames = [
        *[field for field in _CORE_HISTORY_FIELDS if field in observed],
        *[field for field in paired_fields if field in observed],
        *[
            field
            for field in observed
            if field not in _CORE_HISTORY_FIELDS and field not in paired_fields
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _plot_history(
    history: list[dict],
    path: Path,
    loss_scale: str,
    *,
    epoch_label: str,
    component_assignment: str,
) -> None:
    """Plot full-set task losses and one assignment's raw components."""
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
            [row[f"{component_assignment}_{name}"] for row in history],
            label=name,
        )
    axes[1].set(
        title=f"{component_assignment.capitalize()} target loss components",
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


def plot_history(
    history: list[dict],
    path: Path,
    loss_scale: str,
    *,
    epoch_label: str,
) -> None:
    """Plot full-set task losses with forget-loss components."""
    _plot_history(
        history,
        path,
        loss_scale,
        epoch_label=epoch_label,
        component_assignment="forget",
    )


def plot_retain_history(
    history: list[dict],
    path: Path,
    loss_scale: str,
    *,
    epoch_label: str,
) -> None:
    """Plot full-set task losses with retain-loss components."""
    _plot_history(
        history,
        path,
        loss_scale,
        epoch_label=epoch_label,
        component_assignment="retain",
    )
