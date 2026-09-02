"""Plot the fixed mini-batch center-loss ablation from saved CSVs and histories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    ("task_difference", "Task loss difference: U - R"),
    ("linear_cka", "Linear CKA: U vs R"),
    ("frechet_latent_distance", "Frechet distance: U vs R"),
)
COMPONENTS = ("recon", "emb_class", "exp_class", "center")
RUN_PATTERN = re.compile(
    r"(?P<step_mode>mini)_epoch_(?P<epochs>\d+).*_center_(?P<center>[^_]+)$"
)


def _parse_grid(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(
            "--grid must use LABEL=PATH, for example mini_center=run/ga_mini_center_lr1e4_epoch30_seed0"
        )
    return label, Path(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _summary_configuration(run_dir: Path, grid_root: Path) -> dict[str, object]:
    """Find the nearest grid summary and recover a row for ``run_dir``."""
    for directory in (run_dir, *run_dir.parents):
        try:
            directory.relative_to(grid_root)
        except ValueError:
            break
        summary_path = directory / "comparison_summary.csv"
        if not summary_path.is_file():
            continue
        summary = pd.read_csv(summary_path)
        match = summary[summary["run"] == run_dir.name]
        if len(match) == 1:
            row = match.iloc[0]
            return {
                "step_mode": str(row["step_mode"]),
                "train_epochs": int(row["epochs"]),
                "learning_rate": float(row["learning_rate"]),
                "train_center_weight": float(row["train_center_weight"]),
            }
    return {}


def _run_configuration(run_dir: Path, run_name: str, grid_root: Path) -> dict[str, object]:
    """Read exact training settings, with a run-name fallback for old runs."""
    summary_path = run_dir / "ckpts" / "summary.json"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        config = summary.get("unlearning_config", {})
        required = {"step_mode", "epochs", "lr", "center_weight"}
        if required <= set(config):
            return {
                "step_mode": str(config["step_mode"]),
                "train_epochs": int(config["epochs"]),
                "learning_rate": float(config["lr"]),
                "train_center_weight": float(config["center_weight"]),
            }

    summary_config = _summary_configuration(run_dir, grid_root)
    if summary_config:
        return summary_config

    match = RUN_PATTERN.fullmatch(Path(run_name).name)
    if not match:
        raise ValueError(
            f"cannot infer training configuration; expected {summary_path} or a standard "
            f"run name, got {run_name!r}"
        )
    center_tag = match.group("center").replace("p", ".").replace("m", "-")
    lr_match = re.search(r"(?:^|/)mini_lr_([^/]+)", run_name)
    learning_rate = float("nan")
    if lr_match:
        learning_rate = float(lr_match.group(1).replace("p", "."))
    return {
        "step_mode": match.group("step_mode"),
        "train_epochs": int(match.group("epochs")),
        "learning_rate": learning_rate,
        "train_center_weight": float(center_tag),
    }


def _one_value(
    frame: pd.DataFrame, *, assignment: str, model: str | None, comparison: str | None,
    column: str, unit: str,
) -> float:
    mask = (frame["unit"] == unit) & (frame["assignment"] == assignment)
    if model is not None:
        mask &= frame["model"] == model
    if comparison is not None:
        mask &= frame["comparison"] == comparison
    values = frame.loc[mask, column]
    if len(values) != 1:
        raise ValueError(
            f"expected one {column} value for unit={unit}, assignment={assignment}, "
            f"model={model}, comparison={comparison}; found {len(values)}"
        )
    return float(values.iloc[0])


def _find_runs(grid_label: str, grid_root: Path, unit: str) -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    histories: list[pd.DataFrame] = []
    evaluation_files = sorted(grid_root.rglob("evaluation/representations/loss_metrics.csv"))
    if not evaluation_files:
        raise FileNotFoundError(f"no evaluation/representations/loss_metrics.csv below {grid_root}")

    for loss_path in evaluation_files:
        representation_dir = loss_path.parent
        run_dir = representation_dir.parent.parent
        similarity_path = representation_dir / "representation_similarity.csv"
        history_path = run_dir / "ckpts" / "history.csv"
        if not similarity_path.is_file():
            raise FileNotFoundError(f"missing representation similarity CSV: {similarity_path}")
        run_name = str(run_dir.relative_to(grid_root))
        configuration = _run_configuration(run_dir, run_name, grid_root)
        losses = pd.read_csv(loss_path)
        similarities = pd.read_csv(similarity_path)
        reference_losses = {
            f"{model}_{assignment}_{loss_name}": _one_value(
                losses,
                assignment=assignment,
                model=model,
                comparison=None,
                column=loss_name,
                unit=unit,
            )
            for model in ("baseline", "retrained")
            for assignment in ("forget", "retain")
            for loss_name in ("task", *COMPONENTS)
        }

        for assignment in ("forget", "retain"):
            unlearned_task = _one_value(
                losses, assignment=assignment, model="unlearned", comparison=None,
                column="task", unit=unit,
            )
            retrained_task = _one_value(
                losses, assignment=assignment, model="retrained", comparison=None,
                column="task", unit=unit,
            )
            rows.append(
                {
                    "grid": grid_label,
                    "run": run_name,
                    **configuration,
                    "assignment": assignment,
                    "task_gap": abs(unlearned_task - retrained_task),
                    "task_difference": unlearned_task - retrained_task,
                    "unlearned_task": unlearned_task,
                    "retrained_task": retrained_task,
                    "linear_cka": _one_value(
                        similarities, assignment=assignment, model=None,
                        comparison="unlearned_vs_retrained", column="linear_cka", unit=unit,
                    ),
                    "frechet_latent_distance": _one_value(
                        similarities, assignment=assignment, model=None,
                        comparison="unlearned_vs_retrained",
                        column="frechet_latent_distance", unit=unit,
                    ),
                }
            )

        if not history_path.is_file():
            print(f"[warning] no history log; loss curves skip this run: {history_path}")
            continue
        history = pd.read_csv(history_path)
        required = {"epoch"} | {f"{group}_{component}" for group in ("forget", "retain") for component in COMPONENTS}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"history log missing {sorted(missing)}: {history_path}")
        history.insert(0, "run", run_name)
        history.insert(0, "grid", grid_label)
        for column, value in reversed(tuple(configuration.items())):
            history.insert(2, column, value)
        for column, value in reversed(tuple(reference_losses.items())):
            history.insert(2, column, value)
        histories.append(history)
    return rows, histories


def _plot_paired_metric(
    axis, frame: pd.DataFrame, *, metric: str, labels: list[str], run_order: list[str]
) -> None:
    values = frame.set_index(["run", "assignment"])[metric]
    x = np.arange(len(run_order))
    width = 0.38
    for offset, assignment, color in ((-width / 2, "forget", "tab:blue"), (width / 2, "retain", "tab:orange")):
        y = [values.loc[(run, assignment)] for run in run_order]
        axis.bar(x + offset, y, width=width, label=assignment, color=color)
    axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    if metric == "task_difference":
        axis.axhline(0, color="black", linewidth=0.9)
    axis.grid(axis="y", alpha=0.25)


def _plot_center_metrics(frame: pd.DataFrame, path_prefix: Path, title: str) -> None:
    """Plot a fixed-LR/fixed-epoch center-loss ablation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configurations = frame.drop_duplicates("run").sort_values("train_center_weight")
    run_order = configurations["run"].tolist()
    labels = [f"center={weight:g}" for weight in configurations["train_center_weight"]]
    for metric, metric_title in METRICS:
        figure, axis = plt.subplots(figsize=(max(7, len(run_order) * 1.5), 4.5))
        _plot_paired_metric(axis, frame, metric=metric, labels=labels, run_order=run_order)
        axis.set_title(metric_title)
        axis.legend()
        figure.suptitle(title, y=1.01)
        figure.tight_layout()
        figure.savefig(path_prefix.with_name(f"{path_prefix.name}_{metric}.png"), dpi=180)
        plt.close(figure)


def _plot_loss_components(history: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = history["run"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for run_name in runs:
        curve = history[history["run"] == run_name].sort_values("epoch")
        axes[0].plot(curve["epoch"], curve["forget_task"], label="forget", color="tab:blue")
        axes[0].plot(curve["epoch"], curve["retain_task"], label="retain", color="tab:orange")
        for component, color in zip(COMPONENTS, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
            axes[1].plot(curve["epoch"], curve[f"forget_{component}"], label=component, color=color)
    reference = history.iloc[0]
    axes[0].axhline(
        reference["baseline_forget_task"], color="tab:blue", linestyle=":",
        label="baseline forget (final)",
    )
    axes[0].axhline(
        reference["baseline_retain_task"], color="tab:orange", linestyle=":",
        label="baseline retain (final)",
    )
    axes[0].axhline(
        reference["retrained_forget_task"], color="tab:blue", linestyle="--",
        label="retrained forget (final)",
    )
    axes[0].axhline(
        reference["retrained_retain_task"], color="tab:orange", linestyle="--",
        label="retrained retain (final)",
    )
    for component, color in zip(COMPONENTS, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        axes[1].axhline(
            reference[f"baseline_forget_{component}"], color=color, linestyle=":",
            label=f"baseline {component} (final)",
        )
        axes[1].axhline(
            reference[f"retrained_forget_{component}"], color=color, linestyle="--",
            label=f"retrained {component} (final)",
        )
    axes[0].set(title="Mean alignment loss", xlabel="ascent epoch", ylabel="loss")
    axes[1].set(title="Forget loss components", xlabel="ascent epoch", ylabel="loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title, y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    histories: list[pd.DataFrame] = []
    for grid_label, grid_root in args.grid:
        if not grid_root.is_dir():
            raise FileNotFoundError(f"grid directory does not exist: {grid_root}")
        rows, grid_histories = _find_runs(grid_label, grid_root, args.unit)
        metric_rows.extend(rows)
        histories.extend(grid_histories)

    metrics = pd.DataFrame(metric_rows).sort_values(["grid", "run", "assignment"])
    metrics.to_csv(output_dir / "unlearned_vs_retrained_metrics.csv", index=False)
    for grid_label, frame in metrics.groupby("grid", sort=True):
        _plot_center_metrics(
            frame,
            output_dir / f"metrics_{_safe_name(grid_label)}",
            grid_label,
        )

    if histories:
        all_history = pd.concat(histories, ignore_index=True)
        all_history.to_csv(output_dir / "all_history.csv", index=False)
        for grid_label, frame in all_history.groupby("grid", sort=True):
            history_columns = ["step_mode", "learning_rate", "train_center_weight"]
            for values, group in frame.groupby(history_columns, sort=True, dropna=False):
                if not isinstance(values, tuple):
                    values = (values,)
                tag = "_".join(f"{column}_{value:g}" if isinstance(value, float) else f"{column}_{value}" for column, value in zip(history_columns, values))
                _plot_loss_components(
                    group,
                    output_dir / f"loss_components_{_safe_name(grid_label)}_{_safe_name(tag)}.png",
                    f"{grid_label}: {tag}",
                )
    else:
        print("[warning] no history.csv files were found; no loss-component plots were written")
    print(f"[done] plots and merged CSVs -> {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=_parse_grid, action="append", required=True,
        help="LABEL=PATH; repeat once per experiment grid",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unit", choices=("patient", "sample"), default="patient")
    main(parser.parse_args())
