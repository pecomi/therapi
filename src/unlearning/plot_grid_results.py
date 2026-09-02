"""Plot unlearned-versus-retrained metrics and epoch histories for experiment grids.

The script accepts both a flat grid (for example full/mini/center) and a
nested grid (for example small-LR/<learning-rate>/<run>).  It reads the
per-run representation-evaluation CSVs and the persisted ``ckpts/history.csv``
files; it does not rerun training, CSG2A, the predictor, or evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    ("task_gap", "Task gap |U - R|"),
    ("linear_cka", "Linear CKA: U vs R"),
    ("frechet_latent_distance", "Frechet distance: U vs R"),
)
COMPONENTS = ("recon", "emb_class", "exp_class", "center")
RUN_PATTERN = re.compile(
    r"(?P<step_mode>full|mini)_epoch_(?P<epochs>\d+)_recon_[^_]+_class_[^_]+_center_(?P<center>[^_]+)$"
)


def _parse_grid(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(
            "--grid must use LABEL=PATH, for example full_mini=run/ga_full_mini_center_grid_seed0"
        )
    return label, Path(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _run_configuration(run_dir: Path, run_name: str) -> dict[str, object]:
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

    match = RUN_PATTERN.fullmatch(Path(run_name).name)
    if not match:
        raise ValueError(
            f"cannot infer training configuration; expected {summary_path} or a standard "
            f"run name, got {run_name!r}"
        )
    center_tag = match.group("center").replace("p", ".").replace("m", "-")
    lr_match = re.search(r"(?:^|/)(?:full|mini)_lr_([^/]+)", run_name)
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
        configuration = _run_configuration(run_dir, run_name)
        losses = pd.read_csv(loss_path)
        similarities = pd.read_csv(similarity_path)

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
        histories.append(history)
    return rows, histories


def _configuration_label(row: pd.Series) -> str:
    lr = row["learning_rate"]
    lr_text = "unknown" if pd.isna(lr) else f"{float(lr):g}"
    return (
        f"{row['step_mode']} | lr={lr_text} | e={int(row['train_epochs'])} "
        f"| center={float(row['train_center_weight']):g}"
    )


def _plot_metrics(frame: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configurations = frame.drop_duplicates("run").copy()
    configurations["plot_label"] = configurations.apply(_configuration_label, axis=1)
    runs = configurations["run"].tolist()
    labels = configurations["plot_label"].tolist()
    figure, axes = plt.subplots(2, 3, figsize=(max(18, len(runs) * 1.1), 10))
    for row_index, assignment in enumerate(("forget", "retain")):
        values = frame[frame["assignment"] == assignment].set_index("run").loc[runs]
        for axis, (column, title) in zip(axes[row_index], METRICS):
            axis.bar(np.arange(len(runs)), values[column].to_numpy())
            axis.set_title(f"{assignment}: {title}")
            axis.set_xticks(np.arange(len(runs)), labels, rotation=45, ha="right", fontsize=8)
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_loss_components(history: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = history["run"].drop_duplicates().tolist()
    figure, axes = plt.subplots(2, 4, figsize=(24, 10), sharex=False)
    for row_index, assignment in enumerate(("forget", "retain")):
        for axis, component in zip(axes[row_index], COMPONENTS):
            column = f"{assignment}_{component}"
            for run_name in runs:
                curve = history[history["run"] == run_name].sort_values("epoch")
                axis.plot(curve["epoch"], curve[column], label=run_name, linewidth=1.4)
            axis.set(
                title=f"{assignment}: {component}",
                xlabel="ascent epoch",
                ylabel="epoch-end evaluation loss",
            )
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=7)
    figure.suptitle(title, y=1.01)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _metric_group_columns(grid_label: str) -> list[str]:
    """Keep only directly comparable configurations in one metrics panel."""
    if "full_mini" in grid_label:
        # Exactly two bars: full versus mini under an identical epoch/center setting.
        return ["train_epochs", "train_center_weight"]
    if "small_lr" in grid_label:
        # Compare LR/epoch choices within one update mode and center-loss condition.
        return ["step_mode", "train_center_weight"]
    return ["step_mode", "learning_rate", "train_center_weight"]


def _group_title(grid: str, columns: list[str], values: tuple[object, ...]) -> str:
    settings = ", ".join(f"{column}={value:g}" if isinstance(value, float) else f"{column}={value}" for column, value in zip(columns, values))
    return f"{grid}: {settings}"


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
        group_columns = _metric_group_columns(grid_label)
        for values, group in frame.groupby(group_columns, sort=True, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            tag = "_".join(f"{column}_{value:g}" if isinstance(value, float) else f"{column}_{value}" for column, value in zip(group_columns, values))
            _plot_metrics(
                group,
                output_dir / f"metrics_{_safe_name(grid_label)}_{_safe_name(tag)}.png",
                _group_title(grid_label, group_columns, values),
            )

    if histories:
        all_history = pd.concat(histories, ignore_index=True)
        all_history.to_csv(output_dir / "all_history.csv", index=False)
        for grid_label, frame in all_history.groupby("grid", sort=True):
            history_columns = ["step_mode", "learning_rate", "train_center_weight"]
            # Each requested epoch is a separate deterministic run.  Retain only the
            # longest run for a configuration, rather than drawing the same prefix
            # three or four times on top of itself.
            longest_epochs = frame.groupby(history_columns, dropna=False)["train_epochs"].transform("max")
            longest = frame[frame["train_epochs"] == longest_epochs]
            for values, group in longest.groupby(history_columns, sort=True, dropna=False):
                if not isinstance(values, tuple):
                    values = (values,)
                tag = "_".join(f"{column}_{value:g}" if isinstance(value, float) else f"{column}_{value}" for column, value in zip(history_columns, values))
                _plot_loss_components(
                    group,
                    output_dir / f"loss_components_{_safe_name(grid_label)}_{_safe_name(tag)}.png",
                    _group_title(grid_label, history_columns, values),
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
