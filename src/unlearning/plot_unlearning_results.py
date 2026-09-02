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
    r"(?P<step_mode>mini)_epoch_(?P<epochs>\d+).*_center_(?P<center>[^_]+)"
    r"(?:_unlearn_seed_(?P<unlearn_seed>\d+))?$"
)


def _parse_experiment(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(
            "--experiment must use LABEL=PATH, for example mini_center=run/unlearn_mini_center0p8_seed_replicates"
        )
    return label, Path(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _summary_configuration(run_dir: Path, experiment_root: Path) -> dict[str, object]:
    """Find the nearest experiment summary and recover a row for ``run_dir``."""
    for directory in (run_dir, *run_dir.parents):
        try:
            directory.relative_to(experiment_root)
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
                "unlearn_seed": int(row.get("unlearn_seed", 0)),
            }
    return {}


def _run_configuration(run_dir: Path, run_name: str, experiment_root: Path) -> dict[str, object]:
    """Read exact training settings, with a run-name fallback for old runs."""
    summary_path = run_dir / "ckpts" / "summary.json"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        config = summary.get("unlearning_config", {})
        required = {"step_mode", "epochs", "lr", "center_weight", "unlearn_seed"}
        if required <= set(config):
            return {
                "step_mode": str(config["step_mode"]),
                "train_epochs": int(config["epochs"]),
                "learning_rate": float(config["lr"]),
                "train_center_weight": float(config["center_weight"]),
                "unlearn_seed": int(config["unlearn_seed"]),
            }

    summary_config = _summary_configuration(run_dir, experiment_root)
    if summary_config:
        return summary_config

    match = RUN_PATTERN.fullmatch(Path(run_name).name)
    if not match:
        raise ValueError(
            f"cannot infer training configuration; expected {summary_path} or a standard "
            f"run name, got {run_name!r}"
        )
    center_tag = match.group("center").replace("p", ".").replace("m", "-")
    # Old result roots used either ``mini_lr_0p0001`` or ``lr1e4``.  Prefer
    # the per-run tag because it is unambiguous and survives a renamed root.
    lr_match = re.search(r"_lr_(?P<lr>[0-9pm]+)_center_", run_name)
    if not lr_match:
        lr_match = re.search(r"(?:^|/)mini_lr_(?P<lr>[^/]+)", run_name)
    learning_rate = float("nan")
    if lr_match:
        learning_rate = float(lr_match.group("lr").replace("p", ".").replace("m", "-"))
    return {
        "step_mode": match.group("step_mode"),
        "train_epochs": int(match.group("epochs")),
        "learning_rate": learning_rate,
        "train_center_weight": float(center_tag),
        "unlearn_seed": int(match.group("unlearn_seed") or 0),
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


def _find_runs(experiment_label: str, experiment_root: Path, unit: str) -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    histories: list[pd.DataFrame] = []
    evaluation_files = sorted(experiment_root.rglob("evaluation/representations/loss_metrics.csv"))
    if not evaluation_files:
        raise FileNotFoundError(f"no evaluation/representations/loss_metrics.csv below {experiment_root}")

    for loss_path in evaluation_files:
        representation_dir = loss_path.parent
        run_dir = representation_dir.parent.parent
        similarity_path = representation_dir / "representation_similarity.csv"
        history_path = run_dir / "ckpts" / "history.csv"
        if not similarity_path.is_file():
            raise FileNotFoundError(f"missing representation similarity CSV: {similarity_path}")
        run_name = str(run_dir.relative_to(experiment_root))
        configuration = _run_configuration(run_dir, run_name, experiment_root)
        losses = pd.read_csv(loss_path)
        similarities = pd.read_csv(similarity_path)
        # ``history.csv`` is the sample-weighted objective used by gradient
        # ascent.  Its reference lines must therefore also be sample means.
        # The metrics below may still use ``unit`` (patient by default).
        reference_losses = {
            f"{model}_{assignment}_{loss_name}": _one_value(
                losses,
                assignment=assignment,
                model=model,
                comparison=None,
                column=loss_name,
                unit="sample",
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
                    "experiment": experiment_label,
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
        epoch_zero = history.loc[history["epoch"] == 0]
        if len(epoch_zero) != 1:
            raise ValueError(f"history must contain exactly one epoch=0 row: {history_path}")
        # This is an invariant: gradient ascent begins from the baseline
        # checkpoint, so its sample-mean epoch 0 must equal the separately
        # evaluated baseline loss.  Do not silently draw incomparable curves.
        for assignment in ("forget", "retain"):
            for loss_name in ("task", *COMPONENTS):
                logged = float(epoch_zero.iloc[0][f"{assignment}_{loss_name}"])
                evaluated = reference_losses[f"baseline_{assignment}_{loss_name}"]
                if not np.isclose(logged, evaluated, rtol=1e-5, atol=1e-7):
                    raise ValueError(
                        "baseline loss mismatch between history epoch 0 and "
                        f"evaluation ({assignment=}, {loss_name=}, "
                        f"history={logged:.8g}, evaluation={evaluated:.8g}). "
                        "Re-run representation evaluation with the same checkpoint, "
                        "split, loss weights, and source/target data."
                    )
        history.insert(0, "run", run_name)
        history.insert(0, "experiment", experiment_label)
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

    configurations = frame.drop_duplicates("run").sort_values("unlearn_seed")
    run_order = configurations["run"].tolist()
    labels = [f"seed={seed}" for seed in configurations["unlearn_seed"]]
    for metric, metric_title in METRICS:
        figure, axis = plt.subplots(figsize=(max(7, len(run_order) * 1.5), 4.5))
        _plot_paired_metric(axis, frame, metric=metric, labels=labels, run_order=run_order)
        axis.set_title(metric_title)
        axis.legend()
        figure.suptitle(title, y=1.01)
        figure.tight_layout()
        figure.savefig(path_prefix.with_name(f"{path_prefix.name}_{metric}.png"), dpi=180)
        plt.close(figure)


def _plot_loss_components(
    history: pd.DataFrame, path: Path, title: str, loss_scale: str
) -> None:
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
    # The dotted baseline control comes from epoch 0 itself, rather than from
    # a separately aggregated table.  It is therefore exactly the state from
    # which this ascent trajectory started.
    reference = history.loc[history["epoch"] == 0].iloc[0]
    axes[0].axhline(
        reference["baseline_forget_task"], color="tab:blue", linestyle=":",
        label="baseline forget (epoch 0)",
    )
    axes[0].axhline(
        reference["baseline_retain_task"], color="tab:orange", linestyle=":",
        label="baseline retain (epoch 0)",
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
            label=f"baseline {component} (epoch 0)",
        )
        axes[1].axhline(
            reference[f"retrained_forget_{component}"], color=color, linestyle="--",
            label=f"retrained {component} (final)",
        )
    axes[0].set(title="Mean alignment loss (sample mean)", xlabel="ascent epoch", ylabel="loss")
    axes[1].set(title="Forget loss components (sample mean)", xlabel="ascent epoch", ylabel="loss")
    for axis in axes:
        if loss_scale == "symlog":
            # Preserve a linear neighbourhood around zero while compressing
            # the very large late-ascent losses.
            axis.set_yscale("symlog", linthresh=1e-2)
        else:
            axis.set_yscale(loss_scale)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title, y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _near_zero_limit(values: list[float]) -> float:
    """A zero-based view containing all baseline/retrained control values."""
    return max(1e-8, max(values) * 1.25)


def _plot_near_zero_losses(history: pd.DataFrame, path: Path, title: str) -> None:
    """Supplementary linear plots that resolve the initial/control-loss range."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = history.loc[history["epoch"] == 0].iloc[0]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    curve = history[history["run"] == history["run"].iloc[0]].sort_values("epoch")
    for assignment, color in (("forget", "tab:blue"), ("retain", "tab:orange")):
        axes[0].plot(curve["epoch"], curve[f"{assignment}_task"], color=color, label=assignment)
        axes[0].axhline(
            reference[f"baseline_{assignment}_task"], color=color, linestyle=":",
            label=f"baseline {assignment} (epoch 0)",
        )
        axes[0].axhline(
            reference[f"retrained_{assignment}_task"], color=color, linestyle="--",
            label=f"retrained {assignment} (final)",
        )
    for component, color in zip(COMPONENTS, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        axes[1].plot(curve["epoch"], curve[f"forget_{component}"], color=color, label=component)
        axes[1].axhline(
            reference[f"baseline_forget_{component}"], color=color, linestyle=":",
            label=f"baseline {component} (epoch 0)",
        )
        axes[1].axhline(
            reference[f"retrained_forget_{component}"], color=color, linestyle="--",
            label=f"retrained {component} (final)",
        )
    task_controls = [
        reference[f"{model}_{assignment}_task"]
        for model in ("baseline", "retrained") for assignment in ("forget", "retain")
    ]
    component_controls = [
        reference[f"{model}_forget_{component}"]
        for model in ("baseline", "retrained") for component in COMPONENTS
    ]
    axes[0].set(
        title="Mean alignment loss: near-zero view", xlabel="ascent epoch", ylabel="loss",
        ylim=(0, _near_zero_limit(task_controls)),
    )
    axes[1].set(
        title="Forget components: near-zero view", xlabel="ascent epoch", ylabel="loss",
        ylim=(0, _near_zero_limit(component_controls)),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"{title}: baseline/retrained neighbourhood", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    histories: list[pd.DataFrame] = []
    for experiment_label, experiment_root in args.experiment:
        if not experiment_root.is_dir():
            raise FileNotFoundError(f"experiment directory does not exist: {experiment_root}")
        rows, experiment_histories = _find_runs(experiment_label, experiment_root, args.unit)
        metric_rows.extend(rows)
        histories.extend(experiment_histories)

    metrics = pd.DataFrame(metric_rows).sort_values(["experiment", "run", "assignment"])
    metrics.to_csv(output_dir / "unlearned_vs_retrained_metrics.csv", index=False)
    for experiment_label, frame in metrics.groupby("experiment", sort=True):
        _plot_center_metrics(
            frame,
            output_dir / f"metrics_{_safe_name(experiment_label)}",
            experiment_label,
        )

    if histories:
        all_history = pd.concat(histories, ignore_index=True)
        all_history.to_csv(output_dir / "all_history.csv", index=False)
        for experiment_label, frame in all_history.groupby("experiment", sort=True):
            history_columns = ["step_mode", "learning_rate", "train_center_weight", "unlearn_seed"]
            for values, group in frame.groupby(history_columns, sort=True, dropna=False):
                if not isinstance(values, tuple):
                    values = (values,)
                tag = "_".join(f"{column}_{value:g}" if isinstance(value, float) else f"{column}_{value}" for column, value in zip(history_columns, values))
                _plot_loss_components(
                    group,
                    output_dir / f"loss_components_{_safe_name(experiment_label)}_{_safe_name(tag)}.png",
                    f"{experiment_label}: {tag}",
                    args.loss_scale,
                )
                _plot_near_zero_losses(
                    group,
                    output_dir / f"loss_near_zero_{_safe_name(experiment_label)}_{_safe_name(tag)}.png",
                    f"{experiment_label}: {tag}",
                )
    else:
        print("[warning] no history.csv files were found; no loss-component plots were written")
    print(f"[done] plots and merged CSVs -> {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=_parse_experiment, action="append", required=True,
        help="LABEL=PATH; repeat once per result directory",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unit", choices=("patient", "sample"), default="patient")
    parser.add_argument(
        "--loss-scale", choices=("linear", "log", "symlog"), default="log",
        help="y-axis scale for the original absolute-loss plot",
    )
    main(parser.parse_args())
