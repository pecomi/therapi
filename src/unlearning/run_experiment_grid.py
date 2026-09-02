"""Run and aggregate gradient-ascent epoch/center-weight experiments."""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_STATE_KEYS = (
    "source_AE",
    "target_weightencoder",
    "emb_dis_classifier",
    "exp_dis_classifier",
)
PARAMETER_GROUPS = {
    "source_encoder": ("source_AE", "encoder."),
    "source_decoder": ("source_AE", "decoder."),
    "target_Q": ("target_weightencoder", "Q."),
    "target_K": ("target_weightencoder", "K."),
    "target_decoder": ("target_weightencoder", "decoder."),
    "latent_classifier": ("emb_dis_classifier", ""),
    "expression_classifier": ("exp_dis_classifier", ""),
}


def _value_tag(value: float) -> str:
    return format(value, "g").replace("-", "m").replace(".", "p")


def _forget_sample_count(split_dir: Path) -> int:
    samples_path = split_dir / "samples.csv"
    samples = pd.read_csv(samples_path)
    if "assignment" not in samples.columns:
        raise ValueError(f"missing assignment column: {samples_path}")
    count = int((samples["assignment"] == "forget").sum())
    if count < 1:
        raise ValueError(f"split contains no forget samples: {samples_path}")
    return count


def _run(command: list[str], dry_run: bool) -> None:
    print(f"[command] {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _core_model_stats(checkpoint: dict) -> tuple[int, int]:
    tensors = [
        tensor
        for state_key in CORE_STATE_KEYS
        for tensor in checkpoint[state_key].values()
        if torch.is_tensor(tensor)
    ]
    return (
        sum(tensor.numel() for tensor in tensors),
        sum(tensor.numel() * tensor.element_size() for tensor in tensors),
    )


def _group_tensors(checkpoint: dict, state_key: str, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in checkpoint[state_key].items()
        if name.startswith(prefix) and torch.is_tensor(tensor)
    }


def _updated_parameter_count(checkpoint: dict) -> int:
    """Count parameters in the groups optimized by gradient_ascent.py."""
    return sum(
        tensor.numel()
        for group_name, (state_key, prefix) in PARAMETER_GROUPS.items()
        if group_name != "source_decoder"
        for tensor in _group_tensors(checkpoint, state_key, prefix).values()
    )


def _relative_parameter_drift(
    baseline: dict, candidate: dict, state_key: str, prefix: str
) -> tuple[int, float]:
    baseline_tensors = _group_tensors(baseline, state_key, prefix)
    candidate_tensors = _group_tensors(candidate, state_key, prefix)
    if baseline_tensors.keys() != candidate_tensors.keys():
        raise ValueError(
            f"state-dict mismatch for {state_key}:{prefix!r}: "
            f"baseline={sorted(baseline_tensors)} candidate={sorted(candidate_tensors)}"
        )
    count = 0
    difference_squared = 0.0
    baseline_squared = 0.0
    for name, baseline_tensor in baseline_tensors.items():
        candidate_tensor = candidate_tensors[name]
        if baseline_tensor.shape != candidate_tensor.shape:
            raise ValueError(
                f"shape mismatch for {state_key}.{name}: "
                f"{tuple(baseline_tensor.shape)} != {tuple(candidate_tensor.shape)}"
            )
        baseline_float = baseline_tensor.to(torch.float64)
        candidate_float = candidate_tensor.to(torch.float64)
        count += baseline_tensor.numel()
        difference_squared += (candidate_float - baseline_float).pow(2).sum().item()
        baseline_squared += baseline_float.pow(2).sum().item()
    denominator = math.sqrt(baseline_squared)
    relative_l2 = math.sqrt(difference_squared) / denominator if denominator > 0 else math.nan
    return count, relative_l2


def _one_value(
    frame: pd.DataFrame,
    *,
    unit: str,
    assignment: str,
    model: str | None = None,
    comparison: str | None = None,
    column: str,
) -> float:
    mask = (frame["unit"] == unit) & (frame["assignment"] == assignment)
    if model is not None:
        mask &= frame["model"] == model
    if comparison is not None:
        mask &= frame["comparison"] == comparison
    selected = frame.loc[mask, column]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {column} row for unit={unit}, assignment={assignment}, "
            f"model={model}, comparison={comparison}; found {len(selected)}"
        )
    return float(selected.iloc[0])


def _plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("forget_task_gap_to_retrained", "Forget task gap to retrained", "lower"),
        ("forget_unlearned_vs_retrained_cka", "Forget CKA: U vs R", "higher"),
        (
            "forget_unlearned_vs_retrained_frechet",
            "Forget Frechet: U vs R",
            "lower",
        ),
        ("retain_task_relative_change", "Retain task relative change", "closer to 0"),
        ("retain_baseline_vs_unlearned_cka", "Retain CKA: B vs U", "higher"),
        (
            "retain_baseline_vs_unlearned_frechet",
            "Retain Frechet: B vs U",
            "lower",
        ),
    )
    labels = summary["run"].tolist()
    x = list(range(len(labels)))
    figure, axes = plt.subplots(2, 3, figsize=(17, 8))
    for axis, (column, title, direction) in zip(axes.ravel(), panels):
        axis.bar(x, summary[column].to_numpy())
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{title} ({direction})")
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _aggregate(
    experiments: list[dict],
    baseline_checkpoint: Path,
    retrained_checkpoint: Path,
    output_root: Path,
) -> None:
    all_losses = []
    all_similarities = []
    summary_rows = []
    drift_rows = []
    model_size_rows = []

    baseline_state = _load_checkpoint(baseline_checkpoint)
    for model_name, checkpoint_path in (
        ("baseline", baseline_checkpoint),
        ("retrained", retrained_checkpoint),
    ):
        checkpoint = _load_checkpoint(checkpoint_path)
        numel, model_bytes = _core_model_stats(checkpoint)
        updated_numel = _updated_parameter_count(checkpoint)
        model_size_rows.append(
            {
                "run": model_name,
                "step_mode": None,
                "epochs": None,
                "learning_rate": None,
                "optimizer_steps_per_epoch": None,
                "expected_optimizer_steps": None,
                "train_recon_weight": None,
                "train_class_weight": None,
                "train_center_weight": None,
                "core_parameter_count": numel,
                "ga_updated_parameter_count": updated_numel,
                "ga_updated_parameter_fraction": updated_numel / numel,
                "core_weight_mib": model_bytes / 1024**2,
                "checkpoint_mib": checkpoint_path.stat().st_size / 1024**2,
            }
        )

    for experiment in experiments:
        run_name = experiment["run"]
        step_mode = experiment["step_mode"]
        epochs = experiment["epochs"]
        learning_rate = experiment["learning_rate"]
        optimizer_steps_per_epoch = experiment["optimizer_steps_per_epoch"]
        expected_optimizer_steps = experiment["expected_optimizer_steps"]
        recon_weight = experiment["recon_weight"]
        class_weight = experiment["class_weight"]
        center_weight = experiment["center_weight"]
        checkpoint_path = experiment["checkpoint"]
        evaluation_dir = experiment["evaluation_dir"]
        loss_frame = pd.read_csv(evaluation_dir / "loss_metrics.csv")
        similarity_frame = pd.read_csv(
            evaluation_dir / "representation_similarity.csv"
        )
        for frame in (loss_frame, similarity_frame):
            frame.insert(0, "train_class_weight", class_weight)
            frame.insert(0, "train_recon_weight", recon_weight)
            frame.insert(0, "train_center_weight", center_weight)
            frame.insert(0, "expected_optimizer_steps", expected_optimizer_steps)
            frame.insert(0, "optimizer_steps_per_epoch", optimizer_steps_per_epoch)
            frame.insert(0, "learning_rate", learning_rate)
            frame.insert(0, "epochs", epochs)
            frame.insert(0, "step_mode", step_mode)
            frame.insert(0, "run", run_name)
        all_losses.append(loss_frame)
        all_similarities.append(similarity_frame)

        def loss(assignment: str, model: str) -> float:
            return _one_value(
                loss_frame,
                unit="patient",
                assignment=assignment,
                model=model,
                column="task",
            )

        def similarity(assignment: str, comparison: str, column: str) -> float:
            return _one_value(
                similarity_frame,
                unit="patient",
                assignment=assignment,
                comparison=comparison,
                column=column,
            )

        forget_baseline = loss("forget", "baseline")
        forget_unlearned = loss("forget", "unlearned")
        forget_retrained = loss("forget", "retrained")
        retain_baseline = loss("retain", "baseline")
        retain_unlearned = loss("retain", "unlearned")
        summary_rows.append(
            {
                "run": run_name,
                "step_mode": step_mode,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "expected_optimizer_steps": expected_optimizer_steps,
                "train_recon_weight": recon_weight,
                "train_class_weight": class_weight,
                "train_center_weight": center_weight,
                "forget_task_baseline": forget_baseline,
                "forget_task_unlearned": forget_unlearned,
                "forget_task_retrained": forget_retrained,
                "forget_task_rise_vs_baseline": forget_unlearned - forget_baseline,
                "forget_task_gap_to_retrained": abs(forget_unlearned - forget_retrained),
                "retain_task_baseline": retain_baseline,
                "retain_task_unlearned": retain_unlearned,
                "retain_task_change_vs_baseline": retain_unlearned - retain_baseline,
                "retain_task_relative_change": (
                    (retain_unlearned - retain_baseline) / abs(retain_baseline)
                    if retain_baseline != 0
                    else math.nan
                ),
                "forget_unlearned_vs_retrained_cka": similarity(
                    "forget", "unlearned_vs_retrained", "linear_cka"
                ),
                "forget_unlearned_vs_retrained_frechet": similarity(
                    "forget", "unlearned_vs_retrained", "frechet_latent_distance"
                ),
                "retain_baseline_vs_unlearned_cka": similarity(
                    "retain", "baseline_vs_unlearned", "linear_cka"
                ),
                "retain_baseline_vs_unlearned_frechet": similarity(
                    "retain", "baseline_vs_unlearned", "frechet_latent_distance"
                ),
            }
        )

        candidate_state = _load_checkpoint(checkpoint_path)
        numel, model_bytes = _core_model_stats(candidate_state)
        updated_numel = _updated_parameter_count(candidate_state)
        model_size_rows.append(
            {
                "run": run_name,
                "step_mode": step_mode,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "expected_optimizer_steps": expected_optimizer_steps,
                "train_recon_weight": recon_weight,
                "train_class_weight": class_weight,
                "train_center_weight": center_weight,
                "core_parameter_count": numel,
                "ga_updated_parameter_count": updated_numel,
                "ga_updated_parameter_fraction": updated_numel / numel,
                "core_weight_mib": model_bytes / 1024**2,
                "checkpoint_mib": checkpoint_path.stat().st_size / 1024**2,
            }
        )
        for group_name, (state_key, prefix) in PARAMETER_GROUPS.items():
            count, relative_l2 = _relative_parameter_drift(
                baseline_state, candidate_state, state_key, prefix
            )
            drift_rows.append(
                {
                    "run": run_name,
                    "step_mode": step_mode,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "expected_optimizer_steps": expected_optimizer_steps,
                    "train_recon_weight": recon_weight,
                    "train_class_weight": class_weight,
                    "train_center_weight": center_weight,
                    "parameter_group": group_name,
                    "parameter_count": count,
                    "relative_l2_from_baseline": relative_l2,
                }
            )

    pd.concat(all_losses, ignore_index=True).to_csv(
        output_root / "all_loss_metrics.csv", index=False
    )
    pd.concat(all_similarities, ignore_index=True).to_csv(
        output_root / "all_representation_similarity.csv", index=False
    )
    comparison_summary = pd.DataFrame(summary_rows).sort_values(
        [
            "step_mode",
            "train_recon_weight",
            "train_class_weight",
            "train_center_weight",
            "epochs",
        ]
    )
    comparison_summary.to_csv(output_root / "comparison_summary.csv", index=False)
    _plot_comparison(comparison_summary, output_root / "comparison_summary.png")
    pd.DataFrame(model_size_rows).to_csv(
        output_root / "model_size.csv", index=False
    )
    pd.DataFrame(drift_rows).to_csv(
        output_root / "parameter_drift.csv", index=False
    )
    print("\n[patient-level comparison]")
    print(comparison_summary.to_string(index=False))
    print(f"[done] aggregate results -> {output_root.resolve()}")


def main(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if any(epoch < 1 for epoch in args.epochs):
        raise ValueError("epochs must be positive")
    if args.full_lr <= 0 or (args.mini_lr is not None and args.mini_lr <= 0):
        raise ValueError("learning rates must be positive")
    loss_weights = (*args.recon_weights, *args.class_weights, *args.center_weights)
    if any(weight < 0 for weight in loss_weights):
        raise ValueError("loss weights must be non-negative")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = Path(args.baseline_checkpoint)
    retrained_checkpoint = Path(args.retrained_checkpoint)
    for label, path in (
        ("baseline checkpoint", baseline_checkpoint),
        ("retrained checkpoint", retrained_checkpoint),
        ("split manifest", Path(args.split_dir) / "samples.csv"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    canonical_name = f"THERAPI_aligner_{args.source}_{args.target}.pt"
    experiments = []
    n_forget = _forget_sample_count(Path(args.split_dir))
    mini_steps_per_epoch = math.ceil(n_forget / args.batch_size)
    mini_lr = (
        args.mini_lr
        if args.mini_lr is not None
        else args.full_lr / mini_steps_per_epoch
    )
    learning_rates = {"full": args.full_lr, "mini": mini_lr}
    steps_per_epoch = {"full": 1, "mini": mini_steps_per_epoch}
    print(
        f"[design] forget_samples={n_forget} batch_size={args.batch_size} "
        f"mini_steps_per_epoch={mini_steps_per_epoch} full_lr={args.full_lr:g} "
        f"mini_lr={mini_lr:g}"
    )

    for step_mode in args.step_modes:
        learning_rate = learning_rates[step_mode]
        for recon_weight in args.recon_weights:
            for class_weight in args.class_weights:
                for center_weight in args.center_weights:
                    for epochs in args.epochs:
                        run_name = (
                            f"{step_mode}_epoch_{epochs:03d}"
                            f"_recon_{_value_tag(recon_weight)}"
                            f"_class_{_value_tag(class_weight)}"
                            f"_center_{_value_tag(center_weight)}"
                        )
                        run_dir = output_root / run_name
                        checkpoint_path = run_dir / "ckpts" / canonical_name
                        evaluation_dir = run_dir / "evaluation" / "representations"
                        experiments.append(
                            {
                                "run": run_name,
                                "step_mode": step_mode,
                                "epochs": epochs,
                                "learning_rate": learning_rate,
                                "optimizer_steps_per_epoch": steps_per_epoch[step_mode],
                                "expected_optimizer_steps": (
                                    steps_per_epoch[step_mode] * epochs
                                ),
                                "recon_weight": recon_weight,
                                "class_weight": class_weight,
                                "center_weight": center_weight,
                                "checkpoint": checkpoint_path,
                                "evaluation_dir": evaluation_dir,
                            }
                        )

                        if not args.skip_training:
                            training_command = [
                                sys.executable,
                                str(SCRIPT_DIR / "gradient_ascent.py"),
                                "--data_dir",
                                args.data_dir,
                                "--source",
                                args.source,
                                "--target",
                                args.target,
                                "--checkpoint",
                                str(baseline_checkpoint),
                                "--split-dir",
                                args.split_dir,
                                "--output-dir",
                                str(run_dir),
                                "--device",
                                args.device,
                                "--original-train-seed",
                                str(args.original_train_seed),
                                "--unlearn-seed",
                                str(args.unlearn_seed),
                                "--tissue-column",
                                args.tissue_column,
                                "--latent-dim",
                                str(args.latent_dim),
                                "--step-mode",
                                step_mode,
                                "--batch-size",
                                str(args.batch_size),
                                "--eval-batch-size",
                                str(args.eval_batch_size),
                                "--lr",
                                str(learning_rate),
                                "--epochs",
                                str(epochs),
                                "--patience",
                                "0",
                                "--recon-weight",
                                str(recon_weight),
                                "--class-weight",
                                str(class_weight),
                                "--center-weight",
                                str(center_weight),
                                "--forget-weight",
                                str(args.forget_weight),
                                "--max-grad-norm",
                                str(args.max_grad_norm),
                            ]
                            if args.info_id_column is not None:
                                training_command.extend(
                                    ["--info-id-column", args.info_id_column]
                                )
                            _run(training_command, args.dry_run)

                        if not args.skip_evaluation:
                            evaluation_command = [
                                sys.executable,
                                str(SCRIPT_DIR / "evaluate_representations.py"),
                                "--data_dir",
                                args.data_dir,
                                "--source",
                                args.source,
                                "--target",
                                args.target,
                                "--baseline-checkpoint",
                                str(baseline_checkpoint),
                                "--unlearned-checkpoint",
                                str(checkpoint_path),
                                "--retrained-checkpoint",
                                str(retrained_checkpoint),
                                "--split-dir",
                                args.split_dir,
                                "--output-dir",
                                str(evaluation_dir),
                                "--device",
                                args.device,
                                "--original-train-seed",
                                str(args.original_train_seed),
                                "--tissue-column",
                                args.tissue_column,
                                "--latent-dim",
                                str(args.latent_dim),
                                "--batch-size",
                                str(args.eval_batch_size),
                                "--num-workers",
                                str(args.num_workers),
                                "--recon-weight",
                                str(args.eval_recon_weight),
                                "--class-weight",
                                str(args.eval_class_weight),
                                "--center-weight",
                                str(args.eval_center_weight),
                                "--frechet-eps",
                                str(args.frechet_eps),
                                "--center-tolerance",
                                str(args.center_tolerance),
                            ]
                            if args.info_id_column is not None:
                                evaluation_command.extend(
                                    ["--info-id-column", args.info_id_column]
                                )
                            _run(evaluation_command, args.dry_run)

    if args.dry_run:
        print("[dry-run] commands only; no aggregate files were written")
        return
    _aggregate(experiments, baseline_checkpoint, retrained_checkpoint, output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--source", default="GDSC")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--retrained-checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-train-seed", type=int, default=0)
    parser.add_argument("--unlearn-seed", type=int, default=0)
    parser.add_argument("--tissue-column", default="tissue_label")
    parser.add_argument("--info-id-column", default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--recon-weights", type=float, nargs="+", default=[0.2])
    parser.add_argument("--class-weights", type=float, nargs="+", default=[0.4])
    parser.add_argument("--center-weights", type=float, nargs="+", default=[0.8])
    parser.add_argument(
        "--step-modes", choices=("full", "mini"), nargs="+", default=["full", "mini"]
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-lr", type=float, default=1e-5)
    parser.add_argument("--mini-lr", type=float, default=None)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--eval-recon-weight", type=float, default=0.2)
    parser.add_argument("--eval-class-weight", type=float, default=0.4)
    parser.add_argument("--eval-center-weight", type=float, default=0.8)
    parser.add_argument("--frechet-eps", type=float, default=1e-6)
    parser.add_argument("--center-tolerance", type=float, default=1e-7)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
