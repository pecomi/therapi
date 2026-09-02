"""Run one fixed mini-batch gradient-ascent unlearning experiment."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FIXED_BATCH_SIZE, FIXED_LR, FIXED_EPOCHS = 128, 1e-4, 30


def _tag(value: float) -> str:
    return format(value, "g").replace(".", "p").replace("-", "m")


def _run(command: list[str], dry_run: bool) -> None:
    print(f"[command] {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _one(frame: pd.DataFrame, assignment: str, model: str | None, comparison: str | None, column: str) -> float:
    mask = (frame["unit"] == "patient") & (frame["assignment"] == assignment)
    if model is not None:
        mask &= frame["model"] == model
    if comparison is not None:
        mask &= frame["comparison"] == comparison
    values = frame.loc[mask, column]
    if len(values) != 1:
        raise ValueError(f"missing unique metric: {assignment=}, {model=}, {comparison=}, {column=}")
    return float(values.iloc[0])


def _aggregate(experiments: list[dict], output_root: Path) -> None:
    rows, histories = [], []
    for experiment in experiments:
        losses = pd.read_csv(experiment["evaluation_dir"] / "loss_metrics.csv")
        similarities = pd.read_csv(experiment["evaluation_dir"] / "representation_similarity.csv")
        history = pd.read_csv(experiment["history_path"])
        history.insert(0, "run", experiment["run"])
        history.insert(0, "unlearn_seed", experiment["unlearn_seed"])
        history.insert(0, "train_center_weight", experiment["center_weight"])
        histories.append(history)
        for assignment in ("forget", "retain"):
            unlearned = _one(losses, assignment, "unlearned", None, "task")
            retrained = _one(losses, assignment, "retrained", None, "task")
            rows.append({
                "run": experiment["run"],
                "train_center_weight": experiment["center_weight"],
                "unlearn_seed": experiment["unlearn_seed"],
                "assignment": assignment,
                "task_loss_difference_unlearned_minus_retrained": unlearned - retrained,
                "linear_cka_unlearned_vs_retrained": _one(similarities, assignment, None, "unlearned_vs_retrained", "linear_cka"),
                "frechet_unlearned_vs_retrained": _one(similarities, assignment, None, "unlearned_vs_retrained", "frechet_latent_distance"),
            })
    pd.DataFrame(rows).to_csv(output_root / "comparison_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output_root / "all_history.csv", index=False)


def main(args: argparse.Namespace) -> None:
    baseline, retrained, split_dir = Path(args.baseline_checkpoint), Path(args.retrained_checkpoint), Path(args.split_dir)
    for label, path in (("baseline checkpoint", baseline), ("retrained checkpoint", retrained), ("split manifest", split_dir / "samples.csv")):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"THERAPI_aligner_{args.source}_{args.target}.pt"
    experiments = []
    center_weight = args.center_weight
    if center_weight < 0:
        raise ValueError("center weight must be non-negative")
    for unlearn_seed in args.unlearn_seeds:
        run_name = f"mini_epoch_{FIXED_EPOCHS:03d}_lr_{_tag(FIXED_LR)}_center_{_tag(center_weight)}_unlearn_seed_{unlearn_seed}"
        run_dir = output_root / run_name
        checkpoint = run_dir / "ckpts" / checkpoint_name
        evaluation_dir = run_dir / "evaluation" / "representations"
        experiments.append({"run": run_name, "center_weight": center_weight, "unlearn_seed": unlearn_seed, "history_path": run_dir / "ckpts" / "history.csv", "evaluation_dir": evaluation_dir})
        if not args.skip_training:
            _run([
                sys.executable, str(SCRIPT_DIR / "gradient_ascent.py"), "--data_dir", args.data_dir,
                "--source", args.source, "--target", args.target, "--checkpoint", str(baseline),
                "--split-dir", str(split_dir), "--output-dir", str(run_dir), "--device", args.device,
                "--original-train-seed", str(args.original_train_seed), "--unlearn-seed", str(unlearn_seed),
                "--step-mode", "mini", "--batch-size", str(FIXED_BATCH_SIZE), "--lr", str(FIXED_LR),
                "--epochs", str(FIXED_EPOCHS), "--patience", "0", "--recon-weight", "0.2",
                "--class-weight", "0.4", "--center-weight", str(center_weight),
            ], args.dry_run)
        if not args.skip_evaluation:
            _run([
                sys.executable, str(SCRIPT_DIR / "evaluate_representations.py"), "--data_dir", args.data_dir,
                "--source", args.source, "--target", args.target, "--baseline-checkpoint", str(baseline),
                "--unlearned-checkpoint", str(checkpoint), "--retrained-checkpoint", str(retrained),
                "--split-dir", str(split_dir), "--output-dir", str(evaluation_dir), "--device", args.device,
                "--original-train-seed", str(args.original_train_seed), "--recon-weight", "0.2",
                "--class-weight", "0.4", "--center-weight", "0.8",
            ], args.dry_run)
    if not args.dry_run:
        _aggregate(experiments, output_root)
        _run([sys.executable, str(SCRIPT_DIR / "plot_unlearning_results.py"), "--experiment", f"mini_center={output_root}", "--output-dir", str(output_root / "plots")], False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="data")
    parser.add_argument("--source", default="GDSC")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--retrained-checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-train-seed", type=int, default=0)
    parser.add_argument("--unlearn-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
