"""Multi-epoch forget-set gradient ascent for a trained THERAPI aligner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from center_loss import CenterLoss
from model import (
    AlignerDataset,
    Emb_Dis_classifier,
    Exp_Dis_classifier,
    SOURCE_AE,
    TARGET_weightencoder,
)
from unlearning.objective import alignment_losses, evaluate_loader, forward_aligner
from unlearning.split import build_sample_table, load_manifest_indices
from utils import set_seed


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _gradient_norm(parameters) -> float:
    return sum(
        parameter.grad.detach().pow(2).sum().item()
        for parameter in parameters
        if parameter.grad is not None
    ) ** 0.5


def _window_diagnostics(
    values: list[float], window: int, reference_loss: float
) -> tuple[float | None, float | None]:
    """Return baseline-normalized absolute slope and range for the latest window."""
    if len(values) < window:
        return None, None
    recent = values[-window:]
    x_mean = (window - 1) / 2
    denominator = sum((index - x_mean) ** 2 for index in range(window))
    mean_loss = sum(recent) / window
    slope = sum(
        (index - x_mean) * (value - mean_loss)
        for index, value in enumerate(recent)
    ) / denominator
    # A fixed baseline scale prevents a linearly diverging loss from looking
    # flat merely because its current magnitude has become large.
    scale = max(abs(reference_loss), 1e-12)
    return abs(slope) / scale, (max(recent) - min(recent)) / scale


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "forget_weight": args.forget_weight,
        "plateau_window": args.plateau_window,
        "min_forget_rise_rtol": args.min_forget_rise_rtol,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.plateau_window < 2:
        raise ValueError("plateau_window must be at least 2")
    if args.min_epochs < 1:
        raise ValueError("min_epochs must be at least 1")
    if args.patience < 0:
        raise ValueError("patience must be non-negative")
    if args.plateau_rtol < 0 or args.plateau_range_rtol < 0:
        raise ValueError("plateau tolerances must be non-negative")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    loss_weights = (args.recon_weight, args.class_weight, args.center_weight)
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError("loss weights must be non-negative and at least one must be positive")


def _plot_history(history, selected_epoch: int, path: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(epochs, [row["forget_task"] for row in history], label="forget")
    axes[0].plot(epochs, [row["retain_task"] for row in history], label="retain")
    axes[0].set(title="Mean alignment loss", xlabel="ascent epoch", ylabel="loss")
    axes[0].legend()
    for name in ("recon", "emb_class", "exp_class", "center"):
        axes[1].plot(epochs, [row[f"forget_{name}"] for row in history], label=name)
    axes[1].set(title="Forget loss components", xlabel="ascent epoch", ylabel="loss")
    axes[1].legend()
    axes[2].plot(
        epochs,
        [row["plateau_window_relative_slope"] for row in history],
        label="window |slope| / baseline",
    )
    axes[2].plot(
        epochs,
        [row["plateau_window_relative_range"] for row in history],
        label="window range / baseline",
    )
    axes[2].axhline(
        args.plateau_rtol, color="tab:blue", linestyle=":", label="slope tolerance"
    )
    axes[2].axhline(
        args.plateau_range_rtol,
        color="tab:orange",
        linestyle=":",
        label="range tolerance",
    )
    axes[2].set(title="Plateau diagnostics", xlabel="ascent epoch", ylabel="relative value")
    axes[2].legend()
    for axis in axes:
        axis.axvline(selected_epoch, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def unlearn(args: argparse.Namespace) -> None:
    _validate_args(args)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    requested_output = Path(args.output_dir)
    output_dir = requested_output if requested_output.name.lower() == "ckpts" else requested_output / "ckpts"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(data_dir / args.source / f"{args.source}_gex.csv", index_col=0)
    source_info = pd.read_csv(data_dir / args.source / f"{args.source}_info.csv")
    target_df = pd.read_csv(data_dir / args.target / f"{args.target}_unlabeled_gex.csv", index_col=0)
    target_info = pd.read_csv(data_dir / args.target / f"{args.target}_unlabeled_info.csv")
    samples = build_sample_table(
        target_df, target_info, tissue_column=args.tissue_column, info_id_column=args.info_id_column
    )
    forget_indices, retain_indices = load_manifest_indices(samples, args.split_dir)

    n_tissue = int(source_info["tissue_label"].nunique())
    source_dataset = AlignerDataset(source_df, args.source, source_info["tissue_label"])
    target_dataset = AlignerDataset(target_df, args.target, target_info[args.tissue_column])
    set_seed(args.original_train_seed)
    source_ae = SOURCE_AE(source_dataset.n_genes, n_tissue, args.latent_dim).to(device)
    target_encoder = TARGET_weightencoder(target_dataset.n_genes, args.latent_dim, len(source_dataset)).to(device)
    emb_classifier = Emb_Dis_classifier(args.latent_dim, n_tissue).to(device)
    exp_classifier = Exp_Dis_classifier(target_dataset.n_genes, args.latent_dim, n_tissue).to(device)
    center = CenterLoss(num_classes=n_tissue, feat_dim=args.latent_dim, device=device).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    source_ae.load_state_dict(checkpoint["source_AE"])
    target_encoder.load_state_dict(checkpoint["target_weightencoder"])
    emb_classifier.load_state_dict(checkpoint["emb_dis_classifier"])
    exp_classifier.load_state_dict(checkpoint["exp_dis_classifier"])
    if "center_criterion" in checkpoint:
        center.load_state_dict(checkpoint["center_criterion"])
        center_source = "checkpoint"
    else:
        center_source = f"reconstructed_from_original_train_seed_{args.original_train_seed}"

    # Same target-loss path as ordinary training. Source decoder has no target
    # gradient; center anchors remain fixed, matching the original optimizer.
    _freeze(source_ae.decoder)
    _freeze(center)
    groups = [
        ("source_encoder", source_ae.encoder.parameters()),
        ("target_Q", target_encoder.Q.parameters()),
        ("target_K", target_encoder.K.parameters()),
        ("target_decoder", target_encoder.decoder.parameters()),
        ("latent_classifier", emb_classifier.parameters()),
        ("expression_classifier", exp_classifier.parameters()),
    ]
    groups = [(name, list(parameters)) for name, parameters in groups]
    trainable = [parameter for _, parameters in groups for parameter in parameters]
    optimizer = torch.optim.Adam(
        [{"name": name, "params": parameters} for name, parameters in groups], lr=args.lr
    )
    models = (source_ae, target_encoder, emb_classifier, exp_classifier)

    set_seed(args.unlearn_seed)
    forget_loader = DataLoader(
        Subset(target_dataset, forget_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.unlearn_seed),
    )
    forget_eval_loader = DataLoader(
        Subset(target_dataset, forget_indices), batch_size=args.eval_batch_size, shuffle=False
    )
    retain_eval_loader = DataLoader(
        Subset(target_dataset, retain_indices), batch_size=args.eval_batch_size, shuffle=False
    )
    source_gex = source_dataset.data.to(device)

    def evaluate(loader):
        return evaluate_loader(
            loader,
            models,
            center,
            source_gex,
            args.recon_weight,
            args.class_weight,
            args.center_weight,
        )

    baseline_forget, baseline_retain = evaluate(forget_eval_loader), evaluate(retain_eval_loader)
    history = [{
        "epoch": 0,
        "optimizer_steps": 0,
        "cumulative_optimizer_steps": 0,
        "relative_forget_change": None,
        "relative_forget_rise_from_baseline": 0.0,
        "plateau_window_relative_slope": None,
        "plateau_window_relative_range": None,
        "plateau_eligible": False,
        "plateau_count": 0,
        "gradient_norm": None,
        **{f"forget_{key}": value for key, value in baseline_forget.items()},
        **{f"retain_{key}": value for key, value in baseline_retain.items()},
    }]
    print(f"[data] forget={len(forget_indices)} retain={len(retain_indices)} micro_batch={args.batch_size}")
    steps_per_epoch = len(forget_loader)
    effective_batch = min(args.batch_size, len(forget_indices))
    print(
        f"[batch] mode={args.step_mode} optimizer_steps_per_epoch={steps_per_epoch} "
        f"effective_batch<={effective_batch}"
    )
    print(f"[baseline] forget={baseline_forget['task']:.6f} retain={baseline_retain['task']:.6f}")

    previous_loss = baseline_forget["task"]
    baseline_loss = baseline_forget["task"]
    evaluated_forget_losses = [baseline_loss]
    plateau_count = 0
    cumulative_steps = 0
    stop_reason = "max_epochs"
    selected_epoch = args.epochs

    def optimizer_update():
        norms = {name: _gradient_norm(parameters) for name, parameters in groups}
        total_norm = _gradient_norm(trainable)
        if not math.isfinite(total_norm) or any(
            not math.isfinite(value) for value in norms.values()
        ):
            raise RuntimeError("non-finite gradient encountered before optimizer step")
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        return total_norm, norms

    for epoch in range(1, args.epochs + 1):
        for module in models:
            module.train()
        source_ae.decoder.eval()
        center.eval()
        step_norms = []
        for target_gex, _, labels in forget_loader:
            optimizer.zero_grad(set_to_none=True)
            target_gex, labels = target_gex.to(device), labels.to(device)
            output = forward_aligner(models, target_gex, source_gex)
            losses = alignment_losses(
                output,
                target_gex,
                labels,
                center,
                args.recon_weight,
                args.class_weight,
                args.center_weight,
            )
            objective = -args.forget_weight * losses["task"]
            if not torch.isfinite(objective):
                raise RuntimeError(f"non-finite ascent objective at epoch {epoch}")
            objective.backward()
            step_norms.append(optimizer_update())

        optimizer_steps = len(step_norms)
        cumulative_steps += optimizer_steps
        gradient_norm = sum(total for total, _ in step_norms) / optimizer_steps
        group_norms = {
            name: sum(norms[name] for _, norms in step_norms) / optimizer_steps
            for name, _ in groups
        }

        forget_metrics, retain_metrics = evaluate(forget_eval_loader), evaluate(retain_eval_loader)
        current_loss = forget_metrics["task"]
        if not math.isfinite(current_loss):
            raise RuntimeError(f"non-finite evaluated forget loss at epoch {epoch}")
        relative_change = abs(current_loss - previous_loss) / max(abs(previous_loss), 1e-12)
        relative_rise = (current_loss - baseline_loss) / max(abs(baseline_loss), 1e-12)
        evaluated_forget_losses.append(current_loss)
        window_slope, window_range = _window_diagnostics(
            evaluated_forget_losses, args.plateau_window, baseline_loss
        )
        plateau_eligible = (
            epoch >= args.min_epochs
            and relative_rise >= args.min_forget_rise_rtol
            and window_slope is not None
            and window_slope <= args.plateau_rtol
            and window_range <= args.plateau_range_rtol
        )
        plateau_count = plateau_count + 1 if plateau_eligible else 0
        history.append({
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "cumulative_optimizer_steps": cumulative_steps,
            "relative_forget_change": relative_change,
            "relative_forget_rise_from_baseline": relative_rise,
            "plateau_window_relative_slope": window_slope,
            "plateau_window_relative_range": window_range,
            "plateau_eligible": plateau_eligible,
            "plateau_count": plateau_count,
            "gradient_norm": gradient_norm,
            **{f"grad_{name}": value for name, value in group_norms.items()},
            **{f"forget_{key}": value for key, value in forget_metrics.items()},
            **{f"retain_{key}": value for key, value in retain_metrics.items()},
        })
        print(
            f"[epoch {epoch}/{args.epochs}] forget={current_loss:.6f} "
            f"retain={retain_metrics['task']:.6f} rel_change={relative_change:.3g} "
            f"rise={relative_rise:.3g} window_slope={window_slope} "
            f"window_range={window_range} steps={optimizer_steps} "
            f"plateau={plateau_count}/{args.patience}"
        )
        previous_loss = current_loss
        if args.patience > 0 and plateau_count >= args.patience:
            selected_epoch, stop_reason = epoch, "forget_loss_plateau"
            break
    else:
        selected_epoch = args.epochs

    final_forget, final_retain = evaluate(forget_eval_loader), evaluate(retain_eval_loader)
    checkpoint_path = output_dir / f"THERAPI_aligner_{args.source}_{args.target}.pt"
    torch.save(
        {
            "epoch": checkpoint.get("epoch"),
            "unlearning_epochs": selected_epoch,
            "unlearning_optimizer_steps": cumulative_steps,
            "source_AE": source_ae.state_dict(),
            "target_weightencoder": target_encoder.state_dict(),
            "emb_dis_classifier": emb_classifier.state_dict(),
            "exp_dis_classifier": exp_classifier.state_dict(),
            "center_criterion": center.state_dict(),
            "optimizer": optimizer.state_dict(),
            "original_checkpoint": str(Path(args.checkpoint).resolve()),
            "split_dir": str(Path(args.split_dir).resolve()),
            "unlearning_config": vars(args),
        },
        checkpoint_path,
    )
    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    _plot_history(history, selected_epoch, output_dir / "loss_curve.png", args)
    plateau_detected = stop_reason == "forget_loss_plateau"
    summary = {
        "objective": "maximize_original_target_alignment_loss_on_forget_set",
        "step_mode": args.step_mode,
        "micro_batch_size": args.batch_size,
        "effective_batch_size": effective_batch,
        "optimizer_steps": cumulative_steps,
        "selected_epoch": selected_epoch,
        "selection_valid": plateau_detected,
        "stop_reason": stop_reason,
        "plateau_rule": {
            "minimum_epochs": args.min_epochs,
            "window": args.plateau_window,
            "minimum_relative_rise_from_baseline": args.min_forget_rise_rtol,
            "maximum_baseline_normalized_absolute_window_slope": args.plateau_rtol,
            "maximum_baseline_normalized_window_range": args.plateau_range_rtol,
            "consecutive_epochs": args.patience,
        },
        "center_source": center_source,
        "baseline_forget": baseline_forget,
        "baseline_retain_evaluation_only": baseline_retain,
        "final_forget": final_forget,
        "final_retain_evaluation_only": final_retain,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[stop] epoch={selected_epoch} reason={stop_reason}")
    if not plateau_detected:
        print(
            "[warning] no valid post-rise plateau was detected; inspect loss_curve.png "
            "before treating the final checkpoint as a selected unlearning result"
        )
    print(f"[done] checkpoint={checkpoint_path.resolve()} curve={(output_dir / 'loss_curve.png').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument("--source", default="GDSC")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-train-seed", type=int, default=0)
    parser.add_argument("--unlearn-seed", type=int, default=0)
    parser.add_argument("--tissue-column", default="tissue_label")
    parser.add_argument("--info-id-column", default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--plateau-rtol", type=float, default=1e-3)
    parser.add_argument("--plateau-range-rtol", type=float, default=5e-3)
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--min-forget-rise-rtol", type=float, default=1e-2)
    parser.add_argument("--step-mode", choices=("mini",), default="mini")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    unlearn(parser.parse_args())
