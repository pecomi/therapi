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


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "forget_weight": args.forget_weight,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    loss_weights = (args.recon_weight, args.class_weight, args.center_weight)
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError("loss weights must be non-negative and at least one must be positive")


def _plot_history(history, path: Path, loss_scale: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["forget_task"] for row in history], label="forget")
    axes[0].plot(epochs, [row["retain_task"] for row in history], label="retain")
    axes[0].set(title="Mean alignment loss", xlabel="ascent epoch", ylabel="loss")
    axes[0].legend()
    for name in ("recon", "emb_class", "exp_class", "center"):
        axes[1].plot(epochs, [row[f"forget_{name}"] for row in history], label=name)
    axes[1].set(title="Forget loss components", xlabel="ascent epoch", ylabel="loss")
    axes[1].legend()
    for axis in axes:
        if loss_scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-2)
        else:
            axis.set_yscale(loss_scale)
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
    _freeze(target_encoder.decoder)
    groups = [
        ("source_encoder", source_ae.encoder.parameters()),
        ("target_Q", target_encoder.Q.parameters()),
        ("target_K", target_encoder.K.parameters()),
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

    cumulative_steps = 0

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
        history.append({
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "cumulative_optimizer_steps": cumulative_steps,
            "gradient_norm": gradient_norm,
            **{f"grad_{name}": value for name, value in group_norms.items()},
            **{f"forget_{key}": value for key, value in forget_metrics.items()},
            **{f"retain_{key}": value for key, value in retain_metrics.items()},
        })
        print(
            f"[epoch {epoch}/{args.epochs}] forget={current_loss:.6f} "
            f"retain={retain_metrics['task']:.6f} steps={optimizer_steps}"
        )

    final_forget, final_retain = evaluate(forget_eval_loader), evaluate(retain_eval_loader)
    checkpoint_path = output_dir / f"THERAPI_aligner_{args.source}_{args.target}.pt"
    torch.save(
        {
            "epoch": checkpoint.get("epoch"),
            "unlearning_epochs": args.epochs,
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
    _plot_history(history, output_dir / "loss_curve.png", args.loss_scale)
    summary = {
        "objective": "maximize_original_target_alignment_loss_on_forget_set",
        "step_mode": args.step_mode,
        "micro_batch_size": args.batch_size,
        "effective_batch_size": effective_batch,
        "optimizer_steps": cumulative_steps,
        "completed_epochs": args.epochs,
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
    print(f"[done] completed_epochs={args.epochs}")
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--step-mode", choices=("mini",), default="mini")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument(
        "--loss-scale", choices=("linear", "log", "symlog"), default="log",
        help="y-axis scale for the saved loss curve",
    )
    unlearn(parser.parse_args())
