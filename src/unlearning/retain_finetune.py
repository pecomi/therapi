"""Joint forget-ascent and retain-descent updates from a baseline aligner."""

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
from torch.utils.data import DataLoader, RandomSampler, Subset

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

LOSS_NAMES = ("task", "recon", "emb_class", "exp_class", "center")


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _gradient_norm(parameters) -> float:
    return sum(
        parameter.grad.detach().pow(2).sum().item()
        for parameter in parameters
        if parameter.grad is not None
    ) ** 0.5


def _plot_history(history, path: Path, loss_scale: str) -> None:
    """Plot actual update losses separately from full-set evaluation losses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_rows = [row for row in history if row["epoch"] > 0]
    train_epochs = [row["epoch"] for row in train_rows]
    eval_epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    axes[0].plot(
        train_epochs,
        [row["train_forget_task"] for row in train_rows],
        label="forget batch loss",
    )
    axes[0].plot(
        train_epochs,
        [row["train_retain_task"] for row in train_rows],
        label="retain batch loss",
    )
    axes[0].set(
        title="Losses used during updates",
        xlabel="epoch",
        ylabel="mean pre-update batch loss",
    )

    axes[1].plot(
        train_epochs,
        [row["train_retain_task"] for row in train_rows],
        label="+ retain",
    )
    axes[1].plot(
        train_epochs,
        [-row["train_forget_task"] for row in train_rows],
        label="- forget",
    )
    axes[1].plot(
        train_epochs,
        [row["train_joint_objective"] for row in train_rows],
        label="joint",
        linewidth=2,
    )
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set(
        title="Signed optimizer objective",
        xlabel="epoch",
        ylabel="retain - forget",
    )

    axes[2].plot(
        eval_epochs,
        [row["forget_task"] for row in history],
        label="forget full set",
    )
    axes[2].plot(
        eval_epochs,
        [row["retain_task"] for row in history],
        label="retain full set",
    )
    axes[2].set(
        title="Post-epoch full-set evaluation",
        xlabel="epoch",
        ylabel="mean loss",
    )

    for index, axis in enumerate(axes):
        scale = "symlog" if index == 1 and loss_scale == "log" else loss_scale
        if scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-2)
        else:
            axis.set_yscale(scale)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    loss_weights = (args.recon_weight, args.class_weight, args.center_weight)
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError(
            "loss weights must be non-negative and at least one must be positive"
        )


def joint_unlearn(args: argparse.Namespace) -> None:
    """Minimize retain loss minus forget loss with equal group weights."""
    _validate_args(args)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    requested_output = Path(args.output_dir)
    output_dir = (
        requested_output
        if requested_output.name.lower() == "ckpts"
        else requested_output / "ckpts"
    )
    checkpoint_path = output_dir / f"THERAPI_aligner_{args.source}_{args.target}.pt"
    input_checkpoint = Path(args.checkpoint)
    if checkpoint_path.resolve() == input_checkpoint.resolve():
        raise ValueError(
            "--output-dir would overwrite the input baseline checkpoint; "
            "use a separate run directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(
        data_dir / args.source / f"{args.source}_gex.csv", index_col=0
    )
    source_info = pd.read_csv(data_dir / args.source / f"{args.source}_info.csv")
    target_df = pd.read_csv(
        data_dir / args.target / f"{args.target}_unlabeled_gex.csv", index_col=0
    )
    target_info = pd.read_csv(
        data_dir / args.target / f"{args.target}_unlabeled_info.csv"
    )
    samples = build_sample_table(
        target_df,
        target_info,
        tissue_column=args.tissue_column,
        info_id_column=args.info_id_column,
    )
    forget_indices, retain_indices = load_manifest_indices(samples, args.split_dir)

    n_tissue = int(source_info["tissue_label"].nunique())
    source_dataset = AlignerDataset(
        source_df, args.source, source_info["tissue_label"]
    )
    target_dataset = AlignerDataset(
        target_df, args.target, target_info[args.tissue_column]
    )

    # Reproduce the legacy center initialization if the checkpoint predates
    # center serialization. All learned modules are then restored exactly.
    set_seed(args.original_train_seed)
    source_ae = SOURCE_AE(
        source_dataset.n_genes, n_tissue, args.latent_dim
    ).to(device)
    target_encoder = TARGET_weightencoder(
        target_dataset.n_genes, args.latent_dim, len(source_dataset)
    ).to(device)
    emb_classifier = Emb_Dis_classifier(args.latent_dim, n_tissue).to(device)
    exp_classifier = Exp_Dis_classifier(
        target_dataset.n_genes, args.latent_dim, n_tissue
    ).to(device)
    center = CenterLoss(
        num_classes=n_tissue, feat_dim=args.latent_dim, device=device
    ).to(device)

    checkpoint = torch.load(input_checkpoint, map_location=device)
    source_ae.load_state_dict(checkpoint["source_AE"])
    target_encoder.load_state_dict(checkpoint["target_weightencoder"])
    emb_classifier.load_state_dict(checkpoint["emb_dis_classifier"])
    exp_classifier.load_state_dict(checkpoint["exp_dis_classifier"])
    if "center_criterion" in checkpoint:
        center.load_state_dict(checkpoint["center_criterion"])
        center_source = "checkpoint"
    else:
        center_source = (
            f"reconstructed_from_original_train_seed_{args.original_train_seed}"
        )

    # Match gradient_ascent.py: optimize the complete target-loss path while
    # keeping the two decoders and the fixed center anchors unchanged.
    _freeze(source_ae.decoder)
    _freeze(target_encoder.decoder)
    _freeze(center)
    source_ae.eval()
    target_encoder.eval()
    emb_classifier.eval()
    exp_classifier.eval()
    center.eval()
    groups = [
        ("source_encoder", list(source_ae.encoder.parameters())),
        ("target_Q", list(target_encoder.Q.parameters())),
        ("target_K", list(target_encoder.K.parameters())),
        ("latent_classifier", list(emb_classifier.parameters())),
        ("expression_classifier", list(exp_classifier.parameters())),
    ]
    trainable = [parameter for _, parameters in groups for parameter in parameters]
    optimizer = torch.optim.Adam(
        [{"name": name, "params": parameters} for name, parameters in groups],
        lr=args.lr,
    )
    models = (source_ae, target_encoder, emb_classifier, exp_classifier)

    set_seed(args.unlearn_seed)
    forget_dataset = Subset(target_dataset, forget_indices)
    retain_dataset = Subset(target_dataset, retain_indices)
    # Retain defines an epoch. Draw the same number of forget examples with
    # replacement so every retain batch has an equally sized forget batch.
    forget_sampler = RandomSampler(
        forget_dataset,
        replacement=True,
        num_samples=len(retain_indices),
        generator=torch.Generator().manual_seed(args.unlearn_seed),
    )
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        sampler=forget_sampler,
        drop_last=False,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.unlearn_seed + 1),
    )
    forget_eval_loader = DataLoader(
        Subset(target_dataset, forget_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    retain_eval_loader = DataLoader(
        Subset(target_dataset, retain_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
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

    input_forget = evaluate(forget_eval_loader)
    input_retain = evaluate(retain_eval_loader)
    history = [
        {
            "epoch": 0,
            "optimizer_steps": 0,
            "cumulative_optimizer_steps": 0,
            "gradient_norm": None,
            **{
                f"train_forget_{name}": None
                for name in LOSS_NAMES
            },
            **{
                f"train_retain_{name}": None
                for name in LOSS_NAMES
            },
            "train_joint_objective": None,
            "joint_objective": input_retain["task"] - input_forget["task"],
            **{f"forget_{key}": value for key, value in input_forget.items()},
            **{f"retain_{key}": value for key, value in input_retain.items()},
        }
    ]
    effective_batch = min(args.batch_size, len(retain_indices))
    print(
        f"[data] forget={len(forget_indices)} retain={len(retain_indices)} "
        f"micro_batch={args.batch_size}"
    )
    print(
        f"[batch] optimizer_steps_per_epoch={len(retain_loader)} "
        f"retain_samples_per_epoch={len(retain_indices)} "
        f"paired_forget_draws_per_epoch={len(retain_indices)} "
        f"effective_batch<={effective_batch}"
    )
    print(
        f"[input baseline] forget={input_forget['task']:.6f} "
        f"retain={input_retain['task']:.6f}"
    )

    cumulative_steps = 0
    for epoch in range(1, args.epochs + 1):
        source_ae.encoder.train()
        target_encoder.Q.train()
        target_encoder.K.train()
        emb_classifier.train()
        exp_classifier.train()

        step_norms = []
        train_sums = {
            assignment: {name: 0.0 for name in LOSS_NAMES}
            for assignment in ("forget", "retain")
        }
        train_sample_count = 0
        for forget_batch, retain_batch in zip(forget_loader, retain_loader):
            optimizer.zero_grad(set_to_none=True)
            forget_gex, _, forget_labels = forget_batch
            retain_gex, _, retain_labels = retain_batch
            if len(forget_gex) != len(retain_gex):
                raise RuntimeError("paired forget/retain batch sizes do not match")
            forget_gex = forget_gex.to(device)
            forget_labels = forget_labels.to(device)
            retain_gex = retain_gex.to(device)
            retain_labels = retain_labels.to(device)

            forget_output = forward_aligner(models, forget_gex, source_gex)
            forget_losses = alignment_losses(
                forget_output,
                forget_gex,
                forget_labels,
                center,
                args.recon_weight,
                args.class_weight,
                args.center_weight,
            )
            retain_output = forward_aligner(models, retain_gex, source_gex)
            retain_losses = alignment_losses(
                retain_output,
                retain_gex,
                retain_labels,
                center,
                args.recon_weight,
                args.class_weight,
                args.center_weight,
            )
            objective = retain_losses["task"] - forget_losses["task"]
            if not all(
                torch.isfinite(value)
                for value in (
                    objective,
                    forget_losses["task"],
                    retain_losses["task"],
                )
            ):
                raise RuntimeError(f"non-finite joint objective at epoch {epoch}")

            batch_size = len(forget_gex)
            for name in LOSS_NAMES:
                train_sums["forget"][name] += (
                    forget_losses[name].detach().item() * batch_size
                )
                train_sums["retain"][name] += (
                    retain_losses[name].detach().item() * batch_size
                )
            train_sample_count += batch_size
            objective.backward()

            group_norms = {
                name: _gradient_norm(parameters) for name, parameters in groups
            }
            total_norm = _gradient_norm(trainable)
            if not math.isfinite(total_norm) or any(
                not math.isfinite(value) for value in group_norms.values()
            ):
                raise RuntimeError(
                    "non-finite gradient encountered before optimizer step"
                )
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            step_norms.append((total_norm, group_norms))

        optimizer_steps = len(step_norms)
        cumulative_steps += optimizer_steps
        gradient_norm = sum(total for total, _ in step_norms) / optimizer_steps
        mean_group_norms = {
            name: sum(norms[name] for _, norms in step_norms) / optimizer_steps
            for name, _ in groups
        }
        train_means = {
            assignment: {
                name: value / train_sample_count
                for name, value in sums.items()
            }
            for assignment, sums in train_sums.items()
        }
        train_joint_objective = (
            train_means["retain"]["task"] - train_means["forget"]["task"]
        )
        forget_metrics = evaluate(forget_eval_loader)
        retain_metrics = evaluate(retain_eval_loader)
        joint_objective = retain_metrics["task"] - forget_metrics["task"]
        if not all(
            math.isfinite(value)
            for value in (
                forget_metrics["task"],
                retain_metrics["task"],
                joint_objective,
            )
        ):
            raise RuntimeError(f"non-finite evaluated loss at epoch {epoch}")
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "cumulative_optimizer_steps": cumulative_steps,
                "gradient_norm": gradient_norm,
                **{
                    f"train_forget_{name}": value
                    for name, value in train_means["forget"].items()
                },
                **{
                    f"train_retain_{name}": value
                    for name, value in train_means["retain"].items()
                },
                "train_joint_objective": train_joint_objective,
                "joint_objective": joint_objective,
                **{
                    f"grad_{name}": value
                    for name, value in mean_group_norms.items()
                },
                **{f"forget_{key}": value for key, value in forget_metrics.items()},
                **{f"retain_{key}": value for key, value in retain_metrics.items()},
            }
        )
        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"forget={forget_metrics['task']:.6f} "
            f"retain={retain_metrics['task']:.6f} "
            f"train_joint={train_joint_objective:.6f} "
            f"eval_joint={joint_objective:.6f} steps={optimizer_steps}"
        )

    final_forget = evaluate(forget_eval_loader)
    final_retain = evaluate(retain_eval_loader)
    torch.save(
        {
            "epoch": checkpoint.get("epoch"),
            "source_AE": source_ae.state_dict(),
            "target_weightencoder": target_encoder.state_dict(),
            "emb_dis_classifier": emb_classifier.state_dict(),
            "exp_dis_classifier": exp_classifier.state_dict(),
            "center_criterion": center.state_dict(),
            "optimizer": optimizer.state_dict(),
            "joint_unlearning_epochs": args.epochs,
            "joint_unlearning_optimizer_steps": cumulative_steps,
            "baseline_checkpoint": str(input_checkpoint.resolve()),
            "split_dir": str(Path(args.split_dir).resolve()),
            "joint_unlearning_config": vars(args),
        },
        checkpoint_path,
    )

    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    _plot_history(history, output_dir / "loss_curve.png", args.loss_scale)

    summary = {
        "objective": "minimize_equal_weight_retain_loss_minus_forget_loss",
        "baseline_checkpoint": str(input_checkpoint.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "trainable_groups": [name for name, _ in groups],
        "frozen_groups": ["source_decoder", "target_decoder", "center"],
        "forget_coefficient": -1.0,
        "retain_coefficient": 1.0,
        "retain_samples_per_epoch": len(retain_indices),
        "paired_forget_draws_per_epoch": len(retain_indices),
        "completed_epochs": args.epochs,
        "micro_batch_size": args.batch_size,
        "effective_batch_size": effective_batch,
        "optimizer_steps": cumulative_steps,
        "loss_curve": str((output_dir / "loss_curve.png").resolve()),
        "center_source": center_source,
        "input_baseline_forget": input_forget,
        "input_baseline_retain": input_retain,
        "final_forget": final_forget,
        "final_retain": final_retain,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"[done] completed_epochs={args.epochs}")
    print(f"[done] checkpoint={checkpoint_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument("--source", default="GDSC")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="baseline THERAPI aligner checkpoint",
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-train-seed", type=int, default=0)
    parser.add_argument("--unlearn-seed", type=int, default=0)
    parser.add_argument("--tissue-column", default="tissue_label")
    parser.add_argument("--info-id-column", default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument(
        "--loss-scale",
        choices=("linear", "log", "symlog"),
        default="symlog",
        help="y-axis scale; signed objective always uses symlog when log is selected",
    )
    joint_unlearn(parser.parse_args())
