"""Joint forget-ascent and retain-descent updates from a baseline aligner."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import cycle
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
from unlearning.loss_history import (
    format_epoch_log,
    plot_history,
    plot_retain_history,
    split_metrics_row,
    write_history,
)
from unlearning.objective import (
    EVALUATION_BATCH_SIZE,
    alignment_losses,
    evaluate_loader,
    forward_aligner,
    neggrad_plus_objective,
)
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


def _save_loss_outputs(history, output_dir: Path, loss_scale: str) -> None:
    """Persist every completed epoch so partial runs still have a curve."""
    write_history(history, output_dir / "history.csv")
    plot_history(
        history,
        output_dir / "loss_curve.png",
        loss_scale,
        epoch_label="NegGrad+ epoch",
    )
    plot_retain_history(
        history,
        output_dir / "retain_loss_curve.png",
        loss_scale,
        epoch_label="NegGrad+ epoch",
    )


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")
    loss_weights = (args.recon_weight, args.class_weight, args.center_weight)
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError(
            "loss weights must be non-negative and at least one must be positive"
        )


def joint_unlearn(args: argparse.Namespace) -> None:
    """Run NegGrad+ by minimizing beta-weighted retain and forget losses."""
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
    set_seed(args.original_train_seed, logger=lambda _: None)
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

    set_seed(args.unlearn_seed, logger=lambda _: None)
    forget_dataset = Subset(target_dataset, forget_indices)
    retain_dataset = Subset(target_dataset, retain_indices)
    # Retain defines an epoch. The shuffled forget loader is cycled until every
    # retain batch is consumed. A new deterministic shuffle is produced when a
    # new outer epoch constructs a fresh iterator over the loader.
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.unlearn_seed),
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
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
    )
    retain_eval_loader = DataLoader(
        Subset(target_dataset, retain_indices),
        batch_size=EVALUATION_BATCH_SIZE,
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
    evaluation_objective = neggrad_plus_objective(
        input_forget["task"], input_retain["task"], args.beta
    )
    history = [
        split_metrics_row(
            0,
            input_forget,
            input_retain,
            evaluation_objective=evaluation_objective,
        )
    ]
    print(
        f"[setup] forget_samples={len(forget_indices)} "
        f"retain_samples={len(retain_indices)} batch_size={args.batch_size} "
        f"beta={args.beta:.6f} "
        f"original_train_seed={args.original_train_seed} "
        f"unlearn_seed={args.unlearn_seed}"
    )
    print(format_epoch_log(history[-1], args.epochs))
    _save_loss_outputs(history, output_dir, args.loss_scale)

    cumulative_steps = 0
    cumulative_forget_samples = 0
    cumulative_retain_samples = 0
    for epoch in range(1, args.epochs + 1):
        source_ae.encoder.train()
        target_encoder.Q.train()
        target_encoder.K.train()
        emb_classifier.train()
        exp_classifier.train()

        step_norms = []
        forget_samples_seen = 0
        retain_samples_seen = 0
        for forget_batch, retain_batch in zip(cycle(forget_loader), retain_loader):
            optimizer.zero_grad(set_to_none=True)
            forget_gex, _, forget_labels = forget_batch
            retain_gex, _, retain_labels = retain_batch
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
            objective = neggrad_plus_objective(
                forget_losses["task"], retain_losses["task"], args.beta
            )
            if not all(
                torch.isfinite(value)
                for value in (
                    objective,
                    forget_losses["task"],
                    retain_losses["task"],
                )
            ):
                raise RuntimeError(f"non-finite joint objective at epoch {epoch}")
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
            optimizer.step()
            step_norms.append((total_norm, group_norms))
            forget_samples_seen += len(forget_gex)
            retain_samples_seen += len(retain_gex)

        optimizer_steps = len(step_norms)
        cumulative_steps += optimizer_steps
        cumulative_forget_samples += forget_samples_seen
        cumulative_retain_samples += retain_samples_seen
        gradient_norm = sum(total for total, _ in step_norms) / optimizer_steps
        mean_group_norms = {
            name: sum(norms[name] for _, norms in step_norms) / optimizer_steps
            for name, _ in groups
        }
        forget_metrics = evaluate(forget_eval_loader)
        retain_metrics = evaluate(retain_eval_loader)
        evaluation_objective = neggrad_plus_objective(
            forget_metrics["task"], retain_metrics["task"], args.beta
        )
        if not all(
            math.isfinite(value)
            for value in (
                forget_metrics["task"],
                retain_metrics["task"],
                evaluation_objective,
            )
        ):
            raise RuntimeError(f"non-finite evaluated loss at epoch {epoch}")
        history.append(
            split_metrics_row(
                epoch,
                forget_metrics,
                retain_metrics,
                evaluation_objective=evaluation_objective,
                gradient_norm=gradient_norm,
                **{
                    f"grad_{name}": value
                    for name, value in mean_group_norms.items()
                },
            )
        )
        _save_loss_outputs(history, output_dir, args.loss_scale)
        print(format_epoch_log(history[-1], args.epochs))

    final_forget = forget_metrics
    final_retain = retain_metrics
    torch.save(
        {
            "epoch": checkpoint.get("epoch"),
            "source_AE": source_ae.state_dict(),
            "target_weightencoder": target_encoder.state_dict(),
            "emb_dis_classifier": emb_classifier.state_dict(),
            "exp_dis_classifier": exp_classifier.state_dict(),
            "center_criterion": center.state_dict(),
            "optimizer": optimizer.state_dict(),
            "method": "neggrad_plus",
            "completed_epochs": args.epochs,
            "optimizer_steps": cumulative_steps,
            "original_checkpoint": str(input_checkpoint.resolve()),
            "split_dir": str(Path(args.split_dir).resolve()),
            "config": vars(args),
        },
        checkpoint_path,
    )

    summary = {
        "method": "neggrad_plus",
        "objective": "minimize_beta_retain_loss_minus_one_minus_beta_forget_loss",
        "objective_coefficients": {
            "forget": -(1.0 - args.beta),
            "retain": args.beta,
        },
        "original_checkpoint": str(input_checkpoint.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "history": str((output_dir / "history.csv").resolve()),
        "loss_curve": str((output_dir / "loss_curve.png").resolve()),
        "retain_loss_curve": str((output_dir / "retain_loss_curve.png").resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "trainable_groups": [name for name, _ in groups],
        "frozen_groups": ["source_decoder", "target_decoder", "center"],
        "sampling": "retain_epoch_with_cycled_shuffled_forget_batches",
        "completed_epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer_steps": cumulative_steps,
        "forget_samples_seen": cumulative_forget_samples,
        "retain_samples_seen": cumulative_retain_samples,
        "effective_forget_passes": cumulative_forget_samples / len(forget_indices),
        "center_source": center_source,
        "config": vars(args),
        "initial_forget": input_forget,
        "initial_retain": input_retain,
        "final_forget": final_forget,
        "final_retain": final_retain,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"[done] completed_epochs={args.epochs} "
        f"checkpoint={checkpoint_path.resolve()} "
        f"history={(output_dir / 'history.csv').resolve()} "
        f"curve={(output_dir / 'loss_curve.png').resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="../data/")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument(
        "--loss-scale",
        choices=("linear", "log", "symlog"),
        default="log",
        help="y-axis scale for the saved full-set loss curve",
    )
    parser.add_argument("--beta", type=float, default=0.95)
    joint_unlearn(parser.parse_args())
