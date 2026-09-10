"""Multi-epoch forget-set gradient ascent for a trained THERAPI aligner."""

from __future__ import annotations

import argparse
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
from unlearning.loss_history import (
    RunLogger,
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
    neggrad_objective,
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


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    loss_weights = (args.recon_weight, args.class_weight, args.center_weight)
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError("loss weights must be non-negative and at least one must be positive")


def _save_loss_outputs(history, output_dir: Path, loss_scale: str) -> None:
    """Persist every completed epoch so partial runs still have a curve."""
    write_history(history, output_dir / "history.csv")
    plot_history(
        history,
        output_dir / "loss_curve.png",
        loss_scale,
        epoch_label="NegGrad epoch",
    )
    plot_retain_history(
        history,
        output_dir / "retain_loss_curve.png",
        loss_scale,
        epoch_label="NegGrad epoch",
    )


def unlearn(args: argparse.Namespace) -> None:
    _validate_args(args)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    requested_output = Path(args.output_dir)
    output_dir = requested_output if requested_output.name.lower() == "ckpts" else requested_output / "ckpts"
    checkpoint_path = output_dir / f"THERAPI_aligner_{args.source}_{args.target}.pt"
    input_checkpoint = Path(args.checkpoint)
    if checkpoint_path.resolve() == input_checkpoint.resolve():
        raise ValueError(
            "--output-dir would overwrite the input baseline checkpoint; "
            "use a separate run directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "training.log")

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
    set_seed(args.original_train_seed, logger=lambda _: None)
    source_ae = SOURCE_AE(source_dataset.n_genes, n_tissue, args.latent_dim).to(device)
    target_encoder = TARGET_weightencoder(target_dataset.n_genes, args.latent_dim, len(source_dataset)).to(device)
    emb_classifier = Emb_Dis_classifier(args.latent_dim, n_tissue).to(device)
    exp_classifier = Exp_Dis_classifier(target_dataset.n_genes, args.latent_dim, n_tissue).to(device)
    center = CenterLoss(num_classes=n_tissue, feat_dim=args.latent_dim, device=device).to(device)

    checkpoint = torch.load(input_checkpoint, map_location=device)
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

    set_seed(args.unlearn_seed, logger=lambda _: None)
    forget_loader = DataLoader(
        Subset(target_dataset, forget_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.unlearn_seed),
    )
    forget_eval_loader = DataLoader(
        Subset(target_dataset, forget_indices), batch_size=EVALUATION_BATCH_SIZE, shuffle=False
    )
    retain_eval_loader = DataLoader(
        Subset(target_dataset, retain_indices), batch_size=EVALUATION_BATCH_SIZE, shuffle=False
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
    history = [
        split_metrics_row(
            0,
            baseline_forget,
            baseline_retain,
            evaluation_objective=neggrad_objective(baseline_forget["task"]),
        )
    ]
    logger(
        f"[setup] forget_samples={len(forget_indices)} "
        f"retain_samples={len(retain_indices)} batch_size={args.batch_size} "
        f"original_train_seed={args.original_train_seed} "
        f"unlearn_seed={args.unlearn_seed}"
    )
    logger(format_epoch_log(history[-1], args.epochs))
    _save_loss_outputs(history, output_dir, args.loss_scale)

    cumulative_steps = 0

    def optimizer_update():
        norms = {name: _gradient_norm(parameters) for name, parameters in groups}
        total_norm = _gradient_norm(trainable)
        if not math.isfinite(total_norm) or any(
            not math.isfinite(value) for value in norms.values()
        ):
            raise RuntimeError("non-finite gradient encountered before optimizer step")
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
            objective = neggrad_objective(losses["task"])
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
        history.append(
            split_metrics_row(
                epoch,
                forget_metrics,
                retain_metrics,
                evaluation_objective=neggrad_objective(current_loss),
                gradient_norm=gradient_norm,
                **{f"grad_{name}": value for name, value in group_norms.items()},
            )
        )
        _save_loss_outputs(history, output_dir, args.loss_scale)
        logger(format_epoch_log(history[-1], args.epochs))

    final_forget, final_retain = forget_metrics, retain_metrics
    torch.save(
        {
            "epoch": checkpoint.get("epoch"),
            "method": "neggrad",
            "completed_epochs": args.epochs,
            "optimizer_steps": cumulative_steps,
            "source_AE": source_ae.state_dict(),
            "target_weightencoder": target_encoder.state_dict(),
            "emb_dis_classifier": emb_classifier.state_dict(),
            "exp_dis_classifier": exp_classifier.state_dict(),
            "center_criterion": center.state_dict(),
            "optimizer": optimizer.state_dict(),
            "original_checkpoint": str(input_checkpoint.resolve()),
            "split_dir": str(Path(args.split_dir).resolve()),
            "config": vars(args),
        },
        checkpoint_path,
    )
    summary = {
        "method": "neggrad",
        "objective": "minimize_negative_forget_target_loss",
        "objective_coefficients": {"forget": -1.0, "retain": 0.0},
        "sampling": "one_shuffled_forget_pass_per_epoch",
        "batch_size": args.batch_size,
        "optimizer_steps": cumulative_steps,
        "forget_samples_seen": args.epochs * len(forget_indices),
        "retain_samples_seen": 0,
        "effective_forget_passes": float(args.epochs),
        "completed_epochs": args.epochs,
        "center_source": center_source,
        "original_checkpoint": str(input_checkpoint.resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "trainable_groups": [name for name, _ in groups],
        "frozen_groups": ["source_decoder", "target_decoder", "center"],
        "config": vars(args),
        "initial_forget": baseline_forget,
        "initial_retain": baseline_retain,
        "final_forget": final_forget,
        "final_retain": final_retain,
        "checkpoint": str(checkpoint_path.resolve()),
        "training_log": str((output_dir / "training.log").resolve()),
        "history": str((output_dir / "history.csv").resolve()),
        "loss_curve": str((output_dir / "loss_curve.png").resolve()),
        "retain_loss_curve": str((output_dir / "retain_loss_curve.png").resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger(
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument(
        "--loss-scale", choices=("linear", "log", "symlog"), default="log",
        help="y-axis scale for the saved loss curve",
    )
    unlearn(parser.parse_args())
