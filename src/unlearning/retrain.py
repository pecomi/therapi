"""Retrain the original THERAPI aligner from scratch using retain TCGA only."""

from __future__ import annotations

import argparse
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
from unlearning.loss_history import plot_history, split_metrics_row, write_history
from unlearning.objective import evaluate_loader
from unlearning.split import build_sample_table, load_manifest_indices
from utils import set_seed


def retrain(args: argparse.Namespace) -> None:
    """Run the original source+target training, replacing TCGA with retain TCGA."""
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    requested_output = Path(args.output_dir)
    # Accept either a run directory or its ckpts directory.  Model artifacts
    # always live under ckpts, matching pipline.sh's run layout.
    output_dir = (
        requested_output
        if requested_output.name.lower() == "ckpts"
        else requested_output / "ckpts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(data_dir / args.source / f"{args.source}_gex.csv", index_col=0)
    source_info = pd.read_csv(data_dir / args.source / f"{args.source}_info.csv")
    target_df = pd.read_csv(
        data_dir / args.target / f"{args.target}_unlabeled_gex.csv", index_col=0
    )
    target_info = pd.read_csv(data_dir / args.target / f"{args.target}_unlabeled_info.csv")
    sample_table = build_sample_table(
        target_df,
        target_info,
        tissue_column=args.tissue_column,
        info_id_column=args.info_id_column,
    )
    forget_indices, retain_indices = load_manifest_indices(sample_table, args.split_dir)

    set_seed(args.seed)
    num_tissue = int(source_info["tissue_label"].nunique())
    source_dataset = AlignerDataset(source_df, args.source, source_info["tissue_label"])
    target_dataset = AlignerDataset(target_df, args.target, target_info[args.tissue_column])
    retain_loader = DataLoader(
        Subset(target_dataset, retain_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
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

    # Keep model construction, losses, and optimizer equivalent to train_aligner.py.
    source_ae = SOURCE_AE(source_dataset.n_genes, num_tissue, args.latent_dim).to(device)
    target_encoder = TARGET_weightencoder(
        target_dataset.n_genes, args.latent_dim, source_df.shape[0]
    ).to(device)
    emb_classifier = Emb_Dis_classifier(args.latent_dim, num_tissue).to(device)
    exp_classifier = Exp_Dis_classifier(
        target_dataset.n_genes, args.latent_dim, num_tissue
    ).to(device)
    center_criterion = CenterLoss(
        num_classes=num_tissue, feat_dim=args.latent_dim, device=device
    ).to(device)
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(source_ae.parameters())
        + list(target_encoder.parameters())
        + list(emb_classifier.parameters())
        + list(exp_classifier.parameters()),
        lr=args.lr,
    )

    source_gex = source_dataset.data.to(device)
    source_labels = source_dataset.dis_label.to(device)
    models = (source_ae, target_encoder, emb_classifier, exp_classifier)

    def evaluate(loader):
        return evaluate_loader(
            loader,
            models,
            center_criterion,
            source_gex,
            args.recon_weight,
            args.class_weight,
            args.center_weight,
        )

    initial_forget = evaluate(forget_eval_loader)
    initial_retain = evaluate(retain_eval_loader)
    history = [
        split_metrics_row(
            0,
            initial_forget,
            initial_retain,
            train_total=None,
            train_source=None,
            train_target_retain=None,
        )
    ]
    print(
        f"[data] forget_samples(excluded)={len(forget_indices)} "
        f"retain_samples(training)={len(retain_indices)}"
    )
    print(
        f"[epoch 0 before update] target_forget={initial_forget['task']:.6f} "
        f"target_retain={initial_retain['task']:.6f}"
    )
    for epoch in range(args.epochs):
        source_ae.train()
        target_encoder.train()
        emb_classifier.train()
        exp_classifier.train()
        sums = {"total": 0.0, "source": 0.0, "target": 0.0}

        for target_gex, _, target_labels in retain_loader:
            target_gex = target_gex.to(device)
            target_labels = target_labels.to(device)
            source_z, source_recon = source_ae(source_gex)
            source_loss = (
                args.recon_weight * mse(source_recon, source_gex)
                + args.center_weight * center_criterion(source_z, source_labels)
                + args.class_weight
                * (
                    cross_entropy(emb_classifier(source_z), source_labels)
                    + cross_entropy(exp_classifier(source_recon), source_labels)
                )
            )

            _, target_latent, target_wgex, target_recon = target_encoder(
                target_gex, source_z, source_gex
            )
            target_loss = (
                args.recon_weight * mse(target_recon, target_gex)
                + args.center_weight * center_criterion(target_latent, target_labels)
                + args.class_weight
                * (
                    cross_entropy(emb_classifier(target_latent), target_labels)
                    + cross_entropy(exp_classifier(target_wgex), target_labels)
                )
            )
            total = source_loss + target_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            sums["total"] += total.item()
            sums["source"] += source_loss.item()
            sums["target"] += target_loss.item()

        train_means = {
            name: value / len(retain_loader) for name, value in sums.items()
        }
        forget_metrics = evaluate(forget_eval_loader)
        retain_metrics = evaluate(retain_eval_loader)
        row = split_metrics_row(
            epoch + 1,
            forget_metrics,
            retain_metrics,
            train_total=train_means["total"],
            train_source=train_means["source"],
            train_target_retain=train_means["target"],
        )
        history.append(row)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"target_forget={row['forget_task']:.6f} "
            f"target_retain={row['retain_task']:.6f} "
            f"mini_batch_train={row['train_total']:.6f}"
        )

    checkpoint_path = output_dir / f"THERAPI_aligner_{args.source}_{args.target}.pt"
    torch.save(
        {
            "epoch": args.epochs - 1,
            "source_AE": source_ae.state_dict(),
            "target_weightencoder": target_encoder.state_dict(),
            "emb_dis_classifier": emb_classifier.state_dict(),
            "exp_dis_classifier": exp_classifier.state_dict(),
            "center_criterion": center_criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_data": "GDSC_plus_retain_TCGA_only",
            "split_dir": str(Path(args.split_dir).resolve()),
            "config": vars(args),
        },
        checkpoint_path,
    )
    history_path = output_dir / "history.csv"
    curve_path = output_dir / "loss_curve.png"
    write_history(history, history_path)
    plot_history(
        history,
        curve_path,
        args.loss_scale,
        epoch_label="retraining epoch",
    )
    print(
        f"[final post-update target mean] forget={history[-1]['forget_task']:.6f} "
        f"retain={history[-1]['retain_task']:.6f}"
    )
    print(f"[done] retain-only retrained checkpoint -> {checkpoint_path.resolve()}")
    print(f"[done] history={history_path.resolve()} curve={curve_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument("--source", default="GDSC")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tissue-column", default="tissue_label")
    parser.add_argument("--info-id-column", default=None)
    parser.add_argument("--epochs", type=int, default=199)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument(
        "--loss-scale", choices=("linear", "log", "symlog"), default="log"
    )
    retrain(parser.parse_args())
