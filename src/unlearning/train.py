"""Forget-only gradient-ascent unlearning for a trained THERAPI aligner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Support direct execution: python src/unlearning/train.py ...
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
from unlearning.split import build_sample_table, load_manifest_indices
from utils import set_seed


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _set_train_modes(source_ae, target_encoder, emb_classifier, exp_classifier) -> None:
    source_ae.train()
    target_encoder.train()
    emb_classifier.train()
    exp_classifier.train()
    # Decoders stay fixed, but the target decoder remains in the graph so its
    # reconstruction gradient still reaches Q/K and the source encoder.
    source_ae.decoder.eval()
    target_encoder.decoder.eval()


def _forward_forget(
    source_ae,
    target_encoder,
    emb_classifier,
    exp_classifier,
    target_gex,
    source_gex,
):
    source_z = source_ae.encoder(source_gex)
    weights, latent, wgex, recon = target_encoder(target_gex, source_z, source_gex)
    return {
        "weights": weights,
        "latent": latent,
        "wgex": wgex,
        "recon": recon,
        "emb_logits": emb_classifier(latent),
        "exp_logits": exp_classifier(wgex),
    }


def _alignment_losses(
    output,
    target_gex,
    labels,
    center_criterion,
    mse,
    cross_entropy,
    recon_weight,
    class_weight,
    center_weight,
):
    recon = mse(output["recon"], target_gex)
    emb_class = cross_entropy(output["emb_logits"], labels)
    exp_class = cross_entropy(output["exp_logits"], labels)
    center = center_criterion(output["latent"], labels)
    total = (
        recon_weight * recon
        + class_weight * (emb_class + exp_class)
        + center_weight * center
    )
    return {
        "total": total,
        "recon": recon,
        "emb_class": emb_class,
        "exp_class": exp_class,
        "center": center,
    }


@torch.no_grad()
def evaluate_loader(
    loader,
    source_ae,
    target_encoder,
    emb_classifier,
    exp_classifier,
    center_criterion,
    source_gex,
    recon_weight,
    class_weight,
    center_weight,
):
    modules = (source_ae, target_encoder, emb_classifier, exp_classifier, center_criterion)
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()

    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    sums = {
        "task": 0.0,
        "recon": 0.0,
        "emb_class": 0.0,
        "exp_class": 0.0,
        "center": 0.0,
    }
    correct_emb = 0
    correct_exp = 0
    count = 0
    for target_gex, _, labels in loader:
        target_gex = target_gex.to(source_gex.device)
        labels = labels.to(source_gex.device)
        output = _forward_forget(
            source_ae,
            target_encoder,
            emb_classifier,
            exp_classifier,
            target_gex,
            source_gex,
        )
        losses = _alignment_losses(
            output,
            target_gex,
            labels,
            center_criterion,
            mse,
            cross_entropy,
            recon_weight,
            class_weight,
            center_weight,
        )
        batch_size = len(target_gex)
        for metric, loss_key in (
            ("task", "total"),
            ("recon", "recon"),
            ("emb_class", "emb_class"),
            ("exp_class", "exp_class"),
            ("center", "center"),
        ):
            sums[metric] += losses[loss_key].item() * batch_size
        correct_emb += (output["emb_logits"].argmax(dim=1) == labels).sum().item()
        correct_exp += (output["exp_logits"].argmax(dim=1) == labels).sum().item()
        count += batch_size

    for module, was_training in zip(modules, modes):
        module.train(was_training)
    return {
        **{metric: value / count for metric, value in sums.items()},
        "emb_accuracy": correct_emb / count,
        "exp_accuracy": correct_exp / count,
        "n_samples": count,
    }


def _trainable_parameter_groups(
    source_ae,
    target_encoder,
    emb_classifier,
    exp_classifier,
    center_criterion,
):
    # Update every parameter on a TCGA loss path, except the two explicitly
    # frozen reconstruction decoders.
    _freeze(source_ae.decoder)
    _freeze(target_encoder.decoder)
    groups = [
        {"name": "source_encoder", "params": list(source_ae.encoder.parameters())},
        {"name": "target_Q", "params": list(target_encoder.Q.parameters())},
        {"name": "target_K", "params": list(target_encoder.K.parameters())},
        {"name": "latent_classifier", "params": list(emb_classifier.parameters())},
        {"name": "expression_classifier", "params": list(exp_classifier.parameters())},
    ]
    # The original THERAPI optimizer did not update CenterLoss parameters.
    # Keep the random anchors fixed while allowing center-loss gradients to
    # flow through latent representations into the encoders.
    _freeze(center_criterion)
    return groups


def _gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().pow(2).sum().item()
    return squared**0.5


def unlearn(args: argparse.Namespace) -> None:
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

    source_gex_df = pd.read_csv(data_dir / args.source / f"{args.source}_gex.csv", index_col=0)
    source_info = pd.read_csv(data_dir / args.source / f"{args.source}_info.csv")
    target_gex_df = pd.read_csv(
        data_dir / args.target / f"{args.target}_unlabeled_gex.csv", index_col=0
    )
    target_info = pd.read_csv(data_dir / args.target / f"{args.target}_unlabeled_info.csv")
    samples = build_sample_table(
        target_gex_df,
        target_info,
        tissue_column=args.tissue_column,
        info_id_column=args.info_id_column,
    )
    forget_indices, retain_indices = load_manifest_indices(samples, args.split_dir)

    num_tissue = int(source_info["tissue_label"].nunique())
    source_dataset = AlignerDataset(source_gex_df, args.source, source_info["tissue_label"])
    target_dataset = AlignerDataset(
        target_gex_df, args.target, target_info[args.tissue_column]
    )

    # Reproduce the original initialization order before loading model weights.
    # This reconstructs the fixed random centers used by train_aligner.py when
    # its seed and architecture match.
    set_seed(args.original_train_seed)
    source_ae = SOURCE_AE(source_dataset.n_genes, num_tissue, args.latent_dim).to(device)
    target_encoder = TARGET_weightencoder(
        target_dataset.n_genes, args.latent_dim, source_gex_df.shape[0]
    ).to(device)
    emb_classifier = Emb_Dis_classifier(args.latent_dim, num_tissue).to(device)
    exp_classifier = Exp_Dis_classifier(
        target_dataset.n_genes, args.latent_dim, num_tissue
    ).to(device)
    center_criterion = CenterLoss(
        num_classes=num_tissue, feat_dim=args.latent_dim, device=device
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    source_ae.load_state_dict(checkpoint["source_AE"])
    target_encoder.load_state_dict(checkpoint["target_weightencoder"])
    emb_classifier.load_state_dict(checkpoint["emb_dis_classifier"])
    exp_classifier.load_state_dict(checkpoint["exp_dis_classifier"])
    if "center_criterion" in checkpoint:
        center_criterion.load_state_dict(checkpoint["center_criterion"])
        center_source = "checkpoint"
    else:
        center_source = f"reconstructed_from_original_train_seed_{args.original_train_seed}"

    parameter_groups = _trainable_parameter_groups(
        source_ae,
        target_encoder,
        emb_classifier,
        exp_classifier,
        center_criterion,
    )
    optimizer = torch.optim.Adam(
        [{"params": group["params"], "name": group["name"]} for group in parameter_groups],
        lr=args.lr,
    )

    # The unlearning seed controls only forget-batch order and later RNG.
    set_seed(args.unlearn_seed)
    forget_loader = DataLoader(
        Subset(target_dataset, forget_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.unlearn_seed),
    )
    forget_eval_loader = DataLoader(
        Subset(target_dataset, forget_indices), batch_size=args.eval_batch_size, shuffle=False
    )
    retain_eval_loader = DataLoader(
        Subset(target_dataset, retain_indices), batch_size=args.eval_batch_size, shuffle=False
    )
    source_gex = source_dataset.data.to(device)
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()

    def evaluate(loader):
        return evaluate_loader(
            loader,
            source_ae,
            target_encoder,
            emb_classifier,
            exp_classifier,
            center_criterion,
            source_gex,
            args.recon_weight,
            args.class_weight,
            args.center_weight,
        )

    baseline_forget = evaluate(forget_eval_loader)
    baseline_retain = evaluate(retain_eval_loader)
    print(f"[data] forget_samples={len(forget_indices)} retain_samples={len(retain_indices)}")
    print(f"[center] {center_source}; center parameters are fixed")
    print(f"[trainable] {[group['name'] for group in parameter_groups]}")
    print(
        f"[baseline] forget_total={baseline_forget['task']:.6f} "
        f"retain_total={baseline_retain['task']:.6f}"
    )

    trainable = [
        parameter
        for group in parameter_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    # Accumulate the exact forget-set mean gradient, then update once.
    _set_train_modes(source_ae, target_encoder, emb_classifier, exp_classifier)
    optimizer.zero_grad(set_to_none=True)
    running = {
        "ascent_objective": 0.0,
        "forget_total": 0.0,
        "forget_recon": 0.0,
        "forget_emb_class": 0.0,
        "forget_exp_class": 0.0,
        "forget_center": 0.0,
    }
    n_forget = len(forget_indices)
    for target_gex, _, labels in forget_loader:
        target_gex = target_gex.to(device)
        labels = labels.to(device)
        output = _forward_forget(
            source_ae,
            target_encoder,
            emb_classifier,
            exp_classifier,
            target_gex,
            source_gex,
        )
        losses = _alignment_losses(
            output,
            target_gex,
            labels,
            center_criterion,
            mse,
            cross_entropy,
            args.recon_weight,
            args.class_weight,
            args.center_weight,
        )
        batch_fraction = len(target_gex) / n_forget
        ascent_objective = -args.forget_weight * losses["total"] * batch_fraction
        if not torch.isfinite(ascent_objective):
            raise RuntimeError("non-finite unlearning loss while accumulating forget gradient")
        ascent_objective.backward()

        running["ascent_objective"] += ascent_objective.item()
        for name in ("total", "recon", "emb_class", "exp_class", "center"):
            running[f"forget_{name}"] += losses[name].item() * batch_fraction

    norms = {
        group["name"]: _gradient_norm(group["params"])
        for group in parameter_groups
    }
    frozen_decoder_grad = any(
        parameter.grad is not None
        for decoder in (source_ae.decoder, target_encoder.decoder)
        for parameter in decoder.parameters()
    )
    print(f"[gradient-check] trainable_group_norms={norms}")
    print(f"[gradient-check] frozen_decoder_has_grad={frozen_decoder_grad}")

    # Intentionally disabled for the first experiment so an exploding update
    # remains observable. Re-enable this line after inspecting the result:
    # torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
    optimizer.step()  # The unlearning parameter update happens exactly once.

    forget_metrics = evaluate(forget_eval_loader)
    # Retain is evaluation-only and never participates in backpropagation.
    retain_metrics = evaluate(retain_eval_loader)
    history = [{
        "update": 1,
        **{f"train_{name}": value for name, value in running.items()},
        **{f"forget_{name}": value for name, value in forget_metrics.items()},
        **{f"retain_{name}": value for name, value in retain_metrics.items()},
    }]
    print(
        f"[update 1/1] forget_total={forget_metrics['task']:.6f} "
        f"forget_recon={forget_metrics['recon']:.6f} "
        f"forget_class={(forget_metrics['emb_class'] + forget_metrics['exp_class']):.6f} "
        f"forget_center={forget_metrics['center']:.6f} "
        f"retain_total(eval_only)={retain_metrics['task']:.6f}"
    )

    final_forget = evaluate(forget_eval_loader)
    final_retain = evaluate(retain_eval_loader)
    output_checkpoint = output_dir / "THERAPI_aligner_unlearned.pt"
    torch.save(
        {
            "epoch": checkpoint.get("epoch"),
            "unlearning_updates": 1,
            "source_AE": source_ae.state_dict(),
            "target_weightencoder": target_encoder.state_dict(),
            "emb_dis_classifier": emb_classifier.state_dict(),
            "exp_dis_classifier": exp_classifier.state_dict(),
            "center_criterion": center_criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "original_checkpoint": str(Path(args.checkpoint).resolve()),
            "split_dir": str(Path(args.split_dir).resolve()),
            "center_source": center_source,
            "trainable_groups": [group["name"] for group in parameter_groups],
            "unlearning_config": vars(args),
        },
        output_checkpoint,
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "completed_updates": 1,
        "objective": "forget_only_gradient_ascent",
        "center_source": center_source,
        "center_parameters_fixed": True,
        "trainable_groups": [group["name"] for group in parameter_groups],
        "baseline_forget": baseline_forget,
        "baseline_retain_evaluation_only": baseline_retain,
        "final_forget": final_forget,
        "final_retain_evaluation_only": final_retain,
        "checkpoint": str(output_checkpoint.resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[done] unlearned checkpoint -> {output_checkpoint.resolve()}")


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    unlearn(parser.parse_args())
