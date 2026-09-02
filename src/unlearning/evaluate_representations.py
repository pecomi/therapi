"""Compare aligner losses and latent geometry for baseline/unlearned/retrained models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
from unlearning.objective import per_sample_alignment_losses
from unlearning.split import build_sample_table, load_manifest_indices
from utils import set_seed

LOSS_NAMES = ("task", "recon", "emb_class", "exp_class", "center")
MODEL_NAMES = ("baseline", "unlearned", "retrained")
COMPARISONS = (
    ("baseline", "unlearned"),
    ("baseline", "retrained"),
    ("unlearned", "retrained"),
)
LEGACY_OUTPUTS = (
    "representation_change_per_sample.csv",
    "representation_change_per_patient.csv",
    "representation_change_group_summary.csv",
    "representation_change_summary.json",
)


def _load_aligner(
    checkpoint_path: str | Path,
    *,
    n_genes: int,
    n_source: int,
    n_tissue: int,
    latent_dim: int,
    device: torch.device,
    original_train_seed: int,
):
    """Recreate the original initialization order and load one aligner."""
    set_seed(original_train_seed, logger=lambda _: None)
    source_ae = SOURCE_AE(n_genes, n_tissue, latent_dim).to(device)
    target_encoder = TARGET_weightencoder(n_genes, latent_dim, n_source).to(device)
    emb_classifier = Emb_Dis_classifier(latent_dim, n_tissue).to(device)
    exp_classifier = Exp_Dis_classifier(n_genes, latent_dim, n_tissue).to(device)
    center = CenterLoss(
        num_classes=n_tissue, feat_dim=latent_dim, device=device
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_ae.load_state_dict(checkpoint["source_AE"])
    target_encoder.load_state_dict(checkpoint["target_weightencoder"])
    emb_classifier.load_state_dict(checkpoint["emb_dis_classifier"])
    exp_classifier.load_state_dict(checkpoint["exp_dis_classifier"])
    if "center_criterion" in checkpoint:
        center.load_state_dict(checkpoint["center_criterion"])
        center_source = "checkpoint"
    else:
        center_source = f"reconstructed_from_seed_{original_train_seed}"

    modules = (source_ae, target_encoder, emb_classifier, exp_classifier, center)
    for module in modules:
        module.eval()
    return modules, center_source


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA for paired samples without constructing n-by-n Gram matrices."""
    x = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    y = y.astype(np.float64) - y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(
        y.T @ y, ord="fro"
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T


def _frechet_distance(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    """Gaussian Frechet distance between two latent distributions."""
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    cov_x = np.atleast_2d(np.cov(x, rowvar=False))
    cov_y = np.atleast_2d(np.cov(y, rowvar=False))
    identity = np.eye(cov_x.shape[0])
    cov_x, cov_y = cov_x + eps * identity, cov_y + eps * identity
    root_x = _psd_sqrt(cov_x)
    trace_cross = np.trace(_psd_sqrt(root_x @ cov_y @ root_x))
    distance = (mean_x - mean_y) @ (mean_x - mean_y)
    distance += np.trace(cov_x) + np.trace(cov_y) - 2 * trace_cross
    return float(max(distance, 0.0))


def _patient_means(values: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    frame["patient_id"] = patient_ids
    return frame.groupby("patient_id", sort=True).mean().to_numpy()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
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
    samples = samples.copy()
    samples["assignment"] = "retain"
    samples.loc[samples["row_index"].isin(forget_indices), "assignment"] = "forget"

    n_tissue = int(source_info["tissue_label"].nunique())
    source_dataset = AlignerDataset(
        source_df, args.source, source_info["tissue_label"]
    )
    target_dataset = AlignerDataset(
        target_df, args.target, target_info[args.tissue_column]
    )
    common = dict(
        n_genes=source_dataset.n_genes,
        n_source=len(source_dataset),
        n_tissue=n_tissue,
        latent_dim=args.latent_dim,
        device=device,
        original_train_seed=args.original_train_seed,
    )
    loaded = {
        "baseline": _load_aligner(args.baseline_checkpoint, **common),
        "unlearned": _load_aligner(args.unlearned_checkpoint, **common),
        "retrained": _load_aligner(args.retrained_checkpoint, **common),
    }
    models = {name: value[0] for name, value in loaded.items()}
    center_sources = {name: value[1] for name, value in loaded.items()}

    baseline_center = models["baseline"][4].centers.detach()
    center_differences = {
        name: (baseline_center - models[name][4].centers.detach()).abs().max().item()
        for name in ("unlearned", "retrained")
    }
    center_max_abs_diff = max(center_differences.values())
    if center_max_abs_diff > args.center_tolerance:
        raise ValueError(
            "the three models do not share fixed centers "
            f"(max_abs_diff={center_max_abs_diff:.8g})"
        )

    source_gex = source_dataset.data.to(device)
    source_latents = {
        name: modules[0].encoder(source_gex) for name, modules in models.items()
    }
    loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loss_chunks = {
        model: {loss: [] for loss in LOSS_NAMES} for model in MODEL_NAMES
    }
    latent_chunks = {model: [] for model in MODEL_NAMES}

    for target_gex, _, labels in loader:
        target_gex = target_gex.to(device)
        labels = labels.to(device)
        for model_name in MODEL_NAMES:
            _, target_encoder, emb_classifier, exp_classifier, center = models[
                model_name
            ]
            _, latent, weighted_gex, reconstruction = target_encoder(
                target_gex, source_latents[model_name], source_gex
            )
            output = {
                "recon": reconstruction,
                "latent": latent,
                "emb_logits": emb_classifier(latent),
                "exp_logits": exp_classifier(weighted_gex),
            }
            losses = per_sample_alignment_losses(
                output,
                target_gex,
                labels,
                center,
                args.recon_weight,
                args.class_weight,
                args.center_weight,
            )
            for loss_name in LOSS_NAMES:
                loss_chunks[model_name][loss_name].append(
                    losses[loss_name].detach().cpu().numpy()
                )
            latent_chunks[model_name].append(latent.detach().cpu().numpy())

    sample_losses = {
        model: {loss: np.concatenate(chunks) for loss, chunks in losses.items()}
        for model, losses in loss_chunks.items()
    }
    sample_latents = {
        model: np.concatenate(chunks) for model, chunks in latent_chunks.items()
    }
    sample_assignments = samples["assignment"].to_numpy()
    sample_patient_ids = samples["patient_id"].to_numpy()

    patient_metadata = samples.groupby("patient_id", sort=True).agg(
        assignment=("assignment", "first")
    )
    patient_assignments = patient_metadata["assignment"].to_numpy()
    patient_losses = {
        model: {
            loss: _patient_means(values[:, None], sample_patient_ids).ravel()
            for loss, values in losses.items()
        }
        for model, losses in sample_losses.items()
    }
    patient_latents = {
        model: _patient_means(values, sample_patient_ids)
        for model, values in sample_latents.items()
    }

    loss_rows = []
    for unit, values, assignments in (
        ("sample", sample_losses, sample_assignments),
        ("patient", patient_losses, patient_assignments),
    ):
        for assignment in ("forget", "retain"):
            mask = assignments == assignment
            for model_name in MODEL_NAMES:
                loss_rows.append(
                    {
                        "unit": unit,
                        "assignment": assignment,
                        "model": model_name,
                        "n": int(mask.sum()),
                        **{
                            loss_name: float(values[model_name][loss_name][mask].mean())
                            for loss_name in LOSS_NAMES
                        },
                    }
                )
    loss_metrics = pd.DataFrame(loss_rows)

    similarity_rows = []
    for unit, values, assignments in (
        ("sample", sample_latents, sample_assignments),
        ("patient", patient_latents, patient_assignments),
    ):
        for assignment in ("forget", "retain"):
            mask = assignments == assignment
            for left, right in COMPARISONS:
                x, y = values[left][mask], values[right][mask]
                similarity_rows.append(
                    {
                        "unit": unit,
                        "assignment": assignment,
                        "comparison": f"{left}_vs_{right}",
                        "n": len(x),
                        "linear_cka": _linear_cka(x, y),
                        "frechet_latent_distance": (
                            _frechet_distance(x, y, args.frechet_eps)
                            if len(x) > 1
                            else float("nan")
                        ),
                    }
                )
    representation_similarity = pd.DataFrame(similarity_rows)

    loss_metrics.to_csv(output_dir / "loss_metrics.csv", index=False)
    representation_similarity.to_csv(
        output_dir / "representation_similarity.csv", index=False
    )
    summary = {
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "unlearned_checkpoint": str(Path(args.unlearned_checkpoint).resolve()),
        "retrained_checkpoint": str(Path(args.retrained_checkpoint).resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "n_forget_samples": len(forget_indices),
        "n_retain_samples": len(retain_indices),
        "n_forget_patients": int((patient_assignments == "forget").sum()),
        "n_retain_patients": int((patient_assignments == "retain").sum()),
        "center_sources": center_sources,
        "center_max_abs_diff": center_max_abs_diff,
        "loss_weights": {
            "reconstruction_mse": args.recon_weight,
            "classification_cross_entropy": args.class_weight,
            "center": args.center_weight,
        },
        "reported_metrics": {
            "loss_metrics.csv": list(LOSS_NAMES),
            "representation_similarity.csv": [
                "linear_cka",
                "frechet_latent_distance",
            ],
        },
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for filename in LEGACY_OUTPUTS:
        (output_dir / filename).unlink(missing_ok=True)

    print(f"[data] forget_samples={len(forget_indices)} retain_samples={len(retain_indices)}")
    print(
        f"[data] forget_patients={summary['n_forget_patients']} "
        f"retain_patients={summary['n_retain_patients']}"
    )
    print("\n[loss metrics]")
    print(loss_metrics.to_string(index=False))
    print("\n[latent CKA / Frechet distance]")
    print(representation_similarity.to_string(index=False))
    print(f"[done] results -> {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument("--source", default="GDSC")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--unlearned-checkpoint", required=True)
    parser.add_argument("--retrained-checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-train-seed", type=int, default=0)
    parser.add_argument("--tissue-column", default="tissue_label")
    parser.add_argument("--info-id-column", default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--center-tolerance", type=float, default=1e-7)
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.8)
    parser.add_argument("--frechet-eps", type=float, default=1e-6)
    evaluate(parser.parse_args())
