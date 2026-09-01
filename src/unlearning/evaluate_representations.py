"""Compare baseline, unlearned, and retain-only retrained THERAPI aligners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise Jensen-Shannon divergence between attention distributions."""
    eps = torch.finfo(p.dtype).eps
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=1)
        + (q * (q.log() - midpoint.log())).sum(dim=1)
    )


def _group_summary(
    frame: pd.DataFrame, unit: str, metrics: list[str]
) -> pd.DataFrame:
    summary = frame.groupby("assignment")[metrics].agg(
        ["count", "mean", "median", "std", "max"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "unit", unit)
    return summary.reset_index()


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA for paired samples without constructing n-by-n Gram matrices."""
    x = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    y = y.astype(np.float64) - y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T


def _frechet_distance(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    """FID-style Gaussian distance between two latent distributions."""
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
    # This also verifies row order, patient leakage, and non-empty assignments.
    forget_indices, retain_indices = load_manifest_indices(samples, args.split_dir)
    # build_sample_table contains audited sample metadata but not the saved
    # assignment. load_manifest_indices validates the manifest and returns its
    # exact row positions, so attach those assignments for group summaries.
    samples = samples.copy()
    samples["assignment"] = "retain"
    samples.loc[samples["row_index"].isin(forget_indices), "assignment"] = "forget"
    if int((samples["assignment"] == "forget").sum()) != len(forget_indices):
        raise AssertionError("failed to attach all forget assignments to sample metadata")

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
    baseline, baseline_center_source = _load_aligner(args.baseline_checkpoint, **common)
    unlearned, unlearned_center_source = _load_aligner(args.unlearned_checkpoint, **common)
    retrained, retrained_center_source = _load_aligner(args.retrained_checkpoint, **common)
    b_source, b_target, b_emb, b_exp, b_center = baseline
    u_source, u_target, u_emb, u_exp, u_center = unlearned
    r_source, r_target, r_emb, r_exp, r_center = retrained

    center_max_abs_diff_baseline_unlearned = (
        b_center.centers.detach() - u_center.centers.detach()
    ).abs().max().item()
    center_max_abs_diff_baseline_retrained = (
        b_center.centers.detach() - r_center.centers.detach()
    ).abs().max().item()
    center_max_abs_diff = max(
        center_max_abs_diff_baseline_unlearned,
        center_max_abs_diff_baseline_retrained,
    )
    if center_max_abs_diff > args.center_tolerance:
        raise ValueError(
            "the three models do not share fixed centers "
            f"(max_abs_diff={center_max_abs_diff:.8g}); comparison would mix center movement "
            "with representation movement"
        )

    source_gex = source_dataset.data.to(device)
    baseline_source_z = b_source.encoder(source_gex)
    unlearned_source_z = u_source.encoder(source_gex)
    retrained_source_z = r_source.encoder(source_gex)
    loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    collected: dict[str, list[np.ndarray]] = {}
    latent_chunks: dict[str, list[np.ndarray]] = {
        "baseline": [], "unlearned": [], "retrained": []
    }

    def collect(name: str, value: torch.Tensor) -> None:
        collected.setdefault(name, []).append(value.detach().cpu().numpy())

    for target_gex, _, labels in loader:
        target_gex = target_gex.to(device)
        labels = labels.to(device)
        b_weights, b_latent, b_wgex, b_recon = b_target(
            target_gex, baseline_source_z, source_gex
        )
        u_weights, u_latent, u_wgex, u_recon = u_target(
            target_gex, unlearned_source_z, source_gex
        )
        r_weights, r_latent, r_wgex, r_recon = r_target(
            target_gex, retrained_source_z, source_gex
        )

        b_emb_logits, u_emb_logits, r_emb_logits = b_emb(b_latent), u_emb(u_latent), r_emb(r_latent)
        b_exp_logits, u_exp_logits, r_exp_logits = b_exp(b_wgex), u_exp(u_wgex), r_exp(r_wgex)
        b_emb_prob = F.softmax(b_emb_logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        u_emb_prob = F.softmax(u_emb_logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        b_exp_prob = F.softmax(b_exp_logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        u_exp_prob = F.softmax(u_exp_logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        r_emb_prob = F.softmax(r_emb_logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        r_exp_prob = F.softmax(r_exp_logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        b_center_distance = (b_latent - b_center.centers[labels]).pow(2).sum(dim=1)
        u_center_distance = (u_latent - u_center.centers[labels]).pow(2).sum(dim=1)
        r_center_distance = (r_latent - r_center.centers[labels]).pow(2).sum(dim=1)

        outputs = {
            "baseline": (
                {"recon": b_recon, "latent": b_latent, "emb_logits": b_emb_logits, "exp_logits": b_exp_logits},
                b_center,
            ),
            "unlearned": (
                {"recon": u_recon, "latent": u_latent, "emb_logits": u_emb_logits, "exp_logits": u_exp_logits},
                u_center,
            ),
            "retrained": (
                {"recon": r_recon, "latent": r_latent, "emb_logits": r_emb_logits, "exp_logits": r_exp_logits},
                r_center,
            ),
        }
        for model_name, (output, center) in outputs.items():
            losses = per_sample_alignment_losses(
                output,
                target_gex,
                labels,
                center,
                args.recon_weight,
                args.class_weight,
                args.center_weight,
            )
            for loss_name, values in losses.items():
                collect(f"loss_{loss_name}_{model_name}", values)
        for model_name, latent in (
            ("baseline", b_latent), ("unlearned", u_latent), ("retrained", r_latent)
        ):
            latent_chunks[model_name].append(latent.detach().cpu().numpy())

        collect("latent_rmse", (u_latent - b_latent).pow(2).mean(dim=1).sqrt())
        collect("latent_rmse_baseline_unlearned", (u_latent - b_latent).pow(2).mean(dim=1).sqrt())
        collect("latent_rmse_baseline_retrained", (r_latent - b_latent).pow(2).mean(dim=1).sqrt())
        collect("latent_rmse_unlearned_retrained", (r_latent - u_latent).pow(2).mean(dim=1).sqrt())
        collect("latent_mae", (u_latent - b_latent).abs().mean(dim=1))
        collect("latent_cosine", F.cosine_similarity(b_latent, u_latent, dim=1))
        collect("attention_mae", (u_weights - b_weights).abs().mean(dim=1))
        collect("attention_js", _js_divergence(b_weights, u_weights))
        collect("attention_js_baseline_unlearned", _js_divergence(b_weights, u_weights))
        collect("attention_js_baseline_retrained", _js_divergence(b_weights, r_weights))
        collect("attention_js_unlearned_retrained", _js_divergence(u_weights, r_weights))
        collect("wgex_rmse", (u_wgex - b_wgex).pow(2).mean(dim=1).sqrt())
        collect("wgex_rmse_baseline_retrained", (r_wgex - b_wgex).pow(2).mean(dim=1).sqrt())
        collect("wgex_rmse_unlearned_retrained", (r_wgex - u_wgex).pow(2).mean(dim=1).sqrt())
        collect("reconstruction_output_rmse_baseline_retrained", (r_recon - b_recon).pow(2).mean(dim=1).sqrt())
        collect("reconstruction_output_rmse_unlearned_retrained", (r_recon - u_recon).pow(2).mean(dim=1).sqrt())
        collect(
            "reconstruction_mse_baseline",
            (b_recon - target_gex).pow(2).mean(dim=1),
        )
        collect(
            "reconstruction_mse_unlearned",
            (u_recon - target_gex).pow(2).mean(dim=1),
        )
        collect("reconstruction_mse_retrained", (r_recon - target_gex).pow(2).mean(dim=1))
        collect("center_distance_baseline", b_center_distance)
        collect("center_distance_unlearned", u_center_distance)
        collect("center_distance_retrained", r_center_distance)
        collect("emb_true_probability_baseline", b_emb_prob)
        collect("emb_true_probability_unlearned", u_emb_prob)
        collect("emb_true_probability_retrained", r_emb_prob)
        collect("exp_true_probability_baseline", b_exp_prob)
        collect("exp_true_probability_unlearned", u_exp_prob)
        collect("exp_true_probability_retrained", r_exp_prob)

    per_sample = samples.copy()
    for name, chunks in collected.items():
        per_sample[name] = np.concatenate(chunks)
    for stem in (
        "reconstruction_mse",
        "center_distance",
        "emb_true_probability",
        "exp_true_probability",
    ):
        per_sample[f"{stem}_delta"] = (
            per_sample[f"{stem}_unlearned"] - per_sample[f"{stem}_baseline"]
        )

    # Positive improvement means unlearning moved the functional output closer
    # to deletion retraining than the original baseline was.
    for stem in ("center_distance", "emb_true_probability", "exp_true_probability"):
        per_sample[f"{stem}_distance_baseline_retrained"] = (
            per_sample[f"{stem}_baseline"] - per_sample[f"{stem}_retrained"]
        ).abs()
        per_sample[f"{stem}_distance_unlearned_retrained"] = (
            per_sample[f"{stem}_unlearned"] - per_sample[f"{stem}_retrained"]
        ).abs()
        per_sample[f"{stem}_improvement_to_retrained"] = (
            per_sample[f"{stem}_distance_baseline_retrained"]
            - per_sample[f"{stem}_distance_unlearned_retrained"]
        )
    for stem in ("latent_rmse", "attention_js", "wgex_rmse", "reconstruction_output_rmse"):
        per_sample[f"{stem}_improvement_to_retrained"] = (
            per_sample[f"{stem}_baseline_retrained"]
            - per_sample[f"{stem}_unlearned_retrained"]
        )

    numeric_metrics = [
        column
        for column in per_sample.columns
        if column
        not in {"row_index", "sample_id", "patient_id", "sample_code", "tissue_label", "assignment"}
    ]
    patient_metadata = (
        per_sample.groupby("patient_id", sort=True)
        .agg(
            assignment=("assignment", "first"),
            tissue_label=("tissue_label", "first"),
            n_samples=("sample_id", "size"),
        )
    )
    patient_metrics = per_sample.groupby("patient_id", sort=True)[numeric_metrics].mean()
    per_patient = patient_metadata.join(patient_metrics).reset_index()

    sample_summary = _group_summary(per_sample, "sample", numeric_metrics)
    patient_summary = _group_summary(per_patient, "patient", numeric_metrics)
    group_summary = pd.concat([sample_summary, patient_summary], ignore_index=True)

    loss_rows = []
    for unit, frame in (("sample", per_sample), ("patient", per_patient)):
        for assignment, group in frame.groupby("assignment"):
            for model_name in ("baseline", "unlearned", "retrained"):
                loss_rows.append({
                    "unit": unit,
                    "assignment": assignment,
                    "model": model_name,
                    "n": len(group),
                    **{
                        name: group[f"loss_{name}_{model_name}"].mean()
                        for name in ("task", "recon", "emb_class", "exp_class", "center")
                    },
                })
    loss_metrics = pd.DataFrame(loss_rows)

    latents = {name: np.concatenate(chunks) for name, chunks in latent_chunks.items()}
    patient_assignments = per_sample.groupby("patient_id", sort=True)["assignment"].first()
    patient_latents = {}
    for name, values in latents.items():
        frame = pd.DataFrame(values)
        frame["patient_id"] = per_sample["patient_id"].to_numpy()
        patient_latents[name] = frame.groupby("patient_id", sort=True).mean().to_numpy()
    similarity_rows = []
    for unit, values, assignments in (
        ("sample", latents, per_sample["assignment"].to_numpy()),
        ("patient", patient_latents, patient_assignments.to_numpy()),
    ):
        for assignment in ("forget", "retain"):
            mask = assignments == assignment
            for left, right in (
                ("baseline", "unlearned"),
                ("baseline", "retrained"),
                ("unlearned", "retrained"),
            ):
                x, y = values[left][mask], values[right][mask]
                similarity_rows.append({
                    "unit": unit,
                    "assignment": assignment,
                    "comparison": f"{left}_vs_{right}",
                    "n": len(x),
                    "linear_cka": _linear_cka(x, y),
                    "frechet_latent_distance": (
                        _frechet_distance(x, y, args.frechet_eps) if len(x) > 1 else float("nan")
                    ),
                })
    representation_similarity = pd.DataFrame(similarity_rows)

    per_sample.to_csv(output_dir / "representation_change_per_sample.csv", index=False)
    per_patient.to_csv(output_dir / "representation_change_per_patient.csv", index=False)
    group_summary.to_csv(output_dir / "representation_change_group_summary.csv", index=False)
    loss_metrics.to_csv(output_dir / "loss_metrics.csv", index=False)
    representation_similarity.to_csv(output_dir / "representation_similarity.csv", index=False)

    report_metrics = [
        "latent_rmse_baseline_unlearned",
        "latent_rmse_baseline_retrained",
        "latent_rmse_unlearned_retrained",
        "latent_rmse_improvement_to_retrained",
        "center_distance_delta",
        "center_distance_improvement_to_retrained",
        "attention_js_baseline_unlearned",
        "attention_js_baseline_retrained",
        "attention_js_unlearned_retrained",
        "attention_js_improvement_to_retrained",
        "reconstruction_mse_delta",
        "reconstruction_output_rmse_improvement_to_retrained",
    ]
    patient_means = per_patient.groupby("assignment")[report_metrics].mean()
    selectivity_ratio = None
    if {"forget", "retain"}.issubset(patient_means.index):
        retain_latent = float(
            patient_means.loc["retain", "latent_rmse_baseline_unlearned"]
        )
        selectivity_ratio = (
            float(patient_means.loc["forget", "latent_rmse_baseline_unlearned"])
            / retain_latent
            if retain_latent != 0
            else None
        )
    summary = {
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "unlearned_checkpoint": str(Path(args.unlearned_checkpoint).resolve()),
        "retrained_checkpoint": str(Path(args.retrained_checkpoint).resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "n_forget_samples": len(forget_indices),
        "n_retain_samples": len(retain_indices),
        "n_forget_patients": int((per_patient["assignment"] == "forget").sum()),
        "n_retain_patients": int((per_patient["assignment"] == "retain").sum()),
        "baseline_center_source": baseline_center_source,
        "unlearned_center_source": unlearned_center_source,
        "retrained_center_source": retrained_center_source,
        "center_max_abs_diff": center_max_abs_diff,
        "center_max_abs_diff_baseline_unlearned": center_max_abs_diff_baseline_unlearned,
        "center_max_abs_diff_baseline_retrained": center_max_abs_diff_baseline_retrained,
        "latent_rmse_forget_to_retain_ratio": selectivity_ratio,
        "loss_weights": {
            "reconstruction_mse": args.recon_weight,
            "classification_cross_entropy": args.class_weight,
            "center": args.center_weight,
        },
        "patient_group_means": patient_means.to_dict(orient="index"),
    }
    with (output_dir / "representation_change_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"[data] forget_samples={len(forget_indices)} retain_samples={len(retain_indices)}")
    print(
        f"[data] forget_patients={summary['n_forget_patients']} "
        f"retain_patients={summary['n_retain_patients']}"
    )
    print(
        f"[center] baseline={baseline_center_source} unlearned={unlearned_center_source} "
        f"retrained={retrained_center_source} "
        f"max_abs_diff={center_max_abs_diff:.8g}"
    )
    print("\n[patient-level means]")
    print(patient_means.to_string())
    print("\n[mean losses]")
    print(loss_metrics.to_string(index=False))
    print("\n[latent CKA / Frechet distance]")
    print(representation_similarity.to_string(index=False))
    print(f"\n[selectivity] forget/retain latent RMSE ratio={selectivity_ratio}")
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
