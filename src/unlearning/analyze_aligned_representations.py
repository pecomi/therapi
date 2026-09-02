"""Post-hoc comparison of weighted target representations from two checkpoints.

This script performs inference only.  It extracts the second output of
``TARGET_weightencoder`` (the attention-weighted sum of source latents) for the
same TCGA rows in original and unlearned models, then applies the representation
metrics already used by ``evaluate_representations.py``.
"""

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

from model import AlignerDataset, SOURCE_AE, TARGET_weightencoder
from unlearning.evaluate_representations import (
    _frechet_distance,
    _linear_cka,
    _patient_means,
)
from unlearning.split import build_sample_table, load_manifest_indices


def _load_alignment_modules(
    checkpoint_path: str | Path,
    *,
    n_genes: int,
    n_source: int,
    n_tissue: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[SOURCE_AE, TARGET_weightencoder]:
    """Load only the modules needed to reconstruct the weighted representation."""
    source_ae = SOURCE_AE(n_genes, n_tissue, latent_dim).to(device)
    target_encoder = TARGET_weightencoder(n_genes, latent_dim, n_source).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_ae.load_state_dict(checkpoint["source_AE"])
    target_encoder.load_state_dict(checkpoint["target_weightencoder"])
    source_ae.eval()
    target_encoder.eval()
    return source_ae, target_encoder


@torch.no_grad()
def extract_aligned_representations(
    loader: DataLoader,
    source_gex: torch.Tensor,
    modules: dict[str, tuple[SOURCE_AE, TARGET_weightencoder]],
) -> dict[str, np.ndarray]:
    """Return each model's z'_t for exactly the same batches and loader order."""
    source_latents = {
        name: source_ae.encoder(source_gex)
        for name, (source_ae, _) in modules.items()
    }
    chunks = {name: [] for name in modules}
    for target_gex, _, _ in loader:
        target_gex = target_gex.to(source_gex.device)
        for name, (_, target_encoder) in modules.items():
            _, aligned_latent, _, _ = target_encoder(
                target_gex, source_latents[name], source_gex
            )
            chunks[name].append(aligned_latent.cpu().numpy())
    return {
        name: np.concatenate(values, axis=0) for name, values in chunks.items()
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> None:
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

    source_dataset = AlignerDataset(
        source_df, args.source, source_info["tissue_label"]
    )
    target_dataset = AlignerDataset(
        target_df, args.target, target_info[args.tissue_column]
    )
    loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    common = dict(
        n_genes=source_dataset.n_genes,
        n_source=len(source_dataset),
        n_tissue=int(source_info["tissue_label"].nunique()),
        latent_dim=args.latent_dim,
        device=device,
    )
    modules = {
        "original": _load_alignment_modules(args.original_checkpoint, **common),
        "unlearned": _load_alignment_modules(args.unlearned_checkpoint, **common),
    }
    source_gex = source_dataset.data.to(device)
    aligned = extract_aligned_representations(loader, source_gex, modules)
    if any(len(values) != len(samples) for values in aligned.values()):
        raise RuntimeError("extracted representation count does not match sample metadata")

    assignments = samples["assignment"].to_numpy()
    patient_ids = samples["patient_id"].to_numpy()
    patient_metadata = samples.groupby("patient_id", sort=True).agg(
        assignment=("assignment", "first")
    )
    patient_aligned = {
        name: _patient_means(values, patient_ids) for name, values in aligned.items()
    }

    rows = []
    for unit, values, unit_assignments in (
        ("sample", aligned, assignments),
        ("patient", patient_aligned, patient_metadata["assignment"].to_numpy()),
    ):
        for assignment in ("forget", "retain"):
            mask = unit_assignments == assignment
            original = values["original"][mask]
            unlearned = values["unlearned"][mask]
            rows.append(
                {
                    "representation": "target_latent_weighted_source",
                    "unit": unit,
                    "assignment": assignment,
                    "comparison": "original_vs_unlearned",
                    "n": len(original),
                    "linear_cka": _linear_cka(original, unlearned),
                    "frechet_latent_distance": (
                        _frechet_distance(original, unlearned, args.frechet_eps)
                        if len(original) > 1
                        else float("nan")
                    ),
                }
            )
    similarity = pd.DataFrame(rows)

    # A single archive keeps both matrices and their audited row identities together.
    np.savez_compressed(
        output_dir / "aligned_representations.npz",
        row_index=samples["row_index"].to_numpy(dtype=np.int64),
        sample_id=samples["sample_id"].to_numpy(dtype=str),
        patient_id=samples["patient_id"].to_numpy(dtype=str),
        assignment=samples["assignment"].to_numpy(dtype=str),
        original=aligned["original"].astype(np.float32, copy=False),
        unlearned=aligned["unlearned"].astype(np.float32, copy=False),
    )
    similarity.to_csv(
        output_dir / "aligned_representation_similarity.csv", index=False
    )
    summary = {
        "analysis": "post_hoc_inference_only",
        "representation": {
            "name": "target_latent_weighted_source",
            "definition": (
                "softmax(Q(target_gex) @ K(source_latent).T / sqrt(latent_dim)) "
                "@ source_latent"
            ),
            "model_forward_output": "TARGET_weightencoder.forward()[1]",
            "latent_dim": args.latent_dim,
        },
        "original_checkpoint": str(Path(args.original_checkpoint).resolve()),
        "unlearned_checkpoint": str(Path(args.unlearned_checkpoint).resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "n_forget_samples": len(forget_indices),
        "n_retain_samples": len(retain_indices),
        "n_forget_patients": int(
            (patient_metadata["assignment"] == "forget").sum()
        ),
        "n_retain_patients": int(
            (patient_metadata["assignment"] == "retain").sum()
        ),
        "metrics_reused_from": "unlearning.evaluate_representations",
        "reported_metrics": ["linear_cka", "frechet_latent_distance"],
        "outputs": [
            "aligned_representations.npz",
            "aligned_representation_similarity.csv",
        ],
    }
    with (output_dir / "aligned_representation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"[data] forget_samples={len(forget_indices)} "
        f"retain_samples={len(retain_indices)}"
    )
    print("\n[aligned representation similarity]")
    print(similarity.to_string(index=False))
    print(f"[done] results -> {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument("--source", default="GDSC")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument("--original-checkpoint", required=True)
    parser.add_argument("--unlearned-checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tissue-column", default="tissue_label")
    parser.add_argument("--info-id-column", default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frechet-eps", type=float, default=1e-6)
    analyze(parser.parse_args())
