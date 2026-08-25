"""Create reproducible, patient-level forget/retain manifests for TCGA."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ID_COLUMNS = (
    "sample_id",
    "Sample_ID",
    "sample_submitter_id",
    "submitter_id",
    "barcode",
    "ID",
)


def tcga_patient_id(barcode: object) -> str:
    """Return the participant barcode (TCGA-TSS-PARTICIPANT)."""
    value = str(barcode).strip()
    parts = value.split("-")
    if len(parts) < 3 or parts[0].upper() != "TCGA":
        raise ValueError(f"invalid TCGA barcode: {value!r}")
    return "-".join(parts[:3])


def tcga_sample_code(barcode: object) -> str | None:
    """Return the two-digit TCGA sample-type code when it is present."""
    parts = str(barcode).strip().split("-")
    if len(parts) < 4 or len(parts[3]) < 2:
        return None
    return parts[3][:2]


def _resolve_info_id_column(info: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in info.columns:
            raise KeyError(
                f"--info-id-column {requested!r} is absent; columns={info.columns.tolist()}"
            )
        return requested
    for name in DEFAULT_ID_COLUMNS:
        if name not in info.columns:
            continue
        values = info[name].dropna().astype(str)
        if not values.empty and values.str.startswith("TCGA-").all():
            return name
    return None


def build_sample_table(
    gex: pd.DataFrame,
    info: pd.DataFrame,
    *,
    tissue_column: str = "tissue_label",
    info_id_column: str | None = None,
) -> pd.DataFrame:
    """Validate GEX/info alignment and return one audited row per GEX row.

    THERAPI's original loader pairs the two files positionally.  If the info
    file exposes a sample identifier, this function additionally verifies the
    identifiers after reducing them to participant level.
    """
    if tissue_column not in info.columns:
        raise KeyError(
            f"tissue column {tissue_column!r} is absent; columns={info.columns.tolist()}"
        )
    if len(gex) != len(info):
        raise ValueError(f"GEX/info row mismatch: {len(gex)} != {len(info)}")
    if info[tissue_column].isna().any():
        raise ValueError(
            f"tissue column {tissue_column!r} contains "
            f"{int(info[tissue_column].isna().sum())} missing values"
        )

    sample_ids = pd.Series(gex.index.astype(str), name="sample_id").reset_index(drop=True)
    try:
        patient_ids = sample_ids.map(tcga_patient_id)
    except ValueError as exc:
        raise ValueError(f"could not parse a GEX index as a TCGA barcode: {exc}") from exc

    resolved_id = _resolve_info_id_column(info, info_id_column)
    if resolved_id is not None:
        info_ids = info[resolved_id].astype(str).reset_index(drop=True)
        try:
            info_patients = info_ids.map(tcga_patient_id)
        except ValueError as exc:
            raise ValueError(
                f"could not parse info column {resolved_id!r} as TCGA barcodes: {exc}"
            ) from exc
        mismatch = patient_ids != info_patients
        if mismatch.any():
            examples = pd.DataFrame(
                {
                    "gex_id": sample_ids[mismatch],
                    "info_id": info_ids[mismatch],
                }
            ).head(10)
            raise ValueError(
                f"GEX/info patient order differs in {int(mismatch.sum())} rows; "
                f"first mismatches:\n{examples.to_string(index=False)}"
            )

    table = pd.DataFrame(
        {
            "row_index": np.arange(len(gex), dtype=np.int64),
            "sample_id": sample_ids,
            "patient_id": patient_ids,
            "sample_code": sample_ids.map(tcga_sample_code),
            "tissue_label": info[tissue_column].reset_index(drop=True),
        }
    )

    tissue_counts = table.groupby("patient_id", sort=True)["tissue_label"].nunique()
    conflicts = tissue_counts[tissue_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            f"{len(conflicts)} patients have conflicting tissue labels; "
            f"examples={conflicts.index[:10].tolist()}"
        )
    return table


def _allocate_stratified_counts(group_sizes: pd.Series, total_forget: int) -> dict[object, int]:
    """Allocate an exact forget count proportionally without emptying a stratum."""
    total = int(group_sizes.sum())
    if total_forget <= 0 or total_forget >= total:
        raise ValueError(f"forget count must be in [1, {total - 1}], got {total_forget}")

    ideal = group_sizes.astype(float) * (total_forget / total)
    capacity = group_sizes.map(lambda n: max(int(n) - 1, 0))
    allocated = np.minimum(np.floor(ideal).astype(int), capacity)
    remaining = total_forget - int(allocated.sum())

    # Largest-remainder allocation with a stable string tie-break.
    order = sorted(
        group_sizes.index,
        key=lambda key: (-(ideal.loc[key] - math.floor(ideal.loc[key])), str(key)),
    )
    while remaining > 0:
        changed = False
        for key in order:
            if allocated.loc[key] < capacity.loc[key]:
                allocated.loc[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError(
                "cannot allocate the requested forget patients without deleting an entire "
                "tissue stratum; lower --forget-ratio or disable stratification"
            )
    return {key: int(value) for key, value in allocated.items()}


def make_patient_split(
    samples: pd.DataFrame,
    *,
    forget_ratio: float,
    split_seed: int,
    stratify: bool = True,
    eligible_patients: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return patient and sample manifests with a forget/retain assignment."""
    if not 0 < forget_ratio < 1:
        raise ValueError(f"forget_ratio must be between 0 and 1, got {forget_ratio}")

    patients = (
        samples.groupby("patient_id", sort=True)
        .agg(tissue_label=("tissue_label", "first"), n_samples=("sample_id", "size"))
        .reset_index()
    )
    eligible = set(patients["patient_id"])
    if eligible_patients is not None:
        requested = {str(value) for value in eligible_patients}
        unknown = requested - eligible
        if unknown:
            raise ValueError(f"eligible-patient list contains unknown IDs: {sorted(unknown)[:10]}")
        eligible = requested

    candidate = patients[patients["patient_id"].isin(eligible)].copy()
    total_forget = int(round(len(candidate) * forget_ratio))
    total_forget = max(1, min(total_forget, len(candidate) - 1))
    rng = np.random.default_rng(split_seed)
    chosen: list[str] = []

    if stratify:
        sizes = candidate.groupby("tissue_label", dropna=False)["patient_id"].size()
        counts = _allocate_stratified_counts(sizes, total_forget)
        for tissue, group in candidate.groupby("tissue_label", sort=True, dropna=False):
            ids = np.asarray(sorted(group["patient_id"].tolist()), dtype=object)
            count = counts[tissue]
            if count:
                chosen.extend(rng.choice(ids, size=count, replace=False).tolist())
    else:
        ids = np.asarray(sorted(candidate["patient_id"].tolist()), dtype=object)
        chosen = rng.choice(ids, size=total_forget, replace=False).tolist()

    forget = set(chosen)
    patients["assignment"] = np.where(patients["patient_id"].isin(forget), "forget", "retain")
    samples = samples.copy()
    samples["assignment"] = np.where(samples["patient_id"].isin(forget), "forget", "retain")

    forget_ids = set(patients.loc[patients["assignment"] == "forget", "patient_id"])
    retain_ids = set(patients.loc[patients["assignment"] == "retain", "patient_id"])
    if forget_ids & retain_ids or forget_ids | retain_ids != set(patients["patient_id"]):
        raise AssertionError("patient-level split invariant failed")
    return patients, samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(
    patient_manifest: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    output_dir: str | Path,
    *,
    metadata: dict[str, object],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    patient_manifest.to_csv(output / "patients.csv", index=False)
    sample_manifest.to_csv(output / "samples.csv", index=False)
    for assignment in ("forget", "retain"):
        patient_manifest[patient_manifest["assignment"] == assignment].to_csv(
            output / f"{assignment}_patients.csv", index=False
        )
        sample_manifest[sample_manifest["assignment"] == assignment].to_csv(
            output / f"{assignment}_samples.csv", index=False
        )

    metadata = dict(metadata)
    metadata.update(
        {
            "n_patients": int(len(patient_manifest)),
            "n_forget_patients": int((patient_manifest["assignment"] == "forget").sum()),
            "n_retain_patients": int((patient_manifest["assignment"] == "retain").sum()),
            "n_samples": int(len(sample_manifest)),
            "n_forget_samples": int((sample_manifest["assignment"] == "forget").sum()),
            "n_retain_samples": int((sample_manifest["assignment"] == "retain").sum()),
            "patients_sha256": _sha256(output / "patients.csv"),
            "samples_sha256": _sha256(output / "samples.csv"),
        }
    )
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def load_manifest_indices(
    samples: pd.DataFrame, manifest_dir: str | Path
) -> tuple[list[int], list[int]]:
    """Validate a saved split against current data and return row indices."""
    manifest = pd.read_csv(Path(manifest_dir) / "samples.csv")
    required = {"row_index", "sample_id", "patient_id", "assignment"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"split manifest is missing columns: {sorted(missing)}")
    if len(manifest) != len(samples):
        raise ValueError(f"manifest/data row mismatch: {len(manifest)} != {len(samples)}")

    dtypes = {"row_index": "int64", "sample_id": "str", "patient_id": "str"}
    current = samples[["row_index", "sample_id", "patient_id"]].astype(dtypes)
    recorded = manifest[["row_index", "sample_id", "patient_id"]].astype(dtypes)
    if not current.equals(recorded):
        raise ValueError("split manifest no longer matches the current TCGA GEX row order")
    if not set(manifest["assignment"]).issubset({"forget", "retain"}):
        raise ValueError("manifest assignment must contain only 'forget' and 'retain'")

    patient_assignments = manifest.groupby("patient_id")["assignment"].nunique()
    leaked = patient_assignments[patient_assignments > 1]
    if not leaked.empty:
        raise ValueError(
            f"patient leakage across forget/retain assignments: {leaked.index[:10].tolist()}"
        )

    forget = manifest.loc[manifest["assignment"] == "forget", "row_index"].astype(int).tolist()
    retain = manifest.loc[manifest["assignment"] == "retain", "row_index"].astype(int).tolist()
    if not forget or not retain or set(forget) & set(retain):
        raise ValueError("forget/retain indices must be non-empty and disjoint")
    return forget, retain
