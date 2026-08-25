"""CLI for creating reproducible TCGA patient-level forget/retain manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Support direct execution: python src/unlearning/make_forget_split.py ...
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unlearning.split import build_sample_table, make_patient_split, write_split


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    gex_path = data_dir / args.target / f"{args.target}_unlabeled_gex.csv"
    info_path = data_dir / args.target / f"{args.target}_unlabeled_info.csv"
    gex = pd.read_csv(gex_path, index_col=0)
    info = pd.read_csv(info_path)

    samples = build_sample_table(
        gex,
        info,
        tissue_column=args.tissue_column,
        info_id_column=args.info_id_column,
    )
    eligible = None
    if args.eligible_patients_file:
        eligible_df = pd.read_csv(args.eligible_patients_file)
        if args.eligible_patient_column not in eligible_df.columns:
            raise KeyError(
                f"eligible patient column {args.eligible_patient_column!r} is absent; "
                f"columns={eligible_df.columns.tolist()}"
            )
        eligible = eligible_df[args.eligible_patient_column].astype(str).tolist()

    patients, samples = make_patient_split(
        samples,
        forget_ratio=args.forget_ratio,
        split_seed=args.split_seed,
        stratify=not args.no_stratify,
        eligible_patients=eligible,
    )
    write_split(
        patients,
        samples,
        args.output_dir,
        metadata={
            "scenario": "random_patient",
            "forget_ratio": args.forget_ratio,
            "split_seed": args.split_seed,
            "stratified": not args.no_stratify,
            "target": args.target,
            "gex_path": str(gex_path.resolve()),
            "info_path": str(info_path.resolve()),
        },
    )
    print(f"[done] split manifest -> {Path(args.output_dir).resolve()}")
    print(patients.groupby(["tissue_label", "assignment"]).size().unstack(fill_value=0))
    print(patients["assignment"].value_counts())
    print(samples["assignment"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument("--target", default="TCGA")
    parser.add_argument("--tissue-column", default="tissue_label")
    parser.add_argument("--info-id-column", default=None)
    parser.add_argument("--forget-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--eligible-patients-file", default=None)
    parser.add_argument("--eligible-patient-column", default="patient_id")
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
