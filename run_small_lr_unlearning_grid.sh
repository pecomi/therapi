#!/usr/bin/env bash
# Smaller-step gradient-ascent comparison with representation evaluation only.
# This script intentionally does NOT call pipline.sh, CSG2A, embed, predictor, or test.

set -euo pipefail

DEVICE="cuda:0"
BASELINE_RUN="baseline_seed0_3"
RETRAIN_RUN="retrain_5pct_seed0"
SPLIT_NAME="random_patient_5pct_seed0"
EXPERIMENT_NAME="ga_small_lr_center_grid_seed0"

BASELINE_CKPT="run/${BASELINE_RUN}/ckpts/THERAPI_aligner_GDSC_TCGA.pt"
RETRAINED_CKPT="run/${RETRAIN_RUN}/ckpts/THERAPI_aligner_GDSC_TCGA.pt"
SPLIT_DIR="splits/${SPLIT_NAME}"
OUTPUT_ROOT="run/${EXPERIMENT_NAME}"

for path in "$BASELINE_CKPT" "$RETRAINED_CKPT" "$SPLIT_DIR/samples.csv"; do
    [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done

run_grid() {
    local mode="$1"
    local lr="$2"
    shift 2
    local lr_tag="${lr//./p}"

    python src/unlearning/run_experiment_grid.py \
        --data_dir data \
        --baseline-checkpoint "$BASELINE_CKPT" \
        --retrained-checkpoint "$RETRAINED_CKPT" \
        --split-dir "$SPLIT_DIR" \
        --output-root "${OUTPUT_ROOT}/${mode}_lr_${lr_tag}" \
        --device "$DEVICE" \
        --original-train-seed 0 \
        --unlearn-seed 0 \
        --step-modes "$mode" \
        --batch-size 128 \
        --epochs "$@" \
        --recon-weights 0.2 \
        --class-weights 0.4 \
        --center-weights 0 0.8 \
        --eval-recon-weight 0.2 \
        --eval-class-weight 0.4 \
        --eval-center-weight 0.8 \
        --full-lr "$lr" \
        --mini-lr "$lr"
}

# mini: 4 Adam updates per epoch for 401 forget samples at batch size 128.
for lr in 5e-7 1e-6; do
    run_grid mini "$lr" 10 12 15 20
done

# full: one Adam update per epoch.
for lr in 2.5e-6 5e-6; do
    run_grid full "$lr" 10 15 20
done

echo "done: representation-only results are under ${OUTPUT_ROOT}"
echo "No CSG2A/GDSC embedding or downstream predictor stage was run."
