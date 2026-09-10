#!/usr/bin/env bash
set -euo pipefail

# Reproducible THERAPI unlearning pipeline.
#
# The predictor is trained only on GDSC, so a new unlearning checkpoint requires
# a new *target* (TCGA) CSG2A embedding, not GDSC embedding or predictor training.
#
# Examples:
#   # Complete new-seed experiment: baseline -> NegGrad+ -> retraining -> evaluation.
#   RUN_NAME=seed1 ORIGINAL_TRAIN_SEED=1 UNLEARN_SEED=1 SPLIT_SEED=1 \
#     ./unlearning_pipeline.sh
#
#   # Also regenerate TCGA embeddings and test existing predictor checkpoints.
#   PREDICTOR_CKPT_DIR=run/predictor_seed0/ckpts \
#   STAGES="split baseline neggrad_plus retrain evaluate embed predictor_test" \
#     ./unlearning_pipeline.sh
#
# To reuse a fixed existing baseline rather than train it, omit ``baseline``:
#   BASELINE_CHECKPOINT=run/baseline_seed0/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
#   STAGES="neggrad_plus retrain evaluate" ./unlearning_pipeline.sh

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=$ROOT/src
DATA=$ROOT/data
PYTHON=${PYTHON:-python}

SOURCE=${SOURCE:-GDSC}
TARGET=${TARGET:-TCGA}
DEVICE=${DEVICE:-cuda:0}
ORIGINAL_TRAIN_SEED=${ORIGINAL_TRAIN_SEED:-0}
UNLEARN_SEED=${UNLEARN_SEED:-0}
SPLIT_SEED=${SPLIT_SEED:-0}
FORGET_RATIO=${FORGET_RATIO:-0.05}

RUN_NAME=${RUN_NAME:-unlearning_neggradplus_seed${UNLEARN_SEED}}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/random_patient_5pct_seed${SPLIT_SEED}}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-}
RETRAIN_CHECKPOINT=${RETRAIN_CHECKPOINT:-}
PREDICTOR_CKPT_DIR=${PREDICTOR_CKPT_DIR:-}
PREDICTOR_NAME=${PREDICTOR_NAME:-THERAPI_predictor}

BASELINE_EPOCHS=${BASELINE_EPOCHS:-199}
BASELINE_BATCH_SIZE=${BASELINE_BATCH_SIZE:-128}
BASELINE_LR=${BASELINE_LR:-1e-3}
NEGGRAD_EPOCHS=${NEGGRAD_EPOCHS:-30}
NEGGRAD_BATCH_SIZE=${NEGGRAD_BATCH_SIZE:-64}
NEGGRAD_LR=${NEGGRAD_LR:-1e-5}
NEGGRAD_PLUS_EPOCHS=${NEGGRAD_PLUS_EPOCHS:-30}
NEGGRAD_PLUS_BATCH_SIZE=${NEGGRAD_PLUS_BATCH_SIZE:-128}
NEGGRAD_PLUS_LR=${NEGGRAD_PLUS_LR:-1e-3}
BETA=${BETA:-0.95}
RETRAIN_EPOCHS=${RETRAIN_EPOCHS:-199}
RETRAIN_BATCH_SIZE=${RETRAIN_BATCH_SIZE:-128}
RETRAIN_LR=${RETRAIN_LR:-1e-3}
LATENT_DIM=${LATENT_DIM:-128}
RECON_WEIGHT=${RECON_WEIGHT:-0.2}
CLASS_WEIGHT=${CLASS_WEIGHT:-0.4}
CENTER_WEIGHT=${CENTER_WEIGHT:-0.8}

CSG2A_CKPT=${CSG2A_CKPT:-$ROOT/src/embedding/CSG2A_LINCSpretrained_Landmark.pt}
STRING_EDGES=${STRING_EDGES:-$ROOT/src/embedding/CSG2A/data/STRING_edges.csv}
DOSE=${DOSE:-0.1}
TIME=${TIME:-1.0}
EMB_BATCH_SIZE=${EMB_BATCH_SIZE:-256}
EMB_WORKERS=${EMB_WORKERS:-0}

# ``embed`` and ``predictor_test`` act on each listed aligner.  Available
# values are baseline, neggrad, neggrad_plus, and retrain.
DEPLOY_METHODS=${DEPLOY_METHODS:-neggrad_plus}
EVAL_UNLEARN_METHOD=${EVAL_UNLEARN_METHOD:-neggrad_plus}
STAGES=${STAGES:-"split baseline neggrad_plus retrain evaluate"}
RESUME=${RESUME:-0}

to_absolute() {
    local path=$1
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$ROOT/$path"
    fi
}

if [ -n "$BASELINE_CHECKPOINT" ]; then
    BASELINE_CHECKPOINT=$(to_absolute "$BASELINE_CHECKPOINT")
fi
if [ -n "$RETRAIN_CHECKPOINT" ]; then
    RETRAIN_CHECKPOINT=$(to_absolute "$RETRAIN_CHECKPOINT")
fi
SPLIT_DIR=$(to_absolute "$SPLIT_DIR")
CSG2A_CKPT=$(to_absolute "$CSG2A_CKPT")
STRING_EDGES=$(to_absolute "$STRING_EDGES")
if [ -n "$PREDICTOR_CKPT_DIR" ]; then
    PREDICTOR_CKPT_DIR=$(to_absolute "$PREDICTOR_CKPT_DIR")
fi

RUN_DIR=$ROOT/run/$RUN_NAME
if [ -d "$RUN_DIR" ] && [ "$RESUME" != 1 ]; then
    suffix=2
    while [ -d "${RUN_DIR}_${suffix}" ]; do
        suffix=$((suffix + 1))
    done
    RUN_DIR=${RUN_DIR}_${suffix}
    RUN_NAME=${RUN_NAME}_${suffix}
fi
mkdir -p "$RUN_DIR"
PIPELINE_LOG=$RUN_DIR/pipeline.log
: > "$PIPELINE_LOG"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

log() {
    printf '\n[%s] === %s ===\n' "$(date '+%F %T')" "$*"
}

has_stage() {
    case " $STAGES " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

if has_stage baseline; then
    BASELINE_CHECKPOINT=$RUN_DIR/baseline/ckpts/THERAPI_aligner_${SOURCE}_${TARGET}.pt
elif [ -z "$BASELINE_CHECKPOINT" ]; then
    echo "BASELINE_CHECKPOINT is required when the baseline stage is omitted" >&2
    exit 1
fi
if has_stage retrain; then
    RETRAIN_CHECKPOINT=$RUN_DIR/retrain/ckpts/THERAPI_aligner_${SOURCE}_${TARGET}.pt
fi

require_file() {
    [ -f "$1" ] || { echo "missing required file: $1" >&2; exit 1; }
}

require_split() {
    require_file "$SPLIT_DIR/samples.csv"
    require_file "$SPLIT_DIR/patients.csv"
}

checkpoint_for() {
    case "$1" in
        baseline) printf '%s\n' "$BASELINE_CHECKPOINT" ;;
        neggrad) printf '%s\n' "$RUN_DIR/neggrad/ckpts/THERAPI_aligner_${SOURCE}_${TARGET}.pt" ;;
        neggrad_plus) printf '%s\n' "$RUN_DIR/neggrad_plus/ckpts/THERAPI_aligner_${SOURCE}_${TARGET}.pt" ;;
        retrain) printf '%s\n' "$RETRAIN_CHECKPOINT" ;;
        *) echo "unknown deployment method: $1" >&2; exit 1 ;;
    esac
}

create_data_view() {
    local method_dir=$1
    local view=$method_dir/data
    local source_path rel destination
    mkdir -p "$view"
    while IFS= read -r -d '' source_path; do
        rel=${source_path#"$DATA"/}
        destination=$view/$rel
        if [ -e "$destination" ] || [ -L "$destination" ]; then
            continue
        fi
        mkdir -p "$(dirname -- "$destination")"
        ln -s "$source_path" "$destination"
    done < <(find "$DATA" -type f -print0)
    if [ ! -e "$view/GDSC_split" ] && [ ! -L "$view/GDSC_split" ]; then
        ln -s "$DATA/GDSC/GDSC_split" "$view/GDSC_split"
    fi
}

embed_target() {
    local method=$1
    local method_dir=$RUN_DIR/$method
    local checkpoint out_pert out_comp
    checkpoint=$(checkpoint_for "$method")
    require_file "$checkpoint"
    create_data_view "$method_dir"
    out_pert=$method_dir/data/$TARGET/${TARGET}_perturbation_float16.npy
    out_comp=$method_dir/data/$TARGET/${TARGET}_perturbation_compound_float16.npy
    # The view initially contains symlinks to shared data.  Remove only those
    # links so this method writes private target embeddings.
    for artifact in "$out_pert" "$out_comp"; do
        if [ -L "$artifact" ]; then
            rm "$artifact"
        fi
    done
    log "embed: $method target embedding"
    "$PYTHON" "$SRC/embedding/csg2a_embed.py" \
        --dataset "$TARGET" \
        --data_dir "$DATA" \
        --aligner_ckpt "$checkpoint" \
        --csg2a_ckpt "$CSG2A_CKPT" \
        --string_edges "$STRING_EDGES" \
        --feature_cache "$DATA/$TARGET/${TARGET}_molfeat_dn_ohfc.p" \
        --out_pert "$out_pert" \
        --out_comp "$out_comp" \
        --dose "$DOSE" \
        --time "$TIME" \
        --batch_size "$EMB_BATCH_SIZE" \
        --num_workers "$EMB_WORKERS" \
        --device "$DEVICE" \
        --seed "$UNLEARN_SEED" \
        --overwrite
}

link_predictor_checkpoints() {
    local method_dir=$1
    local fold source_checkpoint destination
    [ -n "$PREDICTOR_CKPT_DIR" ] || {
        echo "PREDICTOR_CKPT_DIR is required for predictor_test" >&2
        exit 1
    }
    for fold in $(seq 0 9); do
        source_checkpoint=$PREDICTOR_CKPT_DIR/${PREDICTOR_NAME}_CV${fold}.pt
        destination=$method_dir/ckpts/${PREDICTOR_NAME}_CV${fold}.pt
        require_file "$source_checkpoint"
        if [ -e "$destination" ] && [ ! -L "$destination" ]; then
            echo "refusing to replace non-symlink predictor checkpoint: $destination" >&2
            exit 1
        fi
        ln -sfn "$source_checkpoint" "$destination"
    done
}

test_predictor() {
    local method=$1
    local method_dir=$RUN_DIR/$method
    create_data_view "$method_dir"
    require_file "$method_dir/data/$TARGET/${TARGET}_perturbation_float16.npy"
    require_file "$method_dir/data/$TARGET/${TARGET}_perturbation_compound_float16.npy"
    link_predictor_checkpoints "$method_dir"
    mkdir -p "$method_dir/output"
    log "predictor_test: $method"
    (
        cd "$method_dir"
        "$PYTHON" "$SRC/test_TCGA.py" \
            --seed "$UNLEARN_SEED" \
            --device "$DEVICE" \
            --data_dir "$method_dir/data/" \
            --model_name "$PREDICTOR_NAME" \
            --output_dir "$method_dir/output/"
    )
}

[ -d "$DATA" ] || { echo "missing data directory: $DATA" >&2; exit 1; }
"$PYTHON" -c 'import torch, pandas' 2>/dev/null || {
    echo "$PYTHON cannot import torch and pandas; activate the experiment environment first" >&2
    exit 1
}

log "unlearning pipeline configuration"
printf '%s\n' \
    "run_dir=$RUN_DIR" \
    "stages=$STAGES" \
    "baseline_checkpoint=$BASELINE_CHECKPOINT" \
    "retrain_checkpoint=${RETRAIN_CHECKPOINT:-generated_by_retrain_stage}" \
    "split_dir=$SPLIT_DIR" \
    "device=$DEVICE" \
    "original_train_seed=$ORIGINAL_TRAIN_SEED" \
    "unlearn_seed=$UNLEARN_SEED" \
    "latent_dim=$LATENT_DIM" \
    "loss_weights=recon:$RECON_WEIGHT,class:$CLASS_WEIGHT,center:$CENTER_WEIGHT" \
    "git_commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)"

if has_stage split; then
    if [ -f "$SPLIT_DIR/samples.csv" ] && [ -f "$SPLIT_DIR/patients.csv" ]; then
        log "split: reusing $SPLIT_DIR"
    else
        log "split: creating patient-level manifest"
        "$PYTHON" "$SRC/unlearning/make_forget_split.py" \
            --data-dir "$DATA" \
            --target "$TARGET" \
            --forget-ratio "$FORGET_RATIO" \
            --split-seed "$SPLIT_SEED" \
            --output-dir "$SPLIT_DIR"
    fi
fi
require_split

if has_stage baseline; then
    log "baseline training"
    mkdir -p "$RUN_DIR/baseline"
    (
        cd "$RUN_DIR/baseline"
        "$PYTHON" "$SRC/train_aligner.py" \
            --data-dir "$DATA" \
            --source "$SOURCE" --target "$TARGET" \
            --split-dir "$SPLIT_DIR" \
            --device "$DEVICE" \
            --seed "$ORIGINAL_TRAIN_SEED" \
            --latent-dim "$LATENT_DIM" \
            --epochs "$BASELINE_EPOCHS" \
            --batch-size "$BASELINE_BATCH_SIZE" \
            --lr "$BASELINE_LR" \
            --recon-weight "$RECON_WEIGHT" \
            --class-weight "$CLASS_WEIGHT" \
            --center-weight "$CENTER_WEIGHT"
    )
fi
require_file "$BASELINE_CHECKPOINT"

if has_stage neggrad; then
    log "NegGrad"
    "$PYTHON" "$SRC/unlearning/gradient_ascent.py" \
        --data-dir "$DATA" \
        --source "$SOURCE" --target "$TARGET" \
        --checkpoint "$BASELINE_CHECKPOINT" \
        --split-dir "$SPLIT_DIR" \
        --output-dir "$RUN_DIR/neggrad" \
        --device "$DEVICE" \
        --original-train-seed "$ORIGINAL_TRAIN_SEED" \
        --unlearn-seed "$UNLEARN_SEED" \
        --latent-dim "$LATENT_DIM" \
        --epochs "$NEGGRAD_EPOCHS" \
        --batch-size "$NEGGRAD_BATCH_SIZE" \
        --lr "$NEGGRAD_LR" \
        --recon-weight "$RECON_WEIGHT" \
        --class-weight "$CLASS_WEIGHT" \
        --center-weight "$CENTER_WEIGHT"
fi

if has_stage neggrad_plus; then
    log "NegGrad+"
    "$PYTHON" "$SRC/unlearning/retain_finetune.py" \
        --data-dir "$DATA" \
        --source "$SOURCE" --target "$TARGET" \
        --checkpoint "$BASELINE_CHECKPOINT" \
        --split-dir "$SPLIT_DIR" \
        --output-dir "$RUN_DIR/neggrad_plus" \
        --device "$DEVICE" \
        --original-train-seed "$ORIGINAL_TRAIN_SEED" \
        --unlearn-seed "$UNLEARN_SEED" \
        --latent-dim "$LATENT_DIM" \
        --epochs "$NEGGRAD_PLUS_EPOCHS" \
        --batch-size "$NEGGRAD_PLUS_BATCH_SIZE" \
        --lr "$NEGGRAD_PLUS_LR" \
        --recon-weight "$RECON_WEIGHT" \
        --class-weight "$CLASS_WEIGHT" \
        --center-weight "$CENTER_WEIGHT" \
        --beta "$BETA"
fi

if has_stage retrain; then
    log "deletion retraining"
    "$PYTHON" "$SRC/unlearning/retrain.py" \
        --data-dir "$DATA" \
        --source "$SOURCE" --target "$TARGET" \
        --split-dir "$SPLIT_DIR" \
        --output-dir "$RUN_DIR/retrain" \
        --device "$DEVICE" \
        --seed "$ORIGINAL_TRAIN_SEED" \
        --latent-dim "$LATENT_DIM" \
        --epochs "$RETRAIN_EPOCHS" \
        --batch-size "$RETRAIN_BATCH_SIZE" \
        --lr "$RETRAIN_LR" \
        --recon-weight "$RECON_WEIGHT" \
        --class-weight "$CLASS_WEIGHT" \
        --center-weight "$CENTER_WEIGHT"
fi

if has_stage evaluate; then
    unlearned_checkpoint=$(checkpoint_for "$EVAL_UNLEARN_METHOD")
    retrained_checkpoint=$(checkpoint_for retrain)
    require_file "$unlearned_checkpoint"
    require_file "$retrained_checkpoint"
    log "representation evaluation: $EVAL_UNLEARN_METHOD"
    "$PYTHON" "$SRC/unlearning/evaluate_representations.py" \
        --data-dir "$DATA" \
        --baseline-checkpoint "$BASELINE_CHECKPOINT" \
        --unlearned-checkpoint "$unlearned_checkpoint" \
        --retrained-checkpoint "$retrained_checkpoint" \
        --split-dir "$SPLIT_DIR" \
        --output-dir "$RUN_DIR/evaluation" \
        --device "$DEVICE" \
        --latent-dim "$LATENT_DIM" \
        --recon-weight "$RECON_WEIGHT" \
        --class-weight "$CLASS_WEIGHT" \
        --center-weight "$CENTER_WEIGHT"
fi

if has_stage embed; then
    require_file "$CSG2A_CKPT"
    require_file "$STRING_EDGES"
    "$PYTHON" -c 'import rdkit, sklearn' 2>/dev/null || {
        echo "$PYTHON cannot import rdkit and sklearn required by CSG2A embedding" >&2
        exit 1
    }
    for method in $DEPLOY_METHODS; do
        embed_target "$method"
    done
fi

if has_stage predictor_test; then
    for method in $DEPLOY_METHODS; do
        test_predictor "$method"
    done
fi

log "done -> $RUN_DIR"
