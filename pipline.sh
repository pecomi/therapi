#!/usr/bin/env bash
set -euo pipefail

# THERAPI end-to-end pipeline
#   1. aligner    : train_aligner.py            (GDSC cell lines <-> TCGA tumors)
#   2. embed      : embedding/csg2a_embed.py    (CSG2A perturbation vectors, GDSC + TCGA)
#   3. predictor  : train_predictor.py          (10-fold CV on GDSC)
#   4. test       : test_TCGA.py                (transfer to TCGA patients)
#
# Every execution gets its own run/<RUN_NAME> folder holding the checkpoints,
# the embeddings and the results, so runs never overwrite each other.
#
# Usage:
#   ./pipline.sh                                  # full pipeline
#   RUN_NAME=dose1uM DOSE=0.01 ./pipline.sh       # override any variable below
#   RESUME=1 RUN_NAME=test3 STAGES="predictor test" ./pipline.sh

RUN_NAME=${RUN_NAME:-test4}
DEVICE=${DEVICE:-cuda:3}
SEED=${SEED:-0}

SOURCE=${SOURCE:-GDSC}
TARGET=${TARGET:-TCGA}

CSG2A_CKPT=${CSG2A_CKPT:-src/embedding/CSG2A_LINCSpretrained_Landmark.pt}
STRING_EDGES=${STRING_EDGES:-src/embedding/CSG2A/data/STRING_edges.csv}
DOSE=${DOSE:-0.1}          # 0.1 == 10 uM  (dose in uM / 100, as in CSG2A pretraining)
TIME=${TIME:-1.0}          # 1.0 == 72 h   (time in h / 72)
EMB_BATCH=${EMB_BATCH:-256}
EMB_WORKERS=${EMB_WORKERS:-0}   # >0 needs a large /dev/shm (64M here kills the workers)

PREDICTOR_NAME=${PREDICTOR_NAME:-THERAPI_predictor}
STAGES=${STAGES:-"aligner embed predictor test"}
RESUME=${RESUME:-0}        # 1 = reuse run/<RUN_NAME> instead of creating a new folder
PYTHON=${PYTHON:-python}

# --------------------------------------------------------------------- layout
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=$ROOT/src
DATA=$ROOT/data
CSG2A_CKPT=$ROOT/$CSG2A_CKPT
STRING_EDGES=$ROOT/$STRING_EDGES

RUN_DIR=$ROOT/run/$RUN_NAME
if [ -d "$RUN_DIR" ] && [ "$RESUME" != 1 ]; then
    n=2
    while [ -d "${RUN_DIR}_${n}" ]; do n=$((n + 1)); done
    echo "[run] run/$RUN_NAME already exists -> using run/${RUN_NAME}_${n}"
    RUN_DIR=${RUN_DIR}_${n}
    RUN_NAME=${RUN_NAME}_${n}
fi
RUN_DATA=$RUN_DIR/data
mkdir -p "$RUN_DIR"/{ckpts,log,output} "$RUN_DATA"

ALIGNER_CKPT=$RUN_DIR/ckpts/THERAPI_aligner_${SOURCE}_${TARGET}.pt

log() { echo -e "\n[$(date '+%F %T')] === $* ==="; }
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

for f in "$CSG2A_CKPT" "$STRING_EDGES"; do
    [ -f "$f" ] || { echo "missing required file: $f" >&2; exit 1; }
done
"$PYTHON" -c 'import torch, pandas, rdkit, sklearn' 2>/dev/null \
    || { echo "$PYTHON cannot import torch/pandas/rdkit/sklearn -- activate the environment first (conda activate theraphi)" >&2; exit 1; }

{
    echo "run_name      $RUN_NAME"
    echo "started       $(date '+%F %T')"
    echo "git_commit    $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "device        $DEVICE"
    echo "seed          $SEED"
    echo "source/target $SOURCE / $TARGET"
    echo "csg2a_ckpt    $CSG2A_CKPT"
    echo "dose/time     $DOSE / $TIME"
    echo "emb_batch     $EMB_BATCH"
    echo "stages        $STAGES"
}

# train_predictor.py and test_TCGA.py read the perturbation matrices from
# <data_dir>/<dataset>/, so each run gets its own view of data/: the shared
# inputs are symlinked and this run's embeddings are written next to them.
( cd "$DATA" && find . -type f ) | sed 's|^\./||' | while read -r rel; do
    dst=$RUN_DATA/$rel
    [ -e "$dst" ] && continue
    mkdir -p "$(dirname -- "$dst")"
    ln -s "$DATA/$rel" "$dst"
done
# train_predictor.py expects the CV split one level above where it is shipped
[ -e "$RUN_DATA/GDSC_split" ] || ln -s "$DATA/GDSC/GDSC_split" "$RUN_DATA/GDSC_split"

# cwd decides where the scripts put ckpts/ and log/
cd "$RUN_DIR"

# -------------------------------------------------------------- 1. aligner
if has_stage aligner; then
    log "1/4 aligner: $SOURCE -> $TARGET"
    "$PYTHON" "$SRC/train_aligner.py" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --data_dir "$RUN_DATA/" \
        --source "$SOURCE" \
        --target "$TARGET"
fi

# ------------------------------------------------------- 2. CSG2A embeddings
embed() {
    local dataset=$1
    shift
    local out_pert=$RUN_DATA/$dataset/${dataset}_perturbation_float16.npy
    local out_comp=$RUN_DATA/$dataset/${dataset}_perturbation_compound_float16.npy
    # never write through a symlink into the shared data/ folder
    for out in "$out_pert" "$out_comp"; do
        [ -L "$out" ] && rm -- "$out"
    done
    "$PYTHON" "$SRC/embedding/csg2a_embed.py" \
        --dataset "$dataset" \
        --data_dir "$DATA" \
        --csg2a_ckpt "$CSG2A_CKPT" \
        --string_edges "$STRING_EDGES" \
        --feature_cache "$DATA/$dataset/${dataset}_molfeat_dn_ohfc.p" \
        --out_pert "$out_pert" \
        --out_comp "$out_comp" \
        --dose "$DOSE" \
        --time "$TIME" \
        --batch_size "$EMB_BATCH" \
        --num_workers "$EMB_WORKERS" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --overwrite "$@"
}

if has_stage embed; then
    # the target domain is aligned onto the cell-line space first, the source
    # domain is already in it
    log "2/4 embeddings: $TARGET (aligned)"
    [ -f "$ALIGNER_CKPT" ] || { echo "missing aligner checkpoint: $ALIGNER_CKPT" >&2; exit 1; }
    embed "$TARGET" --aligner_ckpt "$ALIGNER_CKPT"

    log "2/4 embeddings: $SOURCE"
    embed "$SOURCE"
fi

# ------------------------------------------------------------ 3. predictor
if has_stage predictor; then
    log "3/4 predictor: 10-fold CV on $SOURCE"
    "$PYTHON" "$SRC/train_predictor.py" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --data_dir "$RUN_DATA/"
fi

# ----------------------------------------------------------------- 4. test
if has_stage test; then
    log "4/4 test: $TARGET"
    "$PYTHON" "$SRC/test_TCGA.py" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --data_dir "$RUN_DATA/" \
        --model_name "$PREDICTOR_NAME" \
        --output_dir "$RUN_DIR/output/"
fi

log "done -> $RUN_DIR"
ls -la "$RUN_DIR/output"
