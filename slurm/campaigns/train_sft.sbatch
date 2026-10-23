#!/bin/bash
# Generative SFT baseline (b), same mix and budget as PathJev (pathjev.train_sft). Requeue-safe.
#   sbatch --export=ALL,MIX=pathjev-p,RUN_NAME=sft-p-s1,SEED=1[,EPOCHS=3,MAX_STEPS=-1,SAVE_STEPS=100] slurm/campaigns/train_sft.sbatch
#SBATCH --job-name=pj-train-sft
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
cd "$REPO"
source slurm/templates/common.sh
MIXDIR="$PATHJEV_WORK/mixes/${MIX:?}"
pathjev_log_provenance "$MIXDIR/mix_info.json" "$MIXDIR/train.jsonl" "${SEED:?}"
# D-020: Gen-SFT max_steps = PathJev Stage B optimiser steps on the same mix (batch 8 x accum 4)
if [ -n "${MAX_STEPS_FROM_MIX:-}" ]; then
    N_STATES=$(grep -c . "$MIXDIR/train.jsonl")
    MAX_STEPS=$(( ( (N_STATES + 7) / 8 + 3 ) / 4 * ${EPOCHS:-3} ))
    echo "max_steps from mix: $N_STATES states -> $MAX_STEPS"
fi
pathjev_register train_sft "$MIXDIR/mix_info.json" "$SEED" "$MIXDIR/train.jsonl" "$PATHJEV_WORK/runs/train_${RUN_NAME:?}" "generative SFT baseline (b), mix=$MIX lr=${LR:-2e-4} epochs=${EPOCHS:-3} max_steps=${MAX_STEPS:--1}"
OUT="$PATHJEV_WORK/runs/train_${RUN_NAME:?}"
mkdir -p "$OUT" && pathjev_snapshot_code "$OUT/code_${SLURM_JOB_ID}" && cp "$MIXDIR/mix_info.json" "$OUT/"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$REPO/external/OmniMed-Jev"
[ -f "$OUT/.done" ] && { echo "already done"; exit 0; }
RESUME=$({ ls -d "$OUT"/checkpoint-* 2>/dev/null || true; } | sort -t- -k2 -n | tail -1)
python -u -m pathjev.train_sft --train "$MIXDIR/train.jsonl" --eval "$MIXDIR/val.jsonl" --out "$OUT" \
    --model "$PATHJEV_WORK/models/medgemma-1.5-4b-it" --seed "$SEED" --epochs "${EPOCHS:-3}" --lr "${LR:-2e-4}" \
    --max-steps "${MAX_STEPS:--1}" --save-steps "${SAVE_STEPS:-100}" ${RESUME:+--resume "$RESUME"} 2>&1 | tee -a "$OUT/train.log"
