#!/bin/bash
# Two-stage PathJev training with upstream scripts/train_medjev.py (OmniMed-Jev recipe):
# Stage A = head only (HEAD_STEPS on a frozen backbone), Stage B = head + LoRA r16 a16 all-linear.
# Requeue-safe: each stage resumes from its newest checkpoint-N, and a finished
# stage is skipped. Checkpoints every SAVE_STEPS (keep it ~30 min of work: low-QoS
# preemption kills after 30 s, GraceTime 0).
#   sbatch --export=ALL,MIX=pathjev-p,RUN_NAME=pathjev-p-s1,SEED=1[,EPOCHS=3,SAVE_STEPS=100,SMOKE=1] \
#          slurm/campaigns/train_jev.sbatch
#SBATCH --job-name=pj-train
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh
MIXDIR="$PATHJEV_WORK/mixes/${MIX:?set MIX}"
pathjev_log_provenance "$MIXDIR/mix_info.json" "$MIXDIR/train.jsonl" "${SEED:?set SEED}"
pathjev_register train_jev "$MIXDIR/mix_info.json" "$SEED" "$MIXDIR/train.jsonl" "$PATHJEV_WORK/runs/train_${RUN_NAME:?}" "PathJev two-stage training, mix=$MIX lr=${LR:-2e-4} epochs=${EPOCHS:-3}"
OUT="$PATHJEV_WORK/runs/train_${RUN_NAME:?set RUN_NAME}"
mkdir -p "$OUT"
pathjev_snapshot_code "$OUT/code_${SLURM_JOB_ID}"
cp "$MIXDIR/mix_info.json" "$OUT/mix_info.json"
UP="$REPO/external/OmniMed-Jev"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$UP"
MODEL="$PATHJEV_WORK/models/medgemma-1.5-4b-it"
COMMON=(--train "$MIXDIR/train.jsonl" --eval "$MIXDIR/val.jsonl" --model "$MODEL" --seed "$SEED"
        --image-size "${IMAGE_SIZE:-896}" --max-length "${MAX_LENGTH:-3072}" --shuffle-candidates
        --per-device-train-batch-size "${BATCH:-8}" --gradient-accumulation-steps "${ACCUM:-4}"
        --learning-rate "${LR:-2e-4}" --dataloader-num-workers 6)
[ -n "${SMOKE:-}" ] && COMMON+=(--smoke)

latest() { { ls -d "$1"/checkpoint-* 2>/dev/null || true; } | sort -t- -k2 -n | tail -1; }
trap 'echo "[slurm] USR1: time limit near; the next requeue resumes from the newest checkpoint"' USR1

nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 30 >> "$OUT/gpu_trace.csv" &
TRACE=$!
trap 'rc=$?; kill $TRACE 2>/dev/null || true; pathjev__register_end $rc' EXIT
cd "$UP"

if [ ! -f "$OUT/head/medjev_head.safetensors" ]; then
    RESUME=$(latest "$OUT/head"); ARGS=(); [ -n "$RESUME" ] && ARGS=(--resume-from-checkpoint "$RESUME")
    python -u scripts/train_medjev.py --stage head --head-steps "${HEAD_STEPS:-200}" --output-dir "$OUT/head" \
        --save-steps "${SAVE_STEPS:-100}" "${COMMON[@]}" "${ARGS[@]}" 2>&1 | tee -a "$OUT/stage_a.log"
fi
if [ ! -f "$OUT/lora/.done" ]; then
    RESUME=$(latest "$OUT/lora"); ARGS=(); [ -n "$RESUME" ] && ARGS=(--resume-from-checkpoint "$RESUME")
    python -u scripts/train_medjev.py --stage lora --init-head "$OUT/head" --output-dir "$OUT/lora" \
        --num-epochs "${EPOCHS:-3}" --save-steps "${SAVE_STEPS:-100}" --eval-steps "${SAVE_STEPS:-100}" \
        --save-total-limit 0 "${COMMON[@]}" "${ARGS[@]}" 2>&1 | tee -a "$OUT/stage_b.log"
    touch "$OUT/lora/.done"
fi
echo "run dir: $OUT"
