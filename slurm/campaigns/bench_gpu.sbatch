#!/bin/bash
# D-027 engineering benchmark on one GPU type (submit with --gres=gpu:h100:1 or gpu:l40:1):
#   1) cached vs full-sequence SFT read-out: equivalence + speed (pathjev.check_readout);
#   2) PathJev Stage B training throughput: 15 optimiser steps on the pilot-p mix;
#   3) PathJev inference throughput: 500 test states.
# Engineering numbers only (no reported result).
#SBATCH --job-name=pj-bench
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
cd "$REPO"
source slurm/templates/common.sh
UP="$REPO/external/OmniMed-Jev"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$UP"
GPU=$(nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=name --format=csv,noheader | tr ' ' '_')
RUN="$PATHJEV_WORK/runs/bench_${GPU}_${SLURM_JOB_ID}"
mkdir -p "$RUN"
pathjev_log_provenance "$PATHJEV_WORK/mixes/pilot-p/mix_info.json" "$PATHJEV_WORK/mixes/pilot-p/train.jsonl" 1
pathjev_register bench_gpu none none "$PATHJEV_WORK/mixes/pilot-p/train.jsonl" "$RUN" "ENGINEERING (D-027): $GPU read-out equivalence/speed, training and inference throughput"
pathjev_snapshot_code "$RUN/code"
MODEL="$PATHJEV_WORK/models/medgemma-1.5-4b-it"

python -u -m pathjev.check_readout --states "$PATHJEV_WORK/mixes/pilot-p/val.jsonl" --model "$MODEL" \
    --adapter "$PATHJEV_WORK/runs/train_pilot-sft-p-s1/checkpoint-563" --n 40 --out "$RUN/readout_check.json"

start=$(date +%s)
(cd "$UP" && python -u scripts/train_medjev.py --stage lora --init-head "$PATHJEV_WORK/runs/train_pilot-p-s1/head" \
    --output-dir "$RUN/train" --num-epochs 1 --max-steps 15 --save-steps 1000 --eval-steps 1000 \
    --train "$PATHJEV_WORK/mixes/pilot-p/train.jsonl" --eval "$PATHJEV_WORK/mixes/pilot-p/val.jsonl" --model "$MODEL" \
    --seed 1 --image-size 896 --max-length 3072 --shuffle-candidates --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 4 --learning-rate 2e-4 --dataloader-num-workers 6) 2>&1 | tee "$RUN/train.log" | tail -5
train_s=$(( $(date +%s) - start ))

head -500 "$PATHJEV_WORK/eval_sets/pilot-stage1/hepatobench.jsonl" > "$RUN/infer_states.jsonl"
start=$(date +%s)
(cd "$UP" && python -u scripts/infer_medjev.py --checkpoint "$PATHJEV_WORK/runs/train_pilot-p-s1/lora/checkpoint-563" \
    --model "$MODEL" --data "$RUN/infer_states.jsonl" --split eval --out-dir "$RUN/infer" --image-size 896 --batch-size 8) \
    > "$RUN/infer.log" 2>&1
infer_s=$(( $(date +%s) - start ))
echo "{\"gpu\": \"$GPU\", \"train_15_steps_wall_s\": $train_s, \"infer_500_states_wall_s\": $infer_s}" | tee "$RUN/timing.json"
grep -aoE "[0-9]+/15 \[[0-9:]+<[0-9:]+, *[0-9.]+s/it" "$RUN/train.log" | tail -1 | tee -a "$RUN/timing.json"
