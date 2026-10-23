#!/bin/bash
# Run a decision model (upstream scripts/infer_medjev.py) on PathJev states and score it.
#   sbatch --export=ALL,STATES=<states.jsonl>,NAME=<run name>[,CHECKPOINT=..][,LIMIT=N][,IMAGE_SIZE=448] \
#          slurm/campaigns/infer_eval.sbatch
# Defaults: released OmniMed-Jev-4B (baseline (a), zero-shot) at its training resolution 448.
#SBATCH --job-name=pj-infer
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh
pathjev_log_provenance "${STATES:?set STATES}" none none
UP="$REPO/external/OmniMed-Jev"
CHECKPOINT="${CHECKPOINT:-$PATHJEV_WORK/models/OmniMed-Jev-4B}"
MODEL="${MODEL:-$PATHJEV_WORK/models/medgemma-1.5-4b-it}"
RUN="$PATHJEV_WORK/runs/infer_${NAME:?set NAME}_${SLURM_JOB_ID}"
pathjev_snapshot_code "$RUN"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$UP"

nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > "$RUN/gpu_trace.csv" &
TRACE=$!
trap 'kill $TRACE 2>/dev/null || true' EXIT

INPUT="$STATES"
if [ -n "${LIMIT:-}" ]; then   # deterministic subsample: every k-th state
    python -c "
import sys
rows=open('$STATES').read().splitlines(); n=int('$LIMIT'); step=max(1,len(rows)//n)
open('$RUN/states.jsonl','w').write('\n'.join(rows[::step][:n])+'\n')"
    INPUT="$RUN/states.jsonl"
fi

cd "$UP"
python -u scripts/infer_medjev.py --checkpoint "$CHECKPOINT" --model "$MODEL" --data "$INPUT" \
    --split eval --out-dir "$RUN" --image-size "${IMAGE_SIZE:-448}" --batch-size "${BATCH:-8}" 2>&1 | tee "$RUN/infer.log"
cd "$REPO"
python -u -m pathjev.evaluate --states "$INPUT" --decisions "$RUN/eval_decisions.jsonl" --out "$RUN/eval.json" \
    | tee "$RUN/eval.log"
echo "run dir: $RUN"
