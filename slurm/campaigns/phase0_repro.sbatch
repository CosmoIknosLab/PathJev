#!/bin/bash
# Phase 0 step 2 (GPU): run the released OmniMed-Jev checkpoint on the released
# test states that Phase 0 materialised, score with upstream's code, compare
# against the reference values fixed in configs/phase0_repro.yaml, and log
# GPU memory over time. Submit after phase0_fetch:
#   sbatch --dependency=afterok:<fetch_job> slurm/campaigns/phase0_repro.sbatch
#SBATCH --job-name=pj-p0-repro
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh
CONFIG=configs/phase0_repro.yaml
pathjev_log_provenance "$CONFIG" none none
pathjev_register phase0_repro "$CONFIG" none none "$PATHJEV_WORK/runs/phase0_repro_${SLURM_JOB_ID}" "Phase 0 repro on ADHOC (D-022); compare with the earlier reproduction: bal. acc 0.7029, ECE 0.1445"

UP="$REPO/external/OmniMed-Jev"
RUN="$PATHJEV_WORK/runs/phase0_repro_${SLURM_JOB_ID}"
pathjev_snapshot_code "$RUN"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$UP"

# GPU memory trace (MiB every 2 s) for the throughput/memory report.
nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits \
    -l 2 > "$RUN/gpu_trace.csv" &
TRACE=$!
trap 'rc=$?; kill $TRACE 2>/dev/null || true; pathjev__register_end $rc' EXIT

python -u -m phase0.filter_states --upstream "$UP" \
    --states data/splits/testing_states.jsonl --out "$RUN/testing.jsonl" | tee "$RUN/filter.json"

cd "$UP"
/usr/bin/time -v python -u scripts/infer_medjev.py \
    --checkpoint "$PATHJEV_WORK/models/OmniMed-Jev-4B" \
    --model "$PATHJEV_WORK/models/medgemma-1.5-4b-it" \
    --data "$RUN/testing.jsonl" --split testing --out-dir "$RUN" \
    2>&1 | tee "$RUN/infer.log"

python -u scripts/eval_medjev_metrics.py --states "$RUN/testing.jsonl" \
    --decisions "$RUN/testing_decisions.jsonl" --output "$RUN/metrics.json" | tee "$RUN/metrics.log"
cd "$REPO"

python -u -m phase0.score_repro --config "$CONFIG" --upstream "$UP" --run-dir "$RUN"
echo "run dir: $RUN"
