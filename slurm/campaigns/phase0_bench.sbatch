#!/bin/bash
# Phase 0 step 3 (GPU): memory/throughput of single-patch vs. mosaic vs.
# multi-image states, inference (released checkpoint) and one training step.
#   sbatch --dependency=afterok:<repro_job> slurm/campaigns/phase0_bench.sbatch
#SBATCH --job-name=pj-p0-bench
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh
pathjev_log_provenance none none 0

UP="$REPO/external/OmniMed-Jev"
RUN="$PATHJEV_WORK/runs/phase0_bench_${SLURM_JOB_ID}"
pathjev_snapshot_code "$RUN"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$UP"

python -u -m phase0.bench_states \
    --model "$PATHJEV_WORK/models/medgemma-1.5-4b-it" \
    --checkpoint "$PATHJEV_WORK/models/OmniMed-Jev-4B" \
    --pathmnist-npz "$PATHJEV_WORK/omnimedjev-datas/raw/medmnistplus/pathmnist_224.testonly.npz" \
    --states "$UP/data/splits/testing_states.jsonl" \
    --out-dir "$RUN" --seed 0
echo "run dir: $RUN"
