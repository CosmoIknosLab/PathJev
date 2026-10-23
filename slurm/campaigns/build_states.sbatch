#!/bin/bash
# Build records, split manifest and decision states for one source (CPU).
#   sbatch --export=ALL,BUILD=configs/build/bracs.yaml -J pj-build-bracs slurm/campaigns/build_states.sbatch
#SBATCH --job-name=pj-build
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu                        # set to your CPU partition (slurm/README.md)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh
# Array mode: BUILD_LIST=<file with one build config per line>, --array=0-(N-1)%4
if [ -n "${BUILD_LIST:-}" ]; then BUILD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BUILD_LIST"); fi
pathjev_log_provenance "${BUILD:?set BUILD=configs/build/<source>.yaml}" none none
pathjev_register build_states "$BUILD" none none "" "states + split manifest for $(basename "$BUILD" .yaml)"
pathjev_snapshot_code "$PATHJEV_WORK/runs/build_${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
PYTHONPATH="$REPO/src" python -u -m pathjev.build --config "$BUILD" | tee "$PATHJEV_WORK/runs/build_${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}/build_info.json"
