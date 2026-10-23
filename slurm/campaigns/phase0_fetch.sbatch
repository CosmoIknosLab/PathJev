#!/bin/bash
# Phase 0 step 1 (CPU): download MedGemma-1.5-4B + OmniMed-Jev-4B (pinned
# revisions) and MedMNIST+ 224 (~30 GB, MD5-verified), then write the PNGs the
# released test states reference. Submit from the repository root:
#   sbatch slurm/campaigns/phase0_fetch.sbatch
#SBATCH --job-name=pj-p0-fetch
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu                        # set to your CPU partition (slurm/README.md)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh
pathjev_log_provenance configs/phase0_repro.yaml none none
pathjev_snapshot_code "$PATHJEV_WORK/runs/phase0_fetch_${SLURM_JOB_ID}"

# Downloads are idempotent: verified files are skipped after a requeue.
PYTHONPATH="$REPO/src" python -u -m phase0.fetch_assets \
    --work "$PATHJEV_WORK" --upstream external/OmniMed-Jev ${SKIP_MODELS:+--skip-models}
du -sh "$PATHJEV_WORK"/models/* "$PATHJEV_WORK"/omnimedjev-datas/raw/medmnistplus
