#!/bin/bash
# Check the release inference package (release/pathjev_infer) against stored PathJev-P seed-1 test outputs.
#   GPU:  source slurm/cluster.env && sbatch slurm/campaigns/release_check.sbatch
#   CPU:  source slurm/cluster.env && sbatch --partition=cpu --gres=none --export=ALL,DEVICE=cpu slurm/campaigns/release_check.sbatch
#SBATCH --job-name=pj-release-check
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
cd "$REPO"
source slurm/templates/common.sh
pathjev_log_provenance "$REPO/release/weights" none none
python -u release/tests/check_release.py --work "$PATHJEV_WORK" --base "$PATHJEV_WORK/models/medgemma-1.5-4b-it" \
    ${DEVICE:+--device $DEVICE}
