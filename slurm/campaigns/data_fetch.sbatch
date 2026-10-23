#!/bin/bash
# Download a priority-1 training source (CPU job, network-bound). Idempotent and
# requeue-safe: verified files are skipped. Usage from the repository root:
#   sbatch --export=ALL,SOURCE=spider slurm/campaigns/data_fetch.sbatch
#   sbatch --export=ALL,SOURCE=panda  slurm/campaigns/data_fetch.sbatch
#SBATCH --job-name=pj-fetch
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu                        # set to your CPU partition (slurm/README.md)
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh
pathjev_log_provenance none none none
export HF_HUB_OFFLINE=0   # this job downloads
DATA="${PATHJEV_DATA:-$HOME/pathjev/data}"
mkdir -p "$DATA"

PYTHONPATH="$REPO/src" python -u -m data.fetch_external "${SOURCE:?set SOURCE (see fetch_external.FETCHERS)}" --out "$DATA" --workers 4
du -sh "$DATA/${SOURCE}"
