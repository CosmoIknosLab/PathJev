#!/bin/bash
# Extract SPIDER central patches only (D-013/D-017); CPU only, no GPU requested.
#   sbatch slurm/campaigns/extract_spider.sbatch
#SBATCH --job-name=pj-extract-spider
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu                        # set to your CPU partition (slurm/README.md)
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/templates/common.sh
pathjev_log_provenance none none none
PYTHONPATH="$PWD/src" python -u -m data.spider_extract --root "$PATHJEV_DATA/spider"
