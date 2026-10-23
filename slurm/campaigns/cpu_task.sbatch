#!/bin/bash
# Small CPU task inside the PathJev environment:  sbatch --export=ALL,CMD="python -m ..." slurm/campaigns/cpu_task.sbatch
#SBATCH --job-name=pj-cpu
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu                        # set to your CPU partition (slurm/README.md)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/templates/common.sh
pathjev_log_provenance none none none
export PYTHONPATH="$PWD/src/shims:$PWD/src:$PWD/external/OmniMed-Jev"
eval "${CMD:?}"
