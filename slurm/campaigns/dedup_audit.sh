#!/bin/bash
# Near-duplicate / provenance audit of the Stage 1 splits (CLAUDE.md §0.4.5). Run through cpu_task.sbatch:
#   source slurm/cluster.env && sbatch --job-name=pj-dedup-audit --cpus-per-task=32 --mem=96G --time=06:00:00 \
#       --export=ALL,CMD="bash slurm/campaigns/dedup_audit.sh" slurm/campaigns/cpu_task.sbatch
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD/analysis:${PYTHONPATH:-}"
python -u analysis/dedup_audit.py --work "$PATHJEV_WORK" --out results/runs/stage1-dedup --workers "${SLURM_CPUS_PER_TASK:-8}"
