#!/bin/bash
# Plan A3.2–A3.5 analysis (D-040), from the eval_a3-* run folders:
#   source slurm/cluster.env && sbatch --job-name=pj-a3-checks --export=ALL,CMD="bash slurm/campaigns/a3_checks.sh" slurm/campaigns/cpu_task.sbatch
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD/analysis"
python -u analysis/a3_checks.py --work "$PATHJEV_WORK" --out results/runs/stage1-a3 --tables paper/generated
