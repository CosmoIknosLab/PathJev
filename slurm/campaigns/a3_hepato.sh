#!/bin/bash
# Amendment A3.1 (post-hoc): HepatoBench grouping sensitivity. CPU only, through cpu_task.sbatch:
#   source slurm/cluster.env && sbatch --job-name=pj-a3-hepato --cpus-per-task=8 --mem=48G --time=02:00:00 \
#       --export=ALL,CMD="bash slurm/campaigns/a3_hepato.sh" slurm/campaigns/cpu_task.sbatch
set -euo pipefail
source slurm/templates/common.sh
pathjev_register "a3_hepato" docs/stage1_analysis_plan.md none "$PATHJEV_WORK/eval_sets/stage1-test/info.json" \
    results/runs/stage1-a3-hepato/summary.json "A3.1 post-hoc HepatoBench grouping sensitivity (D-040)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd analysis
python -u a3_hepato.py --work "$PATHJEV_WORK" --out ../results/runs/stage1-a3-hepato --resamples 10000 --procs 4
