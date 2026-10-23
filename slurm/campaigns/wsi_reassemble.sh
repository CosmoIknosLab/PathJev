#!/bin/bash
# Recompute demo_stats.json (and the per-slide figure in the run folder) for every slide of the WSI demo,
# from existing outputs only (no tiling, clustering or inference). Run through cpu_task.sbatch:
#   source slurm/cluster.env && sbatch --job-name=pj-wsi-reassemble --export=ALL,CMD="bash slurm/campaigns/wsi_reassemble.sh" slurm/campaigns/cpu_task.sbatch
set -euo pipefail
CONFIG=configs/wsi/demo-hiesd.yaml
NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['name'])")
for SLIDE in $(python -c "import json; print(' '.join(json.load(open('$PATHJEV_WORK/wsi_demo/$NAME/slides.json'))['chosen']))"); do
    python -u -m pathjev.wsi_demo assemble --config "$CONFIG" --slide "$SLIDE"
done
