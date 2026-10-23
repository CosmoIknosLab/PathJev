#!/bin/bash
# Qualitative WSI demo (D-037), one array task per slide listed in $PATHJEV_WORK/wsi_demo/<name>/slides.json:
# tiles -> H-optimus-0 features -> PCA + GMM -> PathJev-P on cluster representatives and on every tile -> figure.
#   source slurm/cluster.env && sbatch --array=0-3 --export=ALL,CONFIG=configs/wsi/demo-hiesd.yaml slurm/campaigns/wsi_demo.sbatch
#SBATCH --job-name=pj-wsi-demo
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
cd "$REPO"
source slurm/templates/common.sh
UP="$REPO/external/OmniMed-Jev"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$UP"
MODEL="$PATHJEV_WORK/models/medgemma-1.5-4b-it"
CONFIG="${CONFIG:?}"
NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['name'])")
RUNNAME=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['labelling']['model_run'])")
BASE="$PATHJEV_WORK/wsi_demo/$NAME"
SLIDE=$(python -c "import json; print(json.load(open('$BASE/slides.json'))['chosen'][${SLURM_ARRAY_TASK_ID:-0}])")
OUT="$BASE/$SLIDE"
CHECKPOINT=$(cat "$PATHJEV_WORK/runs/train_$RUNNAME/selection/best.txt")
mkdir -p "$OUT"
pathjev_log_provenance "$CHECKPOINT" "$CONFIG" none
pathjev_register "wsi_demo" "$CONFIG" none "$CONFIG" "$OUT" "slide=$SLIDE checkpoint=$CHECKPOINT"
pathjev_snapshot_code "$OUT/code_${SLURM_JOB_ID}"

[ -s "$OUT/tiles.jsonl" ] || python -u -m pathjev.wsi_demo tile --config "$CONFIG" --slide "$SLIDE"
echo "tiles: $(grep -c . "$OUT/tiles.jsonl")"
[ -s "$OUT/tiles.npz" ] || python -u -m pathjev.baselines features --states "$OUT/tiles.jsonl" --out "$OUT/tiles.npz"
python -u -m pathjev.wsi_demo cluster --config "$CONFIG" --slide "$SLIDE"
for part in reps tiles; do
    dest=$([ $part = reps ] && echo reps_decisions.jsonl || echo dense_decisions.jsonl)
    start=$(date +%s)
    (cd "$UP" && python -u scripts/infer_medjev.py --checkpoint "$CHECKPOINT" --model "$MODEL" --data "$OUT/$part.jsonl" \
        --split eval --out-dir "$OUT/infer_$part" --image-size 896 --batch-size 8) > "$OUT/infer_$part.log" 2>&1
    mv "$OUT/infer_$part/eval_decisions.jsonl" "$OUT/$dest"
    echo "{\"part\": \"$part\", \"states\": $(grep -c . "$OUT/$part.jsonl"), \"seconds\": $(( $(date +%s) - start ))}" >> "$OUT/timing.jsonl"
done
python -u -m pathjev.wsi_demo assemble --config "$CONFIG" --slide "$SLIDE" --figdir "$REPO/paper/figures"
