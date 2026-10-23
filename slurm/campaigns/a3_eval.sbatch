#!/bin/bash
# Plan amendment A3.2–A3.5 (docs/stage1_analysis_plan.md, D-040): one array task per line of TASK_LIST
# (slurm/campaigns/a3_tasks.sh writes it). Line format, space-separated:
#   ARM TAG CHECKPOINT EVAL_DIR IMAGE_SIZE MODE SETS...
#   ARM  = jev | omnimedjev | sft          MODE = all (every decision) | main (main Choice only)
#                                                 | variants (main Choice, primary + A3.4 read-out variants)
#   SETS = set names (files EVAL_DIR/<set>.jsonl)
#   source slurm/cluster.env && sbatch --array=0-N%8 --export=ALL,TASK_LIST=<file> slurm/campaigns/a3_eval.sbatch
#SBATCH --job-name=pj-a3
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
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

read -r ARM TAG CHECKPOINT EVAL_DIR SIZE MODE SETS <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_LIST:?}")"
RUN="$PATHJEV_WORK/runs/eval_a3-$TAG"
mkdir -p "$RUN"
FIRST="$EVAL_DIR/${SETS%% *}.jsonl"
pathjev_log_provenance "$CHECKPOINT" "$FIRST" none
pathjev_register "a3_eval_${ARM}" "$CHECKPOINT" none "$FIRST" "$RUN" \
    "plan=A3 arm=$ARM tag=$TAG image_size=$SIZE mode=$MODE eval_dir=$EVAL_DIR sets=$(echo $SETS | tr ' ' '+')"
pathjev_snapshot_code "$RUN/code_${SLURM_JOB_ID}"
echo "checkpoint: $CHECKPOINT" > "$RUN/checkpoint.txt"

for s in $SETS; do
    f="$EVAL_DIR/$s.jsonl"
    dec="$RUN/${s}_decisions.jsonl"
    if [ -s "$dec" ] && [ "$(grep -c . "$dec")" -eq "$(grep -c . "$f")" ]; then echo "done: $s"; continue; fi
    start=$(date +%s)
    case "$ARM" in
      jev|omnimedjev)
        [ "$MODE" = all ] || { echo "MODE=$MODE not supported for $ARM"; exit 2; }
        (cd "$UP" && python -u scripts/infer_medjev.py --checkpoint "$CHECKPOINT" --model "$MODEL" --data "$f" \
            --split eval --out-dir "$RUN/$s" --image-size "$SIZE" --batch-size 8) > "$RUN/${s}_infer.log" 2>&1
        mv "$RUN/$s/eval_decisions.jsonl" "$dec" ;;
      sft)
        QID=$(python -c "import sys; sys.path.insert(0, 'analysis'); from stage1_analysis import MAIN_Q; print(MAIN_Q['$s'])")
        EXTRA=()
        [ "$MODE" = variants ] && EXTRA=(--variants-out "$RUN/${s}_variants.jsonl")
        [ "$MODE" = all ] && QARGS=() || QARGS=(--qids "$QID")
        python -u -m pathjev.generative --states "$f" --out "$dec" --readout text-slow --model "$MODEL" \
            --adapter "$CHECKPOINT" --image-size "$SIZE" "${QARGS[@]}" "${EXTRA[@]}" > "$RUN/${s}_infer.log" 2>&1 ;;
      *) echo "unknown ARM=$ARM"; exit 2 ;;
    esac
    echo "{\"set\": \"$s\", \"states\": $(grep -c . "$f"), \"seconds\": $(( $(date +%s) - start ))}" >> "$RUN/timing.jsonl"
done
echo "run dir: $RUN"
