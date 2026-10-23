#!/bin/bash
# After a training run: score every saved checkpoint on the full validation mix (array),
# select by the D-020 rule (min mean val NLL over Choice decisions), then score the
# selected checkpoint on the evaluation set (skipped with NO_TEST=1, e.g. grid runs).
# Submitted with afterok on the training job.
#   ARM=jev|sft RUN_NAME=<train run> MIX=<mix name> EVAL_SET=<dir>
#SBATCH --job-name=pj-post-train
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=cpu                        # set to your CPU partition (slurm/README.md)
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
cd "$REPO"
source slurm/templates/common.sh
OUT="$PATHJEV_WORK/runs/train_${RUN_NAME:?}"
CK_ROOT=$([ "${ARM:?}" = jev ] && echo "$OUT/lora" || echo "$OUT")
LIST="$OUT/checkpoints.txt"
{ ls -d "$CK_ROOT"/checkpoint-* 2>/dev/null || true; } | sort -t- -k2 -n > "$LIST"
N=$(grep -c . "$LIST" || true)
[ "$N" -gt 0 ] || { echo "no checkpoints in $CK_ROOT"; exit 1; }
VAL="$PATHJEV_WORK/mixes/${MIX:?}/val.jsonl"
# Gen-SFT scoring is the slow arm: PJ_GRES_SFT (e.g. gpu:h100:2) shards it over GPUs (D-034).
GRES=$([ "$ARM" = sft ] && echo "${PJ_GRES_SFT:-${PJ_GRES:-gpu:1}}" || echo "${PJ_GRES:-gpu:1}")
SEL="$OUT/selection"
mkdir -p "$SEL"
A=$(sbatch --parsable --gres="$GRES" --array=0-$((N - 1))%4 -J "pj-val-${RUN_NAME}" \
    --export=ALL,ARM="$ARM",NAME="val-${RUN_NAME}",CKPT_LIST="$LIST",FILES="$VAL",EVAL_SET=,RESAMPLES=0 \
    slurm/campaigns/eval_arm.sbatch)
PAIRS=""
while read -r ck; do PAIRS="$PAIRS $ck=$PATHJEV_WORK/runs/eval_val-${RUN_NAME}_$(basename "$ck")/val_eval.json"; done < "$LIST"
S=$(sbatch --parsable --dependency=afterok:"$A" -J "pj-select-${RUN_NAME}" \
    --export=ALL,CMD="python -m pathjev.select_checkpoint $PAIRS --out $SEL/selection.json > $SEL/best.txt" \
    slurm/campaigns/cpu_task.sbatch)
if [ -n "${NO_TEST:-}" ]; then   # grid runs: selection only, test data never touched
    echo "checkpoints: $N  val array: $A  select: $S  test: none (NO_TEST)" | tee "$SEL/jobs.txt"; exit 0
fi
ls "${EVAL_SET:?}"/*.jsonl > "$SEL/test_files.txt"
NT=$(grep -c . "$SEL/test_files.txt")
T=$(sbatch --parsable --gres="$GRES" --dependency=afterok:"$S" -J "pj-test-${RUN_NAME}" --array=0-$((NT - 1))%4 \
    --export=ALL,ARM="$ARM",NAME="test-${RUN_NAME}",CHECKPOINT_FILE="$SEL/best.txt",FILE_LIST="$SEL/test_files.txt",EVAL_SET=,FILES=,CKPT_LIST=,RESAMPLES=2000 \
    slurm/campaigns/eval_arm.sbatch)
echo "checkpoints: $N  val array: $A  select: $S  test: $T" | tee "$SEL/jobs.txt"
