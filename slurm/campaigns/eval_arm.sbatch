#!/bin/bash
# Score one arm on a list of state files (same states for every arm) and evaluate each file.
#   ARM=omnimedjev|jev|sft|conch|letter|features   NAME=<run name>
#   FILES="<a.jsonl> <b.jsonl> ..."  or  EVAL_SET=<dir of *.jsonl>
#   CHECKPOINT=<dir> | CHECKPOINT_FILE=<file holding a dir> | CKPT_LIST=<file, one dir per line; array job>
#   RESAMPLES=2000 (0 for checkpoint selection on val)   IMAGE_SIZE (default 448 omnimedjev, else 896)
#   source slurm/cluster.env && sbatch --export=ALL,ARM=conch,NAME=pilot-conch,EVAL_SET=... slurm/campaigns/eval_arm.sbatch
#SBATCH --job-name=pj-eval
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
cd "$REPO"
source slurm/templates/common.sh
UP="$REPO/external/OmniMed-Jev"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$UP"
MODEL="$PATHJEV_WORK/models/medgemma-1.5-4b-it"

if [ -n "${CKPT_LIST:-}" ]; then CHECKPOINT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CKPT_LIST"); fi
# FILE_LIST: one state file per array task (shards a slow arm by evaluation set; same outputs).
if [ -n "${FILE_LIST:-}" ]; then FILES=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILE_LIST"); fi
if [ -n "${CHECKPOINT_FILE:-}" ]; then CHECKPOINT=$(cat "$CHECKPOINT_FILE"); fi
[ "${ARM:?}" = omnimedjev ] && CHECKPOINT="${CHECKPOINT:-$PATHJEV_WORK/models/OmniMed-Jev-4B}"
# FILES wins over EVAL_SET (EVAL_SET can leak in from a parent job's environment).
if [ -z "${FILES:-}" ] && [ -n "${EVAL_SET:-}" ]; then FILES=$(ls "$EVAL_SET"/*.jsonl); fi
# Checkpoint-selection runs must read the validation mix only (D-020; no test data).
if [[ "${NAME:-}" == val-* ]]; then
    for f in ${FILES:?}; do [ "$(basename "$f")" = val.jsonl ] || { echo "refusing: selection run on $f"; exit 3; }; done
fi
SIZE="${IMAGE_SIZE:-$([ "$ARM" = omnimedjev ] && echo 448 || echo 896)}"
TAG="${NAME:?}${CHECKPOINT:+_$(basename "$CHECKPOINT")}"
RUN="$PATHJEV_WORK/runs/eval_${TAG}"
mkdir -p "$RUN"
pathjev_log_provenance "${CHECKPOINT:-none}" "$(echo ${FILES:?} | cut -d' ' -f1)" none
pathjev_register "eval_${ARM}" "${CHECKPOINT:-none}" none "$(echo $FILES | cut -d' ' -f1)" "$RUN" \
    "arm=$ARM name=$NAME image_size=$SIZE files=$(echo $FILES | wc -w)"
pathjev_snapshot_code "$RUN/code_${SLURM_JOB_ID}"
echo "checkpoint: ${CHECKPOINT:-none}" > "$RUN/checkpoint.txt"

nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 30 >> "$RUN/gpu_trace.csv" &
TRACE=$!
for f in $FILES; do
    set_name=$(basename "$f" .jsonl)
    dec="$RUN/${set_name}_decisions.jsonl"
    start=$(date +%s)
    case "$ARM" in
      omnimedjev|jev)
        (cd "$UP" && python -u scripts/infer_medjev.py --checkpoint "$CHECKPOINT" --model "$MODEL" --data "$f" \
            --split eval --out-dir "$RUN/$set_name" --image-size "$SIZE" --batch-size "${BATCH:-8}") \
            2>&1 | tee "$RUN/${set_name}_infer.log"
        mv "$RUN/$set_name/eval_decisions.jsonl" "$dec" ;;
      sft)    # one process per allocated GPU on contiguous shards, concatenated in order (D-034):
              # each state is scored independently, so the output equals a single-process run.
              IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"; NG=${#GPUS[@]}
              if [ "$NG" -eq 1 ]; then
                  python -u -m pathjev.generative --states "$f" --out "$dec" --readout text-slow --model "$MODEL" \
                      --adapter "$CHECKPOINT" --image-size "$SIZE"
              else
                  SH="$RUN/_shards_$set_name"; mkdir -p "$SH"
                  split -n l/"$NG" -d -a 2 "$f" "$SH/states_"
                  pids=()
                  for i in $(seq 0 $((NG - 1))); do p=$(printf %02d "$i")
                      CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -u -m pathjev.generative --states "$SH/states_$p" \
                          --out "$SH/dec_$p" --readout text-slow --model "$MODEL" --adapter "$CHECKPOINT" \
                          --image-size "$SIZE" > "$SH/log_$p" 2>&1 & pids+=($!)
                  done
                  for pid in "${pids[@]}"; do wait "$pid"; done   # set -e: any failed shard fails the job
                  cat $(for i in $(seq 0 $((NG - 1))); do printf "$SH/dec_%02d " "$i"; done) > "$dec"
                  [ "$(grep -c . "$dec")" -eq "$(grep -c . "$f")" ] || { echo "shard output incomplete"; exit 4; }
              fi ;;
      letter) python -u -m pathjev.generative --states "$f" --out "$dec" --readout letter --model "$MODEL" \
                  --image-size "$SIZE" ;;
      conch)  python -u -m pathjev.baselines conch --states "$f" --out "$dec" ;;
      features) python -u -m pathjev.baselines features --states "$f" --out "$RUN/${set_name}.npz" ;;
      *) echo "unknown ARM=$ARM"; exit 2 ;;
    esac
    echo "{\"set\": \"$set_name\", \"states\": $(grep -c . "$f"), \"seconds\": $(( $(date +%s) - start ))}" >> "$RUN/timing.jsonl"
    if [ "$ARM" != features ]; then
        python -u -m pathjev.evaluate --states "$f" --decisions "$dec" --out "$RUN/${set_name}_eval.json" \
            --resamples "${RESAMPLES:-2000}" > "$RUN/${set_name}_eval.log"
    fi
done
kill $TRACE 2>/dev/null || true
echo "run dir: $RUN"
