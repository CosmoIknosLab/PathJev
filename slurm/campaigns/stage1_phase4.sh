#!/bin/bash
# Stage 1, Phase 4 campaign (D-024, D-027..D-029). Run from the repository root on the login node.
#   stage1_phase4.sh prep                      # mix stage1-p (40k) + evaluation sets (A1)
#   stage1_phase4.sh grid                      # lr {1e-4,2e-4} x epochs {3,5} x {PathJev, Gen-SFT}, seed 1;
#                                              #   D-020 checkpoint selection on val, NO test scoring
#   stage1_phase4.sh seeds <jev-run> <sft-run> # selected configs (analysis/select_config.py): seeds 2, 3
#                                              #   trained + selected + tested; seed-1 runs tested
#   stage1_phase4.sh baselines                 # (a) x3 variants, (c), (e), (d) on stage1-test; secondary full sets
# Add --dry-run as last argument to print the sbatch commands only. GPU jobs request H100 (D-028).
set -euo pipefail
MODE="${1:?prep|grid|seeds|baselines}"; shift
DRY=""; [ "${!#:-}" = "--dry-run" ] && DRY=--dry-run
cd "$(dirname "$0")/../.."
source slurm/cluster.env
W="${PATHJEV_WORK:-$HOME/pathjev/work}"
LOG="logs/slurm/phase4_${MODE}_$(date +%Y%m%d_%H%M%S)${DRY:+_dry}.txt"; mkdir -p logs/slurm; : > "$LOG"
GPU=(--gres=gpu:h100:1)
export PJ_GRES=gpu:h100:1   # inherited by post_train for its val/test arrays
export PJ_GRES_SFT=gpu:h100:2   # Gen-SFT val/test scoring sharded over 2 GPUs (D-034)
SAVE=1250   # optimiser steps per epoch: 40,000 states / (8 x 4)
sub() { local label="$1"; shift; local id
    if [ -n "$DRY" ]; then id=$(( $(wc -l < "$LOG") + 9001 )); echo "[dry] $label: sbatch $*" >&2
    else id=$(sbatch --parsable "$@"); fi
    echo "$label $id" >> "$LOG"; echo "$id"; }
TEST="$W/eval_sets/stage1-test"; FULL="$W/eval_sets/stage1-test-full"
train_and_select() {  # arm lr epochs seed name eval_set(or empty for NO_TEST) [dependency]
    local arm=$1 lr=$2 ep=$3 seed=$4 name=$5 set=$6 dep=${7:-}
    local script=$([ "$arm" = jev ] && echo train_jev || echo train_sft)
    local extra=$([ "$arm" = sft ] && echo ",MAX_STEPS_FROM_MIX=1" || echo "")
    local t=$(sub "train-$name" "${GPU[@]}" ${dep:+--dependency=afterok:$dep} -J "pj-$name" \
        --export=ALL,MIX=stage1-p,RUN_NAME=$name,SEED=$seed,EPOCHS=$ep,LR=$lr,SAVE_STEPS=$SAVE$extra slurm/campaigns/$script.sbatch)
    local exp="ALL,ARM=$arm,RUN_NAME=$name,MIX=stage1-p,EVAL_SET=${set:-$TEST}"
    [ -z "$set" ] && exp="$exp,NO_TEST=1"
    sub "post-$name" --dependency=afterok:$t --export=$exp slurm/campaigns/post_train.sbatch >/dev/null
}
case "$MODE" in
prep)
    sub prep -J pj-prep --time=02:00:00 --export=ALL,CMD="python -m pathjev.mix --config configs/train/stage1-p.yaml && python -m pathjev.evalsets --config configs/eval/stage1-test.yaml && python -m pathjev.evalsets --config configs/eval/stage1-test-full.yaml" slurm/campaigns/cpu_task.sbatch >/dev/null ;;
grid)
    for arm in jev sft; do for lr in 1e-4 2e-4; do for ep in 3 5; do
        train_and_select $arm $lr $ep 1 "s1-$arm-lr$lr-e$ep-s1" ""
    done; done; done ;;
seeds)
    JEV="${1:?selected PathJev grid run}"; SFT="${2:?selected Gen-SFT grid run}"
    for pair in "jev $JEV" "sft $SFT"; do set -- $pair; arm=$1; run=$2
        lr=$(echo "$run" | sed -E 's/.*-lr([0-9.e-]+)-e([0-9]+)-s1$/\1/'); ep=$(echo "$run" | sed -E 's/.*-lr([0-9.e-]+)-e([0-9]+)-s1$/\2/')
        [[ "$lr" =~ ^[0-9.]+e-[0-9]+$ && "$ep" =~ ^[0-9]+$ ]] || { echo "cannot parse lr/epochs from $run"; exit 2; }
        for seed in 2 3; do train_and_select $arm $lr $ep $seed "${run%-s1}-s$seed" "$TEST"; done
        # seed-1 run: test its already-selected checkpoint
        SEL="$W/runs/train_$run/selection"; ls "$TEST"/*.jsonl > "/tmp/pj_testfiles_$run.txt"
        [ -z "$DRY" ] && cp "/tmp/pj_testfiles_$run.txt" "$SEL/test_files.txt"
        G=$([ "$arm" = sft ] && echo "--gres=$PJ_GRES_SFT" || echo "${GPU[@]}")
        sub "test-$run" $G --array=0-7%4 -J "pj-test-$run" \
            --export=ALL,ARM=$arm,NAME=test-$run,CHECKPOINT_FILE=$SEL/best.txt,FILE_LIST=$SEL/test_files.txt,EVAL_SET=,FILES=,CKPT_LIST=,RESAMPLES=2000 \
            slurm/campaigns/eval_arm.sbatch >/dev/null
    done ;;
baselines)
    for arm in omnimedjev conch letter; do
        sub "zs-$arm" "${GPU[@]}" -J "pj-s1-$arm" --export=ALL,ARM=$arm,NAME=s1-$arm,EVAL_SET=$TEST slurm/campaigns/eval_arm.sbatch >/dev/null
        sub "zs-$arm-full" "${GPU[@]}" -J "pj-s1-$arm-full" --export=ALL,ARM=$arm,NAME=s1-$arm-full,EVAL_SET=$FULL slurm/campaigns/eval_arm.sbatch >/dev/null
    done
    VA=$(sub variants -J pj-variants --time=00:30:00 --export=ALL,CMD="python -m pathjev.variants --src $TEST" slurm/campaigns/cpu_task.sbatch)
    for v in no-other permuted; do
        sub "zs-omnimedjev-$v" "${GPU[@]}" --dependency=afterok:$VA -J "pj-s1-omnimedjev-$v" \
            --export=ALL,ARM=omnimedjev,NAME=s1-omnimedjev-$v,EVAL_SET=$TEST-$v slurm/campaigns/eval_arm.sbatch >/dev/null
    done
    # (d): features of the training mix + primary sets, and of the full sets (separate run dirs:
    # the two set families share file names), then the probe on each family.
    if [ -n "${PROBE_ONLY:-}" ]; then F1=${F1_ID:?}; F2=${F2_ID:?}; else
    F1=$(sub features "${GPU[@]}" -J pj-s1-feat --export=ALL,ARM=features,NAME=s1-features,FILES="$W/mixes/stage1-p/train.jsonl $(printf "$TEST/%s.jsonl " bracs sicapv2 hepatobench hiesd spider-colorectal spider-thorax spider-breast spider-skin)" slurm/campaigns/eval_arm.sbatch)
    F2=$(sub features-full "${GPU[@]}" -J pj-s1-feat-full --export=ALL,ARM=features,NAME=s1-features-full,EVAL_SET=$FULL slurm/campaigns/eval_arm.sbatch); fi
    for fam in test full; do
        if [ $fam = test ]; then SETDIR=$TEST; SUF=""; else SETDIR=$FULL; SUF="-full"; fi
        FD=$W/runs/eval_s1-features$SUF; PD=$W/runs/eval_s1-probe$SUF
        sub "probe-$fam" --dependency=afterok:$F1:$F2 -J pj-s1-probe-$fam --cpus-per-task=16 --mem=128G --time=1-00:00:00 \
            --export=ALL,CMD="mkdir -p $PD && for f in $SETDIR/*.jsonl; do n=\$(basename \$f .jsonl); python -m pathjev.baselines probe --train-states $W/mixes/stage1-p/train.jsonl --train-feats $W/runs/eval_s1-features/train.npz --test-states \$f --test-feats $FD/\$n.npz --out $PD/\${n}_decisions.jsonl && python -m pathjev.evaluate --states \$f --decisions $PD/\${n}_decisions.jsonl --out $PD/\${n}_eval.json > $PD/\${n}_eval.log; done" \
            slurm/campaigns/cpu_task.sbatch >/dev/null
    done
    ;;
*) echo "usage: $0 prep|grid|seeds|baselines [--dry-run]"; exit 2 ;;
esac
echo "submitted ($MODE); ids in $LOG"
