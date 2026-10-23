#!/bin/bash
# Stage 1 (D-024) job chains on ADHOC. Run from the repository root on the login node:
#   slurm/campaigns/stage1_chain.sh validate [--dry-run]   # cluster/data/repro/smoke checks
#   slurm/campaigns/stage1_chain.sh pilot    [--dry-run]   # Phase 3 pilot, Stage 1 arms (D-020)
# validate: GPU probe; checksum verification of the copied data; rebuild of all states
#   with a byte-level check of data/manifests against EXPECTED.sha256; Phase 0 repro;
#   smoke runs of every arm. pilot: PathJev-P and Gen-SFT (1 epoch, same step budget),
#   checkpoint selection on val (D-020), and the zero-shot / probe baselines, all scored
#   on the same evaluation subsets (configs/eval/pilot-stage1.yaml); baseline (a) also without
#   the "other" candidate and with permuted candidate order (pathjev.variants).
# Job IDs are written to logs/slurm/chain_<mode>_<date>.txt.

set -euo pipefail
MODE="${1:?validate|pilot}"
DRY="${2:-}"
cd "$(dirname "$0")/../.."
source slurm/cluster.env
mkdir -p logs/slurm
LOG="logs/slurm/chain_${MODE}_$(date +%Y%m%d_%H%M%S)${DRY:+_dry}.txt"
: > "$LOG"
sub() {   # sub <label> <sbatch args...>; prints the job id
    local label="$1"; shift
    local id
    if [ "$DRY" = "--dry-run" ]; then id=$(( $(wc -l < "$LOG" 2>/dev/null || echo 0) + 1001 )); echo "[dry] $label: sbatch $*" >&2
    else id=$(sbatch --parsable "$@"); fi
    echo "$label $id" >> "$LOG"
    echo "$id"
}
W="${PATHJEV_WORK:-$HOME/pathjev/work}"   # same default as slurm/templates/common.sh

case "$MODE" in
validate)
    P=$(sub probe slurm/campaigns/gpu_probe.sbatch)
    V=$(sub verify -J pj-verify --cpus-per-task=4 --time=06:00:00 \
        --export=ALL,CMD="python -m data.verify_fetch --data \$PATHJEV_DATA --out \$PATHJEV_WORK/verify_fetch.json" \
        slurm/campaigns/cpu_task.sbatch)
    ls configs/build/*.yaml > logs/slurm/build_list.txt
    NB=$(grep -c . logs/slurm/build_list.txt)
    B=$(sub build --dependency=afterok:$V --array=0-$((NB - 1))%4 \
        --export=ALL,BUILD_LIST=logs/slurm/build_list.txt slurm/campaigns/build_states.sbatch)
    M=$(sub manifests --dependency=afterok:$B -J pj-manifests --time=00:10:00 \
        --export=ALL,CMD="sha256sum -c data/manifests/EXPECTED.sha256" slurm/campaigns/cpu_task.sbatch)
    F=$(sub p0fetch --export=ALL,SKIP_MODELS=1 slurm/campaigns/phase0_fetch.sbatch)
    R=$(sub p0repro --dependency=afterok:$F:$P slurm/campaigns/phase0_repro.sbatch)
    E=$(sub sets --dependency=afterok:$M -J pj-sets --time=01:00:00 \
        --export=ALL,CMD="python -m pathjev.mix --config configs/train/smoke.yaml && python -m pathjev.evalsets --config configs/eval/smoke.yaml && python -m pathjev.evalsets --config configs/eval/pilot-stage1.yaml" \
        slurm/campaigns/cpu_task.sbatch)
    sub smoke-jev --dependency=afterok:$E:$P -J pj-smoke-jev --time=02:00:00 \
        --export=ALL,MIX=smoke,RUN_NAME=smoke-adhoc,SEED=1,SMOKE=1 slurm/campaigns/train_jev.sbatch >/dev/null
    sub smoke-sft --dependency=afterok:$E:$P -J pj-smoke-sft --time=02:00:00 \
        --export=ALL,MIX=smoke,RUN_NAME=smoke-sft-adhoc,SEED=1,MAX_STEPS=4 slurm/campaigns/train_sft.sbatch >/dev/null
    for arm in omnimedjev conch letter features; do
        sub smoke-$arm --dependency=afterok:$E:$P -J pj-smoke-$arm --time=02:00:00 \
            --export=ALL,ARM=$arm,NAME=smoke-$arm,EVAL_SET=$W/eval_sets/smoke,RESAMPLES=200 \
            slurm/campaigns/eval_arm.sbatch >/dev/null
    done
    ;;
pilot)
    SET="$W/eval_sets/pilot-stage1"
    X=$(sub mix -J pj-mix --time=01:00:00 \
        --export=ALL,CMD="python -m pathjev.mix --config configs/train/pilot-p.yaml" slurm/campaigns/cpu_task.sbatch)
    TJ=$(sub train-jev --dependency=afterok:$X -J pj-pilot-jev \
         --export=ALL,MIX=pilot-p,RUN_NAME=pilot-p-s1,SEED=1,EPOCHS=1,SAVE_STEPS=100 slurm/campaigns/train_jev.sbatch)
    TS=$(sub train-sft --dependency=afterok:$X -J pj-pilot-sft \
         --export=ALL,MIX=pilot-p,RUN_NAME=pilot-sft-p-s1,SEED=1,EPOCHS=1,MAX_STEPS_FROM_MIX=1,SAVE_STEPS=100 \
         slurm/campaigns/train_sft.sbatch)
    sub post-jev --dependency=afterok:$TJ \
        --export=ALL,ARM=jev,RUN_NAME=pilot-p-s1,MIX=pilot-p,EVAL_SET=$SET slurm/campaigns/post_train.sbatch >/dev/null
    sub post-sft --dependency=afterok:$TS \
        --export=ALL,ARM=sft,RUN_NAME=pilot-sft-p-s1,MIX=pilot-p,EVAL_SET=$SET slurm/campaigns/post_train.sbatch >/dev/null
    for arm in omnimedjev conch letter; do
        sub zs-$arm -J pj-pilot-$arm --export=ALL,ARM=$arm,NAME=pilot-$arm,EVAL_SET=$SET slurm/campaigns/eval_arm.sbatch >/dev/null
    done
    VA=$(sub variants -J pj-variants --time=00:30:00 \
         --export=ALL,CMD="python -m pathjev.variants --src $SET" slurm/campaigns/cpu_task.sbatch)
    for v in no-other permuted; do   # stage1_analysis_plan §6.4: baseline (a) diagnostics
        sub zs-omnimedjev-$v --dependency=afterok:$VA -J pj-pilot-omnimedjev-$v \
            --export=ALL,ARM=omnimedjev,NAME=pilot-omnimedjev-$v,EVAL_SET=$SET-$v slurm/campaigns/eval_arm.sbatch >/dev/null
    done
    FE=$(sub features --dependency=afterok:$X -J pj-pilot-feat \
         --export=ALL,ARM=features,NAME=pilot-features,FILES="$W/mixes/pilot-p/train.jsonl $(printf "$SET/%s.jsonl " bracs sicapv2 hepatobench hiesd spider-colorectal spider-thorax spider-breast spider-skin)" \
         slurm/campaigns/eval_arm.sbatch)
    FD="$W/runs/eval_pilot-features"; PD="$W/runs/eval_pilot-probe"
    sub probe --dependency=afterok:$FE -J pj-pilot-probe --cpus-per-task=16 --mem=64G --time=06:00:00 \
        --export=ALL,CMD="mkdir -p $PD && for f in $SET/*.jsonl; do n=\$(basename \$f .jsonl); python -m pathjev.baselines probe --train-states $W/mixes/pilot-p/train.jsonl --train-feats $FD/train.npz --test-states \$f --test-feats $FD/\$n.npz --out $PD/\${n}_decisions.jsonl && python -m pathjev.evaluate --states \$f --decisions $PD/\${n}_decisions.jsonl --out $PD/\${n}_eval.json > $PD/\${n}_eval.log; done" \
        slurm/campaigns/cpu_task.sbatch >/dev/null
    ;;
*) echo "usage: $0 validate|pilot [--dry-run]"; exit 2 ;;
esac
echo "submitted ($MODE); job ids in $LOG"
