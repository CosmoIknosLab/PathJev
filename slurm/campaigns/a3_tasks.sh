#!/bin/bash
# Write the A3 task lists for slurm/campaigns/a3_eval.sbatch (plan A3.2–A3.5, D-040):
#   $PATHJEV_WORK/checks/a3/tasks_blank.txt   A3.2 (also the throughput pilot)
#   $PATHJEV_WORK/checks/a3/tasks_bulk.txt    A3.3, A3.4, A3.5
#   bash slurm/campaigns/a3_tasks.sh
set -euo pipefail
W="${PATHJEV_WORK:?}"
OUT="$W/checks/a3"; mkdir -p "$OUT"
ES="$W/eval_sets"
CLEAN="bracs sicapv2 hepatobench hiesd"
ALL="bracs sicapv2 hepatobench hiesd spider-colorectal spider-thorax spider-breast spider-skin"
OMJ="$W/models/OmniMed-Jev-4B"
jev() { cat "$W/runs/train_s1-jev-lr2e-4-e3-s$1/selection/best.txt"; }
sft() { cat "$W/runs/train_s1-sft-lr2e-4-e5-s$1/selection/best.txt"; }

{
  echo "jev blank-jev-s1 $(jev 1) $ES/a3-blank 896 all $CLEAN"
  echo "sft blank-sft-s1 $(sft 1) $ES/a3-blank 896 main $CLEAN"
  echo "omnimedjev blank-omnimedjev $OMJ $ES/a3-blank 448 all $CLEAN"
} > "$OUT/tasks_blank.txt"

{
  echo "omnimedjev omnimedjev-896 $OMJ $ES/stage1-test 896 all $ALL"                   # A3.3
  for s in 1 2 3; do echo "sft sftvar-s$s $(sft $s) $ES/stage1-test 896 variants $CLEAN"; done   # A3.4
  for k in 1 2 3 4 5; do echo "sft perm$k-sft-s1 $(sft 1) $ES/a3-perm-$k 896 main $CLEAN"; done  # A3.5
  for k in $(seq 1 10); do
    for s in 1 2 3; do echo "jev perm$k-jev-s$s $(jev $s) $ES/a3-perm-$k 896 all $CLEAN"; done
    echo "omnimedjev perm$k-omnimedjev $OMJ $ES/a3-perm-$k 448 all $CLEAN"
  done
} > "$OUT/tasks_bulk.txt"
wc -l "$OUT"/tasks_*.txt
