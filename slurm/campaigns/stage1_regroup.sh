#!/bin/bash
# Second-review corrections (2026-10-23), CPU only, through cpu_task.sbatch:
#   source slurm/cluster.env && sbatch --job-name=pj-s1-regroup --cpus-per-task=8 --mem=48G --time=04:00:00 \
#       --export=ALL,CMD="bash slurm/campaigns/stage1_regroup.sh" slurm/campaigns/cpu_task.sbatch
# HiESD groups = patients (official map, plan §3) via pathjev.groups; SPIDER-skin duplicates of training images excluded.
# 1) probe re-cross-fitted on HiESD (capped + full) with patient groups in outer and inner folds;
# 2) final analysis exactly as job 7270 (10,000 resamples) + temperature transfer + macros + tables;
# 3) data statement / test-set table, secondary metrics, A3.1 (HepatoBench blocks and no-HepatoBench).
set -euo pipefail
source slurm/templates/common.sh
W="$PATHJEV_WORK"
pathjev_register "stage1_regroup" docs/stage1_analysis_plan.md none data/manifests/hiesd_patient_slide_mapping.csv \
    results/runs/stage1-analysis/summary.json "HiESD patient groups + duplicate exclusions (second review)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for v in "stage1-test eval_s1-probe eval_s1-features" "stage1-test-full eval_s1-probe-full eval_s1-features-full"; do
    set -- $v
    python -u -m pathjev.baselines probe --train-states "$W/mixes/stage1-p/train.jsonl" \
        --train-feats "$W/runs/eval_s1-features/train.npz" --test-states "$W/eval_sets/$1/hiesd.jsonl" \
        --test-feats "$W/runs/$3/hiesd.npz" --out "$W/runs/$2/hiesd_decisions.jsonl"
    python -u -m pathjev.evaluate --states "$W/eval_sets/$1/hiesd.jsonl" --decisions "$W/runs/$2/hiesd_decisions.jsonl" \
        --out "$W/runs/$2/hiesd_eval.json" > "$W/runs/$2/hiesd_eval.log"
done
python -u analysis/stage1_analysis.py --work "$W" --out results/runs/stage1-analysis --resamples 10000
python -u analysis/stage1_temperature_transfer.py --work "$W" --out results/runs/stage1-temperature
python -u analysis/export_macros.py --work "$W" --out paper/generated/numbers.tex \
    --analysis results/runs/stage1-analysis/summary.json --temperature results/runs/stage1-temperature/summary.json
python -u analysis/paper_tables.py --analysis results/runs/stage1-analysis/summary.json \
    --temperature results/runs/stage1-temperature/summary.json --wsi "$W/wsi_demo/demo-hiesd" --out paper/generated
python -u analysis/data_statement.py --work "$W"
(cd analysis && python -u secondary_metrics.py --work "$W")
(cd analysis && python -u a3_hepato.py --work "$W" --out ../results/runs/stage1-a3-hepato --resamples 10000 --procs 4)
echo "regroup done"
