#!/bin/bash
# GPU runner for non-Jev baselines and scorers, writing upstream-format decision files.
#   KIND=conch    STATES=... OUT=...                      (c) CONCH v1 zero-shot
#   KIND=features STATES=... OUT=....npz                  H-optimus-0 features for (d)
#   KIND=letter   STATES=... OUT=... [ADAPTER=]           (e) Inner-Jev letter read-out (base model)
#   KIND=text     STATES=... OUT=... ADAPTER=<sft run>    (b) generative SFT, candidate log-likelihood
#SBATCH --job-name=pj-baseline
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
cd "$REPO"
source slurm/templates/common.sh
pathjev_log_provenance "${STATES:?}" none none
mkdir -p "$(dirname "${OUT:?}")"
pathjev_snapshot_code "$(dirname "$OUT")/code_${SLURM_JOB_ID}"
export PYTHONPATH="$REPO/src/shims:$REPO/src:$REPO/external/OmniMed-Jev"
case "${KIND:?}" in
  conch)    python -u -m pathjev.baselines conch --states "$STATES" --out "$OUT" ${LIMIT:+--limit $LIMIT} ;;
  features) python -u -m pathjev.baselines features --states "$STATES" --out "$OUT" ;;
  letter|text) python -u -m pathjev.generative --states "$STATES" --out "$OUT" --readout "$KIND" \
                 --model "$PATHJEV_WORK/models/medgemma-1.5-4b-it" ${ADAPTER:+--adapter "$ADAPTER"} \
                 --image-size "${IMAGE_SIZE:-896}" ${LIMIT:+--limit $LIMIT} ;;
  *) echo "unknown KIND=$KIND"; exit 2 ;;
esac
