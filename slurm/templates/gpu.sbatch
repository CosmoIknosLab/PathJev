#!/bin/bash
# PathJev GPU job template. Copy into slurm/campaigns/ and edit the lines marked EDIT.
# Submit from the repository root so relative paths resolve, after `source slurm/cluster.env`
# (account, QoS and other cluster-wide settings; see slurm/README.md).
#
# The experiments of the paper ran on nodes with 2 x H100 80 GB; memory must always be requested.
#
#SBATCH --job-name=pathjev-EDIT
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1                       # gpu:h100:1 or gpu:l40:1 to pin the type
#SBATCH --cpus-per-task=8                  # EDIT
#SBATCH --mem=48G                          # EDIT
#SBATCH --time=04:00:00                    # EDIT: pilot measurement + 20-30%
#SBATCH --requeue
#SBATCH --signal=B:USR1@300                # 5 min warning before the time limit

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$REPO"
source slurm/templates/common.sh           # env, caches, provenance logging
pathjev_log_provenance "${CONFIG:-none}" "${SPLIT_MANIFEST:-none}" "${SEED:-none}"

# Forward the time-limit warning to the Python process so it can checkpoint.
trap 'echo "[slurm] USR1 received: time limit near"; kill -USR1 "$child" 2>/dev/null || true' USR1

# EDIT: the actual work. Keep heavy outputs under $PATHJEV_WORK, not the repo.
python -m EDIT.module --config "${CONFIG:?}" --out "$PATHJEV_WORK/runs/${SLURM_JOB_ID}" &
child=$!
wait "$child"
