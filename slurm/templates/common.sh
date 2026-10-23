# Sourced by every PathJev sbatch script. Sets up the environment and logs the
# provenance fields required by CLAUDE.md §5/§7.
#
# The project does not use git (D-003). Code provenance is a content hash of the
# project's code directories plus a tarball snapshot of them stored with the run.

PATHJEV_CODE_DIRS=(src configs slurm analysis tests)

# Software environment: project virtual environment (override with PATHJEV_VENV).
export PATHJEV_VENV="${PATHJEV_VENV:-$PWD/.venv}"   # override only for diagnostics (D-007)
source "$PATHJEV_VENV/bin/activate"

# Heavy outputs (checkpoints, caches, predictions) live outside the repository.
export PATHJEV_WORK="${PATHJEV_WORK:-$HOME/pathjev/work}"
export PATHJEV_DATA="${PATHJEV_DATA:-$HOME/pathjev/data}"
# HF cache (models + the user's HF token, written by `huggingface-cli login`).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
# Models are local copies with pinned revisions; jobs never contact the Hub unless a
# fetch job sets HF_HUB_OFFLINE=0 explicitly.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$PATHJEV_WORK"

pathjev_sha256() { [ -f "$1" ] && sha256sum "$1" | cut -c1-16 || echo "none"; }

# Deterministic hash of every file under the code directories (path + content).
pathjev_code_hash() {
    local dirs=()
    for d in "${PATHJEV_CODE_DIRS[@]}"; do [ -d "$d" ] && dirs+=("$d"); done
    find "${dirs[@]}" -type f -not -name '*.pyc' -not -path '*/__pycache__/*' -print0 \
        | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16
}

# Writes the code snapshot next to the run outputs; call once the run dir exists.
pathjev_snapshot_code() {
    local run_dir="$1"
    mkdir -p "$run_dir"
    tar --exclude='__pycache__' -czf "$run_dir/code_snapshot.tar.gz" \
        $(for d in "${PATHJEV_CODE_DIRS[@]}"; do [ -d "$d" ] && echo "$d"; done)
    echo "$(pathjev_code_hash)" > "$run_dir/code_hash.txt"
}

pathjev_log_provenance() {
    local config="$1" manifest="$2" seed="$3"
    echo "=== PathJev job provenance"
    echo "job_id:         ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
    echo "node:           $(hostname)"
    echo "start:          $(date -Is)"
    echo "restart_count:  ${SLURM_RESTART_COUNT:-0}"
    echo "gpus:           ${CUDA_VISIBLE_DEVICES:-none}"
    echo "code_hash:      $(pathjev_code_hash)"
    # git is not installed on ADHOC compute nodes: the clone is a detached HEAD, read it directly.
    [ -f external/OmniMed-Jev/.git/HEAD ] && echo "upstream:       OmniMed-Jev $(cat external/OmniMed-Jev/.git/HEAD)"
    echo "config:         $config (sha256:$(pathjev_sha256 "$config"))"
    echo "split_manifest: $manifest (sha256:$(pathjev_sha256 "$manifest"))"
    echo "seed:           $seed"
    echo "python:         $(python --version 2>&1)"
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null; then
        nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    fi
    echo "==="
}

# Appends one line to results/registry.jsonl when the job ends (CLAUDE.md §7).
# Usage, right after pathjev_log_provenance:
#   pathjev_register <experiment> <config> <seed> <split_manifest_or_mix> <metrics_path> "<notes>"
pathjev_register() {
    export PJ_REG_EXP="$1" PJ_REG_CFG="$2" PJ_REG_SEED="$3" PJ_REG_MANIFEST="$4" PJ_REG_METRICS="$5" PJ_REG_NOTES="$6"
    export PJ_REG_START="$(date -Is)" PJ_REG_CODE="$(pathjev_code_hash)" PJ_REG_FILE="$PWD/results/registry.jsonl"
    export PJ_REG_CFG_HASH="$(pathjev_sha256 "$2")" PJ_REG_MAN_HASH="$(pathjev_sha256 "$4")"
    trap 'pathjev__register_end $?' EXIT
}

pathjev__register_end() {
    local rc="$1"
    PJ_REG_RC="$rc" python - <<'PY' || true
import fcntl, json, os, time
e = os.environ
job = e.get("SLURM_JOB_ID", "local") + (f"_{e['SLURM_ARRAY_TASK_ID']}" if e.get("SLURM_ARRAY_TASK_ID") else "")
row = {"run_id": f"{e['PJ_REG_EXP']}_{job}", "experiment": e["PJ_REG_EXP"], "config": e["PJ_REG_CFG"],
       "config_hash": e["PJ_REG_CFG_HASH"], "code_hash": e["PJ_REG_CODE"], "git_commit": None,
       "seed": e["PJ_REG_SEED"], "slurm_job_id": job, "cluster": "adhoc",
       "status": "completed" if e["PJ_REG_RC"] == "0" else "failed",
       "split_manifest_hash": e["PJ_REG_MAN_HASH"], "start": e["PJ_REG_START"],
       "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "metrics_path": e["PJ_REG_METRICS"] or None,
       "notes": e["PJ_REG_NOTES"]}
with open(e["PJ_REG_FILE"], "a") as handle:   # absolute: jobs may cd into the upstream clone
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.write(json.dumps(row) + "\n")
PY
    return "$rc"
}
