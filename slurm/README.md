# Slurm scripts

The experiments ran with Slurm. The scripts in `campaigns/` are the job scripts actually used. To run them on another cluster:

1. Copy `cluster.env.example` to `cluster.env`. Set your account, QoS and the paths `PATHJEV_WORK` (heavy outputs, about 1 TB for all runs), `PATHJEV_DATA` (raw datasets) and `PATHJEV_VENV` (the Python environment).
2. Partitions. Every script requests `--partition=gpu` or `--partition=cpu`. Edit these to your partition names, or override them on the command line (`sbatch --partition=... `).
3. GPU types. Some scripts pin `--gres=gpu:h100:1`, the GPU type of the paper's runs. Change the type to match your cluster. Gen-SFT scoring is sharded over the GPUs of a node (`PJ_GRES_SFT`, for example `gpu:h100:2`).
4. Submit from the repository root: `source slurm/cluster.env && sbatch slurm/campaigns/<job>.sbatch`.

`templates/common.sh` is sourced by every job. It activates the environment, sets the cache directories and logs provenance: code hash, configuration hash, split-manifest hash, seed, node and GPU. It also appends a line to `results/registry.jsonl` and stores a snapshot of the code next to each run.

Stage 1 is driven by `campaigns/stage1_chain.sh` (configuration grid → selection → seeds) and `campaigns/stage1_phase4.sh` (test and baseline jobs). The post-hoc analyses use `campaigns/a3_*.sh`, `campaigns/stage1_regroup.sh` and `campaigns/dedup_audit.sh`. Several scripts take their command through `--export=ALL,CMD="bash <script>"`. Never put commas inside an `--export` value: Slurm splits on them.
