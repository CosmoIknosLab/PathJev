#!/bin/bash
# First job on a new cluster: GPU/driver, CUDA from torch, 4-bit bitsandbytes matmul,
# outbound network from the compute node, and the offline model cache.
#   source slurm/cluster.env && sbatch slurm/campaigns/gpu_probe.sbatch
#SBATCH --job-name=pj-gpu-probe
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu                        # set to your GPU partition (slurm/README.md)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/templates/common.sh
pathjev_log_provenance none none none
pathjev_register gpu_probe none none none "" "ENGINEERING CHECK: driver, CUDA, bitsandbytes 4-bit, network, model cache on ADHOC"
nvidia-smi
python - <<'PY'
import torch, bitsandbytes as bnb, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
lin = bnb.nn.Linear4bit(256, 128, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
print("4-bit matmul ok", tuple(lin(x).shape))
print("transformers", transformers.__version__, "peft", peft.__version__, "bnb", bnb.__version__)
PY
for url in https://huggingface.co https://zenodo.org https://github.com; do
    printf '%s ' "$url"; /usr/bin/curl -s -o /dev/null -m 15 -w '%{http_code}\n' "$url" || echo unreachable
done
ls "$PATHJEV_WORK/models" "$HF_HOME/hub"
