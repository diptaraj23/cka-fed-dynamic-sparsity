#!/bin/bash -l
#SBATCH --job-name=cifar10_cka
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-cifar10-cka-%j.out
#SBATCH --error=slurm-cifar10-cka-%j.err

set -u

cd /mnt/aiongpfs/users/dsen/cka-fed-dynamic-sparsity

eval "$(micromamba shell hook --shell bash)"
micromamba activate cka-cifar
export PYTHONNOUSERSITE=1

echo "Host: $(hostname)"
echo "Started: $(date)"
python --version

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Torch file:", torch.__file__)
print("CUDA build:", torch.version.cuda)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")

x = torch.randn(8, 3, 32, 32, device="cuda")
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(x)
torch.cuda.synchronize()
print("CUDA conv smoke passed:", tuple(y.shape))
PY

python experiments/run_experiment.py \
  --dataset cifar10 \
  --suite cka_strength \
  --suite-id cifar10_cka_strength_fixed_${SLURM_JOB_ID} \
  --continue_on_error

echo "Finished cka_strength: $(date)"