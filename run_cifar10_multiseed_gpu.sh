#!/bin/bash -l
#SBATCH --job-name=cifar10_multiseed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-cifar10-multiseed-%j.out
#SBATCH --error=slurm-cifar10-multiseed-%j.err

set -u

cd /mnt/aiongpfs/users/dsen/cka-fed-dynamic-sparsity

eval "$(micromamba shell hook --shell bash)"
micromamba activate cka-cifar

echo "Host: $(hostname)"
echo "Started: $(date)"
python --version
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

python experiments/run_experiment.py \
  --dataset cifar10 \
  --suite multiseed \
  --continue_on_error

echo "Finished multiseed: $(date)"