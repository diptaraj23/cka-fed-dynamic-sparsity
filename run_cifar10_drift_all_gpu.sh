#!/bin/bash -l
#SBATCH --job-name=cifar10-drift-all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-cifar10-drift-all-%j.out
#SBATCH --error=slurm-cifar10-drift-all-%j.err

set -uo pipefail

cd "$HOME/cka-fed-dynamic-sparsity"

source "$HOME/micromamba/etc/profile.d/micromamba.sh"
micromamba activate cka-cifar

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Branch:"
git branch --show-current
git log --oneline -5

echo "CUDA preflight:"
python - <<'PY'
import torch

torch.backends.cudnn.enabled = False

print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this Slurm job.")

x = torch.randn(8, 3, 32, 32, device="cuda")
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(x)
torch.cuda.synchronize()
print("CUDA conv smoke passed:", tuple(y.shape))
PY

if [ $? -ne 0 ]; then
  echo "CUDA preflight failed. Exiting."
  exit 1
fi

echo "Dry-run check: CIFAR-10 drift all suite"
python experiments/run_experiment.py \
  --dataset cifar10 \
  --suite all \
  --methods cka_feddst \
  --cka-signal drift \
  --sparsities 0.5 0.7 0.8 0.9 0.95 \
  --suite-id cifar10_drift_all \
  --dry_run

if [ $? -ne 0 ]; then
  echo "Dry-run failed. Exiting."
  exit 1
fi

echo "Launching CIFAR-10 drift all suite..."
python experiments/run_experiment.py \
  --dataset cifar10 \
  --suite all \
  --methods cka_feddst \
  --cka-signal drift \
  --sparsities 0.5 0.7 0.8 0.9 0.95 \
  --suite-id cifar10_drift_all \
  --continue_on_error

RUN_EXIT=$?

echo "Aggregating multiseed drift results..."
python experiments/aggregate_results.py \
  --suite multiseed \
  --log_dir results/logs/multiseed/cifar10_drift_all \
  --output_dir results/averaged/multiseed/cifar10_drift_all

echo "Aggregating CKA-strength drift results..."
python experiments/aggregate_results.py \
  --suite cka_strength \
  --log_dir results/logs/cka_strength_sweep/cifar10_drift_all \
  --output_dir results/averaged/cka_strength_sweep/cifar10_drift_all

echo "Generating multiseed drift plots..."
python experiments/plot_results.py \
  --avg_dir results/averaged/multiseed/cifar10_drift_all \
  --plot_dir results/plots/multiseed/cifar10_drift_all

echo "Generating CKA-strength drift plots..."
python experiments/plot_results.py \
  --avg_dir results/averaged/cka_strength_sweep/cifar10_drift_all \
  --plot_dir results/plots/cka_strength_sweep/cifar10_drift_all

echo "Job finished at: $(date)"
exit "$RUN_EXIT"