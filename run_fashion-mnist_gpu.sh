#!/bin/bash -l
#SBATCH --job-name=fashion_fl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /mnt/aiongpfs/users/dsen/cka-fed-dynamic-sparsity

module load lang/Python/3.11.5-GCCcore-13.2.0
source .venv/bin/activate

python experiments/run_experiment.py --dataset fashion_mnist --suite all