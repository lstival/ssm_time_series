#!/bin/bash
#SBATCH --comment=install_mamba
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/install/install_%j.out
#SBATCH --error=logs/install/install_%j.err
#SBATCH --job-name=install_mamba
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs/install

# Load modules
module load GPU
module load CUDA/12.4.0

# Print info
which nvcc
nvcc --version

export PATH=/home/WUR/stiva001/WUR/timeseries/bin:$PATH
which python
python --version

# Downgrade torch/numpy to match CUDA 12.4
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install packaging numpy setuptools wheel --upgrade

# Install with no build isolation to use correct torch
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
