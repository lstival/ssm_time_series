#!/bin/bash
#SBATCH --comment=lotsa_optimized_v2
#SBATCH --time=4320
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/lotsa_optimized_v2/train_lotsa_optimized_v2_%j.out
#SBATCH --error=logs/lotsa_optimized_v2/train_lotsa_optimized_v2_%j.err
#SBATCH --job-name=lotsa_opt_v2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/lotsa_optimized_v2

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/cosine_training.py \
    --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_optimized_v2.yaml

# Job: ICML v2 encoder training with ablation optimizations
#
# Optimizations from ablation studies:
# 1. RP Strategy: joint (25% faster, same accuracy)
#    - Ablation A finding
#    - Expected: ~297ms per batch (vs ~395ms baseline)
#    - 50 epochs at 200 batches/epoch = ~2.8 GPU hours saved
#
# 2. Config: lotsa_optimized_v2.yaml
#    - Clean separation from base configs
#    - Ready for alignment loss improvements
#
# Next steps after completion:
# 1. Monitor training speed vs baseline (compare to train_lotsa_CLIP job)
# 2. Train forecast head with this encoder
# 3. Evaluate on ICML datasets
#
# Dependency usage:
# sbatch src/scripts/anunna_train_lotsa_optimized_v2.sh
