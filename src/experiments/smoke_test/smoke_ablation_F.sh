#!/bin/bash
#SBATCH --comment=smoke_ablation_F
#SBATCH --time=60
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/ablation_F_%j.out
#SBATCH --error=logs/smoke/ablation_F_%j.err
#SBATCH --job-name=smoke_ablF
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/smoke

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
REPO=/home/WUR/stiva001/WUR/ssm_time_series

# Uses checkpoint saved by ablation A (per_channel strategy)
CHECKPOINT_DIR="${REPO}/results/smoke/ablation_A/strategy_per_channel"

echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/ablation_F_manifold.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/experiments/smoke_test/smoke_config.yaml" \
    --data_dir "${REPO}/ICML_datasets" \
    --results_dir "${REPO}/results/smoke/ablation_F" \
    --seed 42

echo "End: $(date)"
