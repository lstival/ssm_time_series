#!/bin/bash
#SBATCH --comment=ablation_A_icml_overview
#SBATCH --time=240
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/ablation_A_icml_%j.out
#SBATCH --error=logs/smoke/ablation_A_icml_%j.err
#SBATCH --job-name=ablA_icml
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

echo "Start: $(date)"

time python3 "${SRC}/experiments/ablation_A_mv_rp.py" \
    --config       "${SRC}/configs/ablations/ablation_A_icml_overview.yaml" \
    --train_epochs 20 \
    --probe_epochs 20 \
    --pretrain_data_dir "${REPO}/ICML_datasets/ETT-small" \
    --data_dir     "${REPO}/ICML_datasets" \
    --strategies   per_channel mean pca joint \
    --results_dir  "${REPO}/results/ablation_A_icml_overview" \
    --seed 42

echo "End: $(date)"
