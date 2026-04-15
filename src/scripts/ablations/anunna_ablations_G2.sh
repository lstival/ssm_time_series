#!/bin/bash
#SBATCH --comment=ablation_G2_complementarity
#SBATCH --time=240
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_G2/ablation_G2_%j.out
#SBATCH --error=logs/ablation_G2/ablation_G2_%j.err
#SBATCH --job-name=ablation_G2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_G2

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/home/WUR/stiva001/WUR/ssm_time_series/checkpoints

CHECKPOINT_DIR="${CHECKPOINTS}/ts_encoder_lotsa_20260330_2251"

echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/ablation_G2_complementarity.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_G2 \
    --seq_len 336 \
    --embed_batch_size 16 \
    --no_comet \
    --seed 42

echo "End: $(date)"

# sbatch src/scripts/anunna_ablations_G2.sh
