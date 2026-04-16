#!/bin/bash
#SBATCH --comment=icml_vl_jepa_micro_probe
#SBATCH --time=360
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icml_vl_jepa_micro/probe_psn_%j.out
#SBATCH --error=logs/icml_vl_jepa_micro/probe_psn_%j.err
#SBATCH --job-name=icml_jepa_mp_psn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/icml_vl_jepa_micro

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/icml_vl_jepa_micro
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_vl_jepa_micro_icml_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No VL-JEPA micro ICML checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing VL-JEPA micro ICML checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/icml_vl_jepa_micro.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/icml_vl_jepa_micro_psn_full \
    --scaler_type standard \
    --per_series_norm \
    --embed_batch_size 16 \
    --datasets ETTm1.csv ETTm2.csv ETTh1.csv ETTh2.csv weather.csv traffic.csv electricity.csv exchange_rate.csv \
    --seq_len 336 \
    --seed 42
