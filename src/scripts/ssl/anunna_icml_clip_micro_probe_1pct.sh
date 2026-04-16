#!/bin/bash
#SBATCH --comment=icml_clip_micro_probe_1pct
#SBATCH --time=360
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icml_clip_micro/probe_1pct_%j.out
#SBATCH --error=logs/icml_clip_micro/probe_1pct_%j.err
#SBATCH --job-name=icml_clip_1pct
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/icml_clip_micro

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/icml_clip_micro
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_clip_micro_icml_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No CLIP micro ICML checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Few-shot 1% probe — CLIP micro ICML checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/icml_clip_micro.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/icml_clip_micro_1pct \
    --scaler_type standard \
    --per_series_norm \
    --embed_batch_size 16 \
    --datasets ETTm1.csv ETTm2.csv ETTh1.csv ETTh2.csv weather.csv traffic.csv electricity.csv exchange_rate.csv \
    --seq_len 336 \
    --few_shot_fraction 0.01 \
    --seed 42
