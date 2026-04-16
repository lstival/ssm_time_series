#!/bin/bash
#SBATCH --comment=simclr_nano_probe_seq336
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/probe_lotsa/simclr_nano_seq336_%j.out
#SBATCH --error=logs/probe_lotsa/simclr_nano_seq336_%j.err
#SBATCH --job-name=simclr_n336
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/probe_lotsa

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/simclr_bimodal_nano
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_simclr_bimodal_nano_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No SimCLR bimodal nano checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing SimCLR-nano (seq_len=336) checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_simclr_bimodal_nano.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/simclr_nano_seq336 \
    --scaler_type standard \
    --seq_len 336 \
    --embed_batch_size 8 \
    --seed 42
