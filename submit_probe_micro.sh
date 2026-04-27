#!/bin/bash
#SBATCH --job-name=probe_micro
#SBATCH --output=/home/WUR/stiva001/WUR/ssm_time_series/logs/probe_lotsa/simclr_micro_lotsa_%j.out
#SBATCH --err=/home/WUR/stiva001/WUR/ssm_time_series/logs/probe_lotsa/simclr_micro_lotsa_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC="/home/WUR/stiva001/WUR/ssm_time_series/src"
CHECKPOINT_DIR="/home/WUR/stiva001/WUR/ssm_time_series/checkpoints/simclr_full/ts_simclr_full_lotsa_20260414_002946"

export PYTHONPATH=$PYTHONPATH:${SRC}

# Run probing for Micro model
python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_simclr_full.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/simclr_micro_lotsa_best \
    --scaler_type standard \
    --seq_len 336 \
    --embed_batch_size 32 \
    --seed 42
