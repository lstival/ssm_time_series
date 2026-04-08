#!/bin/bash
#SBATCH --comment=probe_lotsa_66260794
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/probe_lotsa/probe_lotsa_66260794_%j.out
#SBATCH --error=logs/probe_lotsa/probe_lotsa_66260794_%j.err
#SBATCH --job-name=probe_66260794
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/probe_lotsa

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINTS}/ts_encoder_lotsa_20260407_2309" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/probe_lotsa_66260794 \
    --scaler_type standard \
    --seed 42
