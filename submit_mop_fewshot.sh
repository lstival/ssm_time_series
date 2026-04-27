#!/bin/bash
#SBATCH --job-name=mop_fewshot
#SBATCH --output=/home/WUR/stiva001/WUR/ssm_time_series/logs/mop_fewshot_%j.out
#SBATCH --err=/home/WUR/stiva001/WUR/ssm_time_series/logs/mop_fewshot_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/WUR/stiva001/WUR/ssm_time_series/src

# Run Few-Shot (5%) on ETTh1
python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_fewshot_probe.py \
    --base_checkpoint_dir /home/WUR/stiva001/WUR/ssm_time_series/checkpoints/simclr_bimodal_nano/ts_simclr_bimodal_nano_lotsa_20260409_182831 \
    --mop_checkpoint /home/WUR/stiva001/WUR/ssm_time_series/results/mop_grid_v1/revin_mlp/mop_flex.pt \
    --few_shot_fraction 0.05 \
    --finetune_epochs 10 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_fewshot_test
