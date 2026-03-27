#!/bin/bash
#SBATCH --comment=exp1_contrastive_ablation
#SBATCH --time=4320
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=exp1_contrastive_ablation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

module load GPU

time srun conda run -n timeseries python /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/exp1_contrastive_ablation.py --epochs 20

# how to call the job
# sbatch src/run_exp1_contrastive_ablation.sh
