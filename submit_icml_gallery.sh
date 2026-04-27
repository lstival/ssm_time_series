#!/bin/bash
#SBATCH --job-name=icml_gallery
#SBATCH --output=/home/WUR/stiva001/WUR/ssm_time_series/logs/icml_gallery_%j.out
#SBATCH --err=/home/WUR/stiva001/WUR/ssm_time_series/logs/icml_gallery_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC="/home/WUR/stiva001/WUR/ssm_time_series/src"
export PYTHONPATH=$PYTHONPATH:${SRC}

python3 "${SRC}/analysis/icml_viz_gallery.py"
