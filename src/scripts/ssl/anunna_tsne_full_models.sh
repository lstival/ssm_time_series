#!/bin/bash
#SBATCH --comment=tsne_full_models_icml
#SBATCH --time=120
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/tsne/tsne_full_models_%j.out
#SBATCH --error=logs/tsne/tsne_full_models_%j.err
#SBATCH --job-name=tsne_full
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/tsne
mkdir -p results/tsne_full_models

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

echo "============================================================"
echo "t-SNE: Full-tier encoders on 7 ICML benchmark datasets"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/tsne_full_models_icml.py \
  --icml_data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir   /home/WUR/stiva001/WUR/ssm_time_series/results/tsne_full_models \
  --samples_per_ds 300 \
  --seed 42

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
