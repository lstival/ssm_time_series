#!/bin/bash
#SBATCH --comment=plot_triangu_vim_pipeline
#SBATCH --time=120
#SBATCH --mem=16000
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/plots/triangu_vim_pipeline_%j.out
#SBATCH --error=logs/plots/triangu_vim_pipeline_%j.err
#SBATCH --job-name=plot_triangu_vim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl

set -euo pipefail

REPO=/home/WUR/stiva001/WUR/ssm_time_series
mkdir -p "${REPO}/logs/plots"

module load GPU || true
source /home/WUR/stiva001/WUR/timeseries/bin/activate

cd "${REPO}"

python3 src/analysis/plot_triangu_vim_pipeline.py
