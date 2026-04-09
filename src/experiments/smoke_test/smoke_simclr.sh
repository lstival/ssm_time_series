#!/bin/bash
#SBATCH --comment=smoke_simclr
#SBATCH --time=30
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/simclr_%j.out
#SBATCH --error=logs/smoke/simclr_%j.err
#SBATCH --job-name=smoke_simclr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/smoke

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
REPO=/home/WUR/stiva001/WUR/ssm_time_series

echo "Start: $(date)"
echo "Testing both SimCLR modes: temporal and visual"
echo "  - 3 epochs, reduced model (model_dim=64, depth=2)"
echo "  - Single dataset: ETTm1.csv"
echo ""

time python3 "${SRC}/experiments/smoke_test/smoke_simclr.py" \
    --config "${SRC}/experiments/smoke_test/smoke_config.yaml" \
    --data_dir "${REPO}/ICML_datasets/ETT-small" \
    --epochs 3 \
    --modes temporal visual \
    --seed 42

EXIT_CODE=$?

echo ""
echo "End: $(date)"
echo "Smoke test exited with code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[PASS] SimCLR smoke test succeeded — safe to submit full training jobs."
else
    echo "[FAIL] SimCLR smoke test failed — check logs above before full submission."
fi

exit ${EXIT_CODE}
