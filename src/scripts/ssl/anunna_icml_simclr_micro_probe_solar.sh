#!/bin/bash
#SBATCH --comment=icml_simclr_micro_probe_solar
#SBATCH --time=180
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icml_simclr_micro/probe_solar_%j.out
#SBATCH --error=logs/icml_simclr_micro/probe_solar_%j.err
#SBATCH --job-name=simclr_m_sol
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/icml_simclr_micro

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/icml_simclr_micro
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_simclr_micro_icml_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No simclr micro ICML checkpoint found in ${CHECKPOINTS}"; exit 1
fi
echo "Probing [solar] simclr micro: ${CHECKPOINT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/icml_simclr_micro.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --datasets solar_AL.txt \
    --probe_epochs 20 \
    --batch_size 1 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/icml_simclr_micro_solar \
    --scaler_type standard \
    --seq_len 336 \
    --embed_batch_size 16 \
    --no_comet \
    --seed 42
