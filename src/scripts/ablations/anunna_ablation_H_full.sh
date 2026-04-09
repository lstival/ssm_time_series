#!/bin/bash
#SBATCH --comment=ablation_H_full
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_H_full/ablation_H_full_%j.out
#SBATCH --error=logs/ablation_H_full/ablation_H_full_%j.err
#SBATCH --job-name=ablation_H_full
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/ablation_H_full
mkdir -p results/ablation_H_full
mkdir -p checkpoints/ablation_H_nano

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

echo "============================================================"
echo "Ablation H Full — Visual Encoder Architecture Validation"
echo "============================================================"
echo "Variants : cnn  rp_ss2d_2  upper_tri_diag"
echo "Model    : Nano (model_dim=128, depth=4, 50 epochs LOTSA)"
echo "Probe    : 9 ICML datasets × 4 horizons (96/192/336/720)"
echo ""

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/ablation_H_full.py \
    --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_nano.yaml \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_H_full \
    --checkpoint_base /home/WUR/stiva001/WUR/ssm_time_series/checkpoints/ablation_H_nano \
    --train_epochs 50 \
    --probe_epochs 20 \
    --batch_size 16 \
    --variants cnn rp_ss2d_2 upper_tri_diag \
    --horizons 96 192 336 720 \
    --seed 42

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
