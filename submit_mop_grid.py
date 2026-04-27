#!/usr/bin/env python3
"""
Submit MoP Experimental Grid to SLURM.
Generates jobs for Zero-Shot and Few-Shot variations.
"""

import os
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_BASE = ROOT_DIR / "results" / "mop_grid_v1"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "simclr_bimodal_nano" / "ts_simclr_bimodal_nano_lotsa_20260409_182831"
CONFIG = SRC_DIR / "configs" / "lotsa_simclr_bimodal_nano.yaml"

# Variations
EXPERIMENTS = [
    {"name": "baseline", "args": "--norm_mode identity --head_type linear"},
    {"name": "revin", "args": "--norm_mode revin --head_type linear"},
    {"name": "minmax", "args": "--norm_mode minmax --head_type linear"},
    {"name": "mlp", "args": "--norm_mode identity --head_type mlp"},
    {"name": "residual", "args": "--norm_mode identity --head_type linear --residual_head"},
    {"name": "ln_head", "args": "--norm_mode identity --head_type linear --use_ln_head"},
    {"name": "scale", "args": "--norm_mode identity --head_type linear --learnable_scale"},
    {"name": "temp_0.5", "args": "--norm_mode identity --head_type linear --temperature 0.5"},
    {"name": "temp_2.0", "args": "--norm_mode identity --head_type linear --temperature 2.0"},
    {"name": "scale_cond", "args": "--norm_mode identity --head_type linear --scale_cond"},
    {"name": "revin_mlp", "args": "--norm_mode revin --head_type mlp --use_ln_head"},
]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --err={log_dir}/{job_name}.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export PYTHONPATH=$PYTHONPATH:{src_dir}

# 1. Training on Source (LOTSA)
python3 {src_dir}/experiments/mop_train_flex.py \\
    --checkpoint_dir {checkpoint_dir} \\
    --config {config} \\
    --results_dir {out_dir} \\
    --mode source \\
    --epochs 10 \\
    --batches_per_epoch 1000 \\
    {extra_args}

# 2. Zero-Shot Evaluation
python3 {src_dir}/experiments/mop_eval_flex.py \\
    --mop_checkpoint {out_dir}/mop_flex.pt \\
    --checkpoint_dir {checkpoint_dir} \\
    --config {config} \\
    --results_dir {out_dir}/eval_zeroshot

# 3. Few-Shot Fine-Tuning and Eval (for each ICML dataset)
# (In a real grid, we might run this as a separate job, but for now we do a sample)
for ds in weather.csv traffic.csv electricity.csv; do
    python3 {src_dir}/experiments/mop_train_flex.py \\
        --checkpoint_dir {checkpoint_dir} \\
        --config {config} \\
        --results_dir {out_dir}/fewshot_$ds \\
        --mode fewshot \\
        --dataset_name $ds \\
        --epochs 5 \\
        --batches_per_epoch 200 \\
        {extra_args}
        
    python3 {src_dir}/experiments/mop_eval_flex.py \\
        --mop_checkpoint {out_dir}/fewshot_$ds/mop_flex.pt \\
        --checkpoint_dir {checkpoint_dir} \\
        --config {config} \\
        --results_dir {out_dir}/eval_fewshot_$ds
done
"""

def submit():
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT_DIR / "logs" / "mop_grid"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    for exp in EXPERIMENTS:
        name = exp["name"]
        extra_args = exp["args"]
        out_dir = RESULTS_BASE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = f"mop_{name}"
        sbatch_content = SBATCH_TEMPLATE.format(
            job_name=job_name,
            log_dir=log_dir,
            src_dir=SRC_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
            config=CONFIG,
            out_dir=out_dir,
            extra_args=extra_args
        )
        
        sbatch_file = out_dir / "job.sh"
        with open(sbatch_file, "w") as f:
            f.write(sbatch_content)
            
        print(f"Submitting job: {job_name}")
        subprocess.run(["sbatch", str(sbatch_file)])

if __name__ == "__main__":
    submit()
