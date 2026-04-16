Ready for review
Select text to add comments on the plan
Plan: Bimodal SimCLR + Multimodal Alignment Comparison
Goal
Implement bimodal SimCLR (temporal + RP, NT-Xent over all 4 views) so all 5 methods share the same training infrastructure:

Method	Bimodal	Loss	EMA
CLIP	✅	InfoNCE cosine	No
GRAM	✅	InfoNCE Gramian vol	No
VL-JEPA	✅	MSE predictive	Yes
BYOL	❌ temporal only	MSE predictive	Yes
SimCLR (current)	❌ temporal only	NT-Xent	No
SimCLR bimodal (new)	✅	NT-Xent multiview	No
All trained on LOTSA micro config (100 epochs) + linear probe on 8 ICML datasets.

Files to Create
1. src/simclr_bimodal_training.py
New training script, modeled after cosine_training.py (bimodal structure) + simclr_training.py (NT-Xent loss).

Architecture:

Two encoders: MambaEncoder (temporal) + UpperTriDiagRPEncoder (visual)
Two projection heads (MLP, same as single SimCLR)
For each batch, produce 4 embeddings: z_t1, z_t2, z_v1, z_v2
z_t1 = proj_t(encoder_t(view1))
z_t2 = proj_t(encoder_t(view2)) ← second augmented view (Gaussian noise, like existing SimCLR)
z_v1 = proj_v(encoder_v(view1_rp))
z_v2 = proj_v(encoder_v(view2_rp))
Loss — multiview NT-Xent:

All 4N embeddings stacked → compute pairwise cosine similarity matrix (4N × 4N)
Positives: (z_t1_i, z_t2_i), (z_t1_i, z_v1_i), (z_t1_i, z_v2_i), 
           (z_t2_i, z_v1_i), (z_t2_i, z_v2_i), (z_v1_i, z_v2_i)
Negatives: all other cross-sample pairs
Loss = mean NT-Xent over all positive pairs
Actually simpler implementation: use 3 pairwise NT-Xent terms:

L_tt = NT-Xent(z_t1, z_t2) — intra-temporal
L_vv = NT-Xent(z_v1, z_v2) — intra-visual
L_tv = NT-Xent(z_t1, z_v1) + NT-Xent(z_t2, z_v2) — cross-modal
Total: L = L_tt + L_vv + L_tv
Checkpoint format (compatible with probe_lotsa_checkpoint.py):

Save time_series_best.pt (temporal encoder) under checkpoint_dir/
Save visual_encoder_best.pt (visual encoder) under checkpoint_dir/
Same as CLIP (cosine_training.py) — probe script loads these names
CLI:

python3 src/simclr_bimodal_training.py --config src/configs/lotsa_simclr_bimodal.yaml
2. src/configs/lotsa_simclr_bimodal.yaml
Micro config (100 epochs) — same hyperparams as micro but with bimodal keys:

experiment_name: "ts_simclr_bimodal_lotsa"
seed: 42
device: "auto"

model:
  model_name: "ts_simclr_bimodal_lotsa"
  rp_encoder: "upper_tri"
  input_dim: 64
  model_dim: 128
  embedding_dim: 64
  depth: 4
  state_dim: 16
  conv_kernel: 4
  expand_factor: 1.5
  dropout: 0.05
  pooling: "mean"
  rp_mv_strategy: "mean"

data:
  dataset_type: "lotsa"
  batch_size: 256
  val_batch_size: 128
  num_workers: 4
  pin_memory: true
  val_ratio: 0.1
  cronos_kwargs:
    context_length: 96
    cache_dir: "/lustre/nobackup/WUR/AIN/stiva001/hf_datasets"
    normalize_per_series: true
    force_offline: true
    drop_last: true
    two_views: true

training:
  epochs: 100
  temperature: 0.2
  learning_rate: 0.001
  weight_decay: 0.0001
  noise_std: 0.01
  max_grad_norm: 1.0
  warmup_epochs: 5
  use_amp: true

logging:
  checkpoint_dir: "../../checkpoints/simclr_bimodal"
  save_best_only: true
  save_last: true
3. src/scripts/ssl/anunna_simclr_bimodal.sh
SLURM training job — same template as anunna_gram.sh (A100, 2880min):

#!/bin/bash
#SBATCH --job-name=simclr_bimodal
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'
#SBATCH --output=logs/simclr_bimodal/train_%j.out
#SBATCH --error=logs/simclr_bimodal/train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl

mkdir -p logs/simclr_bimodal
export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
time python3 "${SRC}/simclr_bimodal_training.py" --config "${SRC}/configs/lotsa_simclr_bimodal.yaml"
4. src/scripts/ssl/anunna_probe_simclr_bimodal.sh
Linear probe job — same as anunna_probe_gram.sh but for simclr_bimodal checkpoints:

CHECKPOINTS=".../checkpoints/simclr_bimodal"
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_simclr_bimodal_lotsa_* | head -1)
# → probe_lotsa_checkpoint.py --config lotsa_simclr_bimodal.yaml --results_dir results/probe_simclr_bimodal
5. src/scripts/ssl/submit_simclr_bimodal.sh
Pipeline submit: train → probe with --dependency=afterok:

TRAIN=$(sbatch --parsable anunna_simclr_bimodal.sh)
PROBE=$(sbatch --parsable --dependency=afterok:${TRAIN} anunna_probe_simclr_bimodal.sh)
Also: Micro configs for remaining methods (missing ones)
Check which methods already have micro configs vs need them. The user said "micro config, 100 epochs" for ALL methods for fair comparison. Current status:

CLIP: lotsa_clip.yaml uses full model (depth=8, model_dim=256). Need lotsa_clip_nano.yaml?
GRAM: same, uses full model. Need lotsa_gram_nano.yaml?
VL-JEPA: lotsa_vl_jepa.yaml — check dims
BYOL: lotsa_byol_temporal.yaml — check dims
Decision: Create micro variants for CLIP, GRAM, BYOL, VL-JEPA so all 6 methods train at same scale (depth=4, model_dim=128, embedding_dim=64, 100 epochs). SimCLR bimodal already targets micro.

This means 6 new/updated configs + 6 SLURM train scripts + 6 probe scripts.

Simpler approach (less work): Only create the missing bimodal SimCLR, and use the existing trained checkpoints for the other methods. The comparison will note different model sizes. Let the user decide — plan assumes micro configs for all 6 methods but will only create new files for methods that don't already have micro configs.

Skill: multimodal-alignment
Create /home/WUR/stiva001/.claude/skills/multimodal-alignment/SKILL.md

The skill will:

Read SSL_METHOD_COMPARISON.md (current results)
Read src/configs/ for all 6 method configs
Read relevant training scripts to explain implementation
Answer questions about loss functions, encoder architecture, and comparison
Implementation Order
src/configs/lotsa_simclr_bimodal.yaml (config first)
src/simclr_bimodal_training.py (core new code)
src/scripts/ssl/anunna_simclr_bimodal.sh (SLURM train)
src/scripts/ssl/anunna_probe_simclr_bimodal.sh (SLURM probe)
src/scripts/ssl/submit_simclr_bimodal.sh (pipeline submit)
Micro configs for CLIP, GRAM, BYOL temporal, VL-JEPA (if needed for fair comparison)
Skill file
Verification
python3 -c "import sys; sys.path.insert(0, 'src'); import simclr_bimodal_training" — no import errors
python3 src/simclr_bimodal_training.py --config src/configs/lotsa_simclr_bimodal.yaml --smoke — 2 batches, no crash