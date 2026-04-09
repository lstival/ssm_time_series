# Training Pipeline Guide

This document describes how the model is trained end-to-end: which datasets are used at each stage, what each component learns, and how to run the full pipeline.

---

## Architecture Overview

The method has **two encoders** trained jointly via a CLIP-style contrastive objective:

| Encoder | Input | Role |
|---------|-------|------|
| **Time-Series Encoder** (`MambaBlock`) | Raw time-series patches | Learns temporal representations |
| **Visual Encoder** (`MambaVisualEncoder`) | Recurrence Plot (RP) of the same series | Learns visual/structural representations |

Both encoders are followed by a projection head. The objective aligns their embeddings in a shared latent space.

---

## Stage 1 — Encoder Pre-training (CLIP-style)

### Goal
Train both encoders to produce aligned representations of the same time series seen as:
- a raw sequence (TS encoder)
- an augmented positive view, also passed through RP transform (visual encoder)

### Loss
`clip_symm` — symmetric cross-entropy over a similarity matrix (standard CLIP InfoNCE).  
> **Important**: `cosine_mse` causes representation collapse (loss → 0 from epoch 1) because it has no repulsion term for negative pairs. Always use `clip_symm` for pre-training from scratch.

### Datasets — LOTSA (HuggingFace `Salesforce/lotsa_data`)
Large-scale diverse time-series corpora. Used to pre-train general-purpose representations.

| Dataset | Source | Approx. Series |
|---------|--------|---------------|
| m4_daily / hourly / monthly / weekly / quarterly / yearly | Salesforce/lotsa_data | ~100k |
| traffic_hourly / traffic_weekly | Salesforce/lotsa_data | 1,724 |
| hospital | Salesforce/lotsa_data | 767 |
| covid_deaths | Salesforce/lotsa_data | 266 |
| pedestrian_counts | Salesforce/lotsa_data | 66 |
| nn5_daily / nn5_weekly | Salesforce/lotsa_data | 222 |
| monash_m3_monthly / quarterly / yearly | Salesforce/lotsa_data | 2,829 |
| taxi_30min | Salesforce/lotsa_data | 67,984 |
| kdd_cup_2018 | Salesforce/lotsa_data | 270 |
| oikolab_weather | Salesforce/lotsa_data | 8 |
| exchange_rate | autogluon/chronos_datasets | 8 |
| saugeenday / us_births / sunspot | Salesforce/lotsa_data | 3 |

**Not available from HuggingFace** (missing during pre-training): `ett_h1`, `ett_h2`, `ett_m1`, `ett_m2`, `electricity_hourly`, `solar_10min`, `solar_weekly`.  
These are used **only** in the linear probe stage (loaded from local `ICML_datasets/`).

### Key config (`src/configs/lotsa_ablation_best.yaml`)
```yaml
data:
  dataset_type: "lotsa"
  cronos_kwargs:
    context_length: 96    # window size fed to both encoders
    two_views: true        # dataloader returns pre-augmented pair (target, target2)

model:
  input_dim: 64            # patch size (Ablation E: best accuracy)
  rp_mv_strategy: "mean"  # multivariate RP strategy (Ablation A: best accuracy)

training:
  alignment_strategy: "clip_symm"  # CLIP symmetric cross-entropy
  epochs: 100
```

### How to run
```bash
sbatch src/scripts/anunna_train_lotsa_ablation_best.sh
```
Checkpoints saved to: `checkpoints/lotsa_ablation_best/ts_encoder_lotsa_ablation_best_<DATE>/`

---

## Stage 2 — Linear Probe Evaluation

### Goal
Evaluate the **frozen** encoders on forecasting. A small linear head is trained on top of fixed encoder embeddings. This measures how much useful information the pre-trained representations capture.

### What is frozen / trained
| Component | Status |
|-----------|--------|
| Time-Series Encoder | **Frozen** (weights from Stage 1) |
| Visual Encoder | **Frozen** (weights from Stage 1) |
| Linear head | **Trained** (20 epochs per dataset×horizon) |

### Datasets — ICML Benchmark (local `ICML_datasets/`)
Standard long-term forecasting benchmarks. **These are NOT seen during pre-training.**

| Dataset | Horizons |
|---------|---------|
| ETTm1.csv, ETTm2.csv | 96, 192, 336, 720 |
| ETTh1.csv, ETTh2.csv | 96, 192, 336, 720 |
| weather.csv | 96, 192, 336, 720 |
| traffic.csv | 96, 192, 336, 720 |
| electricity.csv | 96, 192, 336, 720 |
| exchange_rate.csv | 96, 192, 336, 720 |
| solar_AL.txt | 96, 192, 336, 720 |

This gives **7 datasets × 4 horizons = 28 evaluation settings**.

### How to run
```bash
# After encoder training finishes:
sbatch src/scripts/anunna_probe_lotsa_ablation_best.sh
# The script auto-detects the latest checkpoint subdir.

# Or with explicit dependency on encoder job:
ENCODER_JOB_ID=<job_id> sbatch --dependency=afterok:<job_id> src/scripts/anunna_probe_lotsa_ablation_best.sh
```
Results saved to: `results/probe_lotsa_ablation_best/probe_lotsa_results.csv`

---

## Current Run (2026-04-03)

| Stage | Job | Status |
|-------|-----|--------|
| Encoder pre-training | 66204988 | RUNNING (clip_symm, fixed) |
| Linear probe | 66205378 | PENDING (depends on 66204988) |

---

## Data Flow Summary

```
LOTSA (HuggingFace, ~24 datasets)
        │
        ▼
  [Stage 1: CLIP Pre-training]
  TS Encoder ◄──── raw patches ────► Visual Encoder
  (MambaBlock)    clip_symm loss    (MambaVisualEncoder + RP transform)
        │
        ▼ freeze both encoders
  [Stage 2: Linear Probe]
  ICML Benchmark datasets (ETT, weather, traffic, electricity, exchange_rate, solar)
  → linear head trained per (dataset, horizon)
  → MSE / MAE reported
```

---

## Known Issues / Lessons Learned

- **`cosine_mse` alignment causes collapse**: MSE between normalized vectors has no repulsion — both encoders converge to the same constant unit vector (loss = 0 from epoch 1, zero gradients). Always use `clip_symm` for pre-training.
- **Checkpoint path is timestamped**: `checkpoints/lotsa_ablation_best/ts_encoder_lotsa_ablation_best_<YYYYMMDD_HHMM>/`. The probe script uses `ls -dt ts_encoder_* | head -1` to auto-detect the latest.
- **ETT/electricity not in LOTSA**: These key benchmarks are missing from the HuggingFace hub and are only used in the probe stage from local files.
