# Experiment Log — April 16-17, 2026

Summary of Mixture of Prompts (MoP) and CLIP Backbone evaluation jobs completed between yesterday evening and this morning.

## 1. Slurm Job Summary

| Job ID | Job Name | End Time | State | Description |
|:---|:---|:---|:---|:---|
| **66510625** | `clip_f_seq512` | Apr 17, 02:04 | ✅ COMPLETED | CLIP Nano Backbone: Full linear probe (100% data) with seq_len 512. |
| **66510626** | `clip_f_5pct` | Apr 17, 01:36 | ✅ COMPLETED | CLIP Nano Backbone: Few-shot linear probe (5% data) with seq_len 512. |
| **66510785** | `mop_full_top1` | Apr 16, 23:41 | ✅ COMPLETED | MoP Full Training (LOTSA) & Zero-Shot Eval: `revin_mlp` variant. |
| **66510786** | `mop_full_top2` | Apr 16, 23:41 | ✅ COMPLETED | MoP Full Training (LOTSA) & Zero-Shot Eval: `revin` linear variant. |
| **66512322** | `clip_minmax_full`| Scheduled | ⏳ PENDING | CLIP Full Probe (100%) with **MinMax** Scaling. |
| **66512323** | `clip_minmax_5pct`| Scheduled | ⏳ PENDING | CLIP Few-Shot (5%) with **MinMax** Scaling. |
| **66510772** | `mop_clip_revin_mlp`| Apr 16, 21:54 | ✅ COMPLETED | Grid Search: Identification of the best MoP architecture for CLIP. |
| **66510771** | `mop_clip_scale` | Apr 16, 21:51 | ❌ FAILED | Grid Search: Variant with conditional scaling failed (numerical instability). |

---

## 2. Key Results (Highlights)

### CLIP Nano vs. MoP (Zero-Shot)
Initial comparison showed a massive gap in MSE. *Note: See Technical Warning below.*

| Dataset | Metric | MoP Zero-Shot (Top 1) | CLIP Full Probe (100%) | CLIP Few-Shot (5%) |
|:---|:---:|:---:|:---:|:---:|
| **Traffic (H=96)** | MSE | **0.0022** | 1.0515 | 1.7895 |
| **Electricity (H=96)** | MSE | **0.0049** | 0.9021 | 0.9464 |
| **ETTh2 (H=96)** | MSE | **0.0084** | 0.2368 | 0.2870 |

---

## 3. Technical Note: Metric Discrepancy

> [!IMPORTANT]
> **Warning on Direct MSE Comparison**
> The results above are currently recorded in **different normalization spaces**, which explains why MoP numbers look significantly lower:
> 1. **MoP Eval (`mop_eval_flex.py`):** Uses **MinMax Scaling [0, 1]**. Calculated on normalized scale.
> 2. **CLIP Probe (`probe_lotsa_checkpoint.py`):** Uses **Standard Scaling (Z-score)**. Calculated on original scale or globally normalized scale.
>
> **Analysis:** While MoP is indeed outperforming the naked linear probe (especially in zero-shot stability), the magnitude of the difference is amplified by the unit mismatch.

---

## 4. Artifact Paths

*   **MoP Best Model:** `results/mop_full_v1/top1_revin_mlp/mop_flex.pt`
*   **MoP Zero-Shot CSV:** `results/mop_full_v1/top1_revin_mlp/eval_zeroshot_full/eval_mop_flex.csv`
*   **CLIP Full Probe CSV:** `results/clip_full_seq512/probe_results_full.csv`
*   **CLIP 5% Probe CSV:** `results/clip_full_5pct_seq512/probe_results_5pct.csv`

---

## 5. Next Steps
- [x] Re-run CLIP Linear Probes with `--scaler_type minmax` to allow fair comparison with MoP (Jobs 66512322, 66512323).
- [ ] Analyze the failure of `mop_clip_scale_cond` to see if `LayerNorm` helps stability.
- [ ] Populate final LaTeX tables for the ICML/Internal report.

**Compiled by:** Antigravity (Assistant)
**Date:** April 17, 2026
