# Additional Experiments for ICML Submission

**Project:** CM-Mamba  
**Audience:** Senior Python / ML / Software Engineers  
**Goal:** Strengthen methodological rigor, validate zero-shot claims, and isolate causal factors behind performance gains.

## Global Experiment Constraints

These constraints apply to all experiments unless explicitly stated otherwise.

*   **Framework:** PyTorch ≥ 2.0
*   **Precision:** fp32 (default), fp16 only if explicitly stated
*   **Optimizer:** AdamW
*   **Learning Rate:** Fixed per experiment, no adaptive tuning
*   **Random Seeds:** `{2024, 2025, 2026}`
*   **Evaluation:** Zero-shot forecasting only (frozen encoder)
*   **Metrics:** MAE, MSE, RMSE
*   **Datasets:** ETTh1, Electricity, Weather

> [!NOTE]
> All results must report mean ± std over seeds.

---

## EXP-1: Necessity of Contrastive Learning

### Objective
Determine whether performance gains come from contrastive alignment, not merely from multimodal input.

### Models to Implement
Train the following four models under identical settings:

1.  **Temporal-Only Baseline**
    *   **Input:** 1D time-series patches
    *   **Encoder:** Mamba
    *   **Loss:** Supervised forecasting loss
2.  **Multimodal Without Contrastive**
    *   **Input:** 1D + Recurrence Plot (RP)
    *   **Encoder:** Two Mamba encoders
    *   **Fusion:** Concatenation `[z_1d || z_2d]`
    *   **Loss:** Supervised forecasting loss only
3.  **Contrastive Multimodal (SSL)**
    *   **Input:** 1D + RP
    *   **Encoder:** Dual Mamba
    *   **Loss:** Contrastive loss only
    *   **Evaluation:** Linear probing
4.  **Contrastive + Fine-Tuning (Upper Bound)**
    *   Same as (3), but fine-tune encoder + forecasting head.

### Implementation Notes
*   Ensure identical initialization across variants.
*   Disable contrastive loss entirely for model (2).
*   Use the same RP generation pipeline for all multimodal variants.
*   Log contrastive loss convergence separately.

### Expected Outcome
*   Model (3) > Model (2) → Contrastive learning is essential
*   Model (1) < Model (3) → Multimodal SSL improves representations
*   Model (4) establishes performance ceiling

---

## EXP-2: Recurrence Plot Information Leakage Test

### Objective
Verify that recurrence plots do not leak future information and are not acting as shortcuts.

### RP Variants
Implement three RP modes:

1.  **Correct RP:** RP computed strictly within each temporal patch.
2.  **Shuffled RP (Negative Control):** Randomly shuffle RP pixels per patch.
3.  **Random RP (Noise Control):** Replace RP with Gaussian noise.

### Reference Implementation
```python
def generate_rp(x, mode="correct"):
    rp = compute_recurrence_plot(x)
    
    if mode == "shuffled":
        flat = rp.flatten()
        np.random.shuffle(flat)
        rp = flat.reshape(rp.shape)
        
    elif mode == "random":
        rp = np.random.normal(0, 1, size=rp.shape)
        
    return rp
```

### Metrics
*   Forecasting MAE / MSE
*   Contrastive alignment accuracy
*   Training stability

### Expected Outcome
*   Correct RP >> Shuffled RP >> Random RP
*   Random RP should collapse to temporal-only baseline performance.

---

## EXP-3: Comparison with Other 2D Time-Series Representations

### Objective
Demonstrate that recurrence plots are a principled choice, not arbitrary.

### Representations to Evaluate
Replace RP with:
*   GASF (Gramian Angular Summation Field)
*   MTF (Markov Transition Field)
*   STFT Spectrogram
*   Recurrence Plot (baseline)

### Implementation Notes
*   Use `pyts.image` for GASF/MTF.
*   Use `scipy.signal.stft` for spectrogram.
*   Resize all 2D representations to $(L, L)$.
*   Keep encoder and loss unchanged.

### Metrics
*   Zero-shot MAE / MSE
*   Contrastive matching accuracy
*   Compute cost (ms / batch)

### Expected Outcome
*   RP best captures short-term dynamics, especially for non-periodic datasets.

---

## EXP-4: Horizon-Specific Error Decomposition

### Objective
Analyze whether multimodal learning improves local vs global horizons differently.

### Horizon Groups
*   **Short-term:** 1–48
*   **Mid-term:** 49–192
*   **Long-term:** 193–720

### Implementation
*   Log forecasting error per timestep.
*   Aggregate per horizon group.

### Expected Outcome
*   RP improves short-term accuracy.
*   Mamba improves long-term accuracy.
*   Contrastive model dominates across all ranges.

---

## EXP-5: Strict Cross-Domain Zero-Shot Evaluation

### Objective
Validate true zero-shot generalization by removing all overlapping domains from pretraining.

### Pretraining Dataset Filter
*   **KEEP ONLY:**
    *   Healthcare: Covid Deaths, Hospital
    *   Generic benchmarks: M1, M3, M4, NN5
*   **REMOVE:**
    *   All Energy datasets
    *   All Traffic / Transport datasets
    *   Exchange Rate
    *   Weather-related datasets

### Implementation Notes
*   Rebuild pretraining dataloader with filtered dataset list.
*   Freeze encoder during evaluation.
*   Compare against “soft zero-shot” setup.

### Expected Outcome
*   Performance drops but multimodal contrastive model remains superior.

---

## EXP-6: Seed Stability and Variance Analysis

### Objective
Demonstrate robustness and reproducibility.

### Setup
*   Run 3–5 random seeds.
*   **Models:**
    *   CM-Mamba (best configuration)
    *   Strong baseline (LightGTS)

### Report
*   Mean ± std
*   Worst-case deviation

---

## EXP-7: Compute-Normalized Comparison

### Objective
Assess efficiency vs accuracy trade-off.

### Metrics
*   Parameters
*   FLOPs
*   Training time
*   Zero-shot MSE

### Visualization
*   **Scatter plot:**
    *   X-axis: GFLOPs
    *   Y-axis: MSE
    *   Bubble size: #parameters

---

## Reporting Requirements

All experiments must include:
*   Dataset split details
*   Random seeds
*   Hardware configuration
*   Full hyperparameter tables

> [!IMPORTANT]
> **Final Notes:**
> Experiments **EXP-1**, **EXP-2**, and **EXP-5** are mandatory for ICML-level rigor. Failure to include them will likely result in rejection due to insufficient causal analysis or weak zero-shot validation.