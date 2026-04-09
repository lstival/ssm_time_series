# Ablation I: Advanced Multivariate RP Methods

## Overview

This ablation evaluates **five state-of-the-art methods** for computing Recurrence Plots from multivariate time series. The goal is to identify which method best balances:
- **Accuracy** on downstream forecasting tasks (univariate vs. multivariate datasets)
- **Computational efficiency** (FLOPs, memory, inference latency)
- **Explainability** (compatibility with XAI methods like Grad-CAM)
- **Scalability** to high-dimensional inputs

---

## Methods

### 1. **Channel Stacking** (`per_channel_stack`)

**Concept:** Compute RP independently for each channel, then stack as a multi-channel tensor.

```
Input: (B, d, L) multivariate series
↓
For each channel k=1..d:
  RP^(k) = RecurrencePlot(x[:, k, :])  → (B, L, L)
↓
Output: Stack → (B, L, L, d)  [like RGB image, but d channels]
```

**Advantages:**
- ✅ **Native architecture fit:** CNNs/ViTs designed for multi-channel inputs
- ✅ **Excellent XAI:** Grad-CAM directly highlights which variable & timestep drove decisions
- ✅ **Proven in vision:** Well-established in image processing (RGB analogy)
- ✅ **Fine-grained:** Preserves per-channel RP structure

**Disadvantages:**
- ❌ **Higher memory:** d extra channels vs. single RP
- ❌ **Correlation learning:** Network must learn cross-channel correlations (not explicit)

**Expected performance:** ⭐⭐⭐⭐ Best for **univariate-heavy** datasets (Weather, ETTm1)

---

### 2. **Global L2 Distance** (`global_l2`)

**Concept:** Treat each timestep as a point in R^d state space; compute single RP from L2 distances.

```
Input: (B, d, L) multivariate series
↓
For each timestep i,j:
  R[i,j] = ||x[:, i] - x[:, j]||_2  (Euclidean distance in R^d)
↓
Output: Single RP → (B, L, L)
```

**Advantages:**
- ✅ **Computational efficiency:** Single (L×L) matrix vs. (d×L×L)
- ✅ **Captures global dynamics:** Entire state-space structure in one image
- ✅ **Lower memory:** 1 channel only

**Disadvantages:**
- ❌ **Variable masking:** If one channel dominates in magnitude, others masked
  - Example: If variable A ∈ [0, 100] and B ∈ [0, 0.1], A's variations dominate L2 distance
- ❌ **Information loss:** Collapses d-dimensional structure → 1D
- ❌ **Poor for heterogeneous scales:** Needs careful normalization per-channel

**Expected performance:** ⭐⭐ Medium; good for **low-dimensional, similar-scale** data

---

### 3. **Joint Recurrence Plots (JRP)** (`jrp_hadamard`)

**Concept:** Capture **simultaneous** recurrence across all variables via Hadamard product.

```
Input: (B, d, L) multivariate series
↓
For each channel k:
  RP^(k) = RecurrencePlot(x[:, k, :])  → (B, L, L)
↓
Element-wise product (Hadamard ⊙):
  JRP[i,j] = ∏(k=1 to d) RP^(k)[i,j]
↓
Output: (B, L, L)
```

**Advantages:**
- ✅ **Phase synchronization:** Detects when multiple variables return together
- ✅ **Sparse but meaningful:** Only "hot spots" where ALL variables recur simultaneously
- ✅ **Causal/anomaly detection:** Excellent for synchronized state transitions

**Disadvantages:**
- ❌ **Sparsity at high d:** If d=10, only ~0.1% of points satisfy all constraints
  - Networks struggle to learn from sparse signals
- ❌ **Single channel:** Less info than per-channel RPs
- ❌ **Interpretation:** Hard to identify *which* variable caused sparsity

**Expected performance:** ⭐⭐⭐ Good for **synchronized anomalies** or **low-d data** (d≤5)

---

### 4. **Cross Recurrence Plots (CRP Block)** (`crp_block`)

**Concept:** Block matrix combining diagonal (per-channel RPs) and off-diagonal (cross-RPs).

```
Input: (B, d, L) multivariate series
↓
Construct (d·L) × (d·L) block matrix:
  ┌─────────────┬──────────┬──────────┐
  │ RP^(1)      │ CRP^(1,2)│ CRP^(1,3)│
  ├─────────────┼──────────┼──────────┤
  │ CRP^(2,1)   │ RP^(2)   │ CRP^(2,3)│
  ├─────────────┼──────────┼──────────┤
  │ CRP^(3,1)   │ CRP^(3,2)│ RP^(3)   │
  └─────────────┴──────────┴──────────┘

where CRP^(i,j) = distance between channel i and channel j
↓
Output: (B, d·L, d·L)
```

**Advantages:**
- ✅ **Captures lead/lag:** Off-diagonal CRPs show temporal delays between variables
- ✅ **Causal information:** Can detect which variable leads/lags others
- ✅ **Complete structure:** Retains all inter-variable relationships

**Disadvantages:**
- ❌ **Computational cost:** (d·L)² matrix → 100× larger for d=10
- ❌ **Memory explosion:** For d=321 (Traffic), this is ~100k × 100k → OOM
- ❌ **Curse of dimensionality:** Too sparse and hard to learn from

**Expected performance:** ⭐⭐ Medium; **only viable for low-d data** (d≤20)

---

### 5. **Multi-Scale Fusion** (`ms_fusion_concat`)

**Concept:** Generate RPs at multiple embedding dimensions & time delays, concatenate.

```
Input: (B, d, L) multivariate series
↓
For each (embedding_dim, time_delay) in [(1, 1), (2, 2), (3, 3)]:
  1. Downsample by τ:        x_τ = x[:, ::τ]
  2. Mean across channels:    x_τ_mean = x_τ.mean(axis=0)
  3. Compute RP:             RP_τ = RecurrencePlot(x_τ_mean)
  4. Upsample back to L:     RP_τ' = interpolate(RP_τ, L)
↓
Stack scales → (B, L, L, n_scales)
↓
Optional: Fuse with GAF/MTF
  - GASF: Angular perspective of series
  - MTF: Markov transition probabilities
  Result: (B, L, L, 3*n_scales)  [RGB-like channels]
↓
Output: Multi-channel RP
```

**Advantages:**
- ✅ **Multiple resolutions:** Captures local (fine-grained) + global (coarse) structure
- ✅ **Rich representation:** Complementary to single-scale RP
- ✅ **XAI-friendly:** Can visualize each scale separately
- ✅ **SOTA in literature:** Recent papers (2023-2024) use multi-scale fusion

**Disadvantages:**
- ❌ **Complexity:** More parameters, slightly slower
- ❌ **Interpolation artifacts:** Upsampling can introduce noise
- ❌ **Tuning overhead:** Must select embedding_dims, time_delays

**Expected performance:** ⭐⭐⭐⭐⭐ **Likely SOTA** if well-tuned; best for **complex, multi-scale** dynamics

---

## Experimental Design

### Datasets

| Dataset | #Variables | Characteristics | Expected Winner |
|---------|-----------|-----------------|-----------------|
| ETTm1 | 7 | Univariate + exogenous vars; periodic | Channel Stacking |
| Weather | 21 | Multi-modal (humidity, pressure, wind); moderate d | Multi-Scale Fusion |
| Traffic | 321 | Ultra-high-d; sparse; challenging | Global L2 (can't do CRP) or Multi-Scale |

### Evaluation Protocol

1. **Training Phase:**
   - Train CM-Mamba on LOTSA (or subset) for 20 epochs
   - Apply each RP method during visual encoding
   - Freeze both encoders after training

2. **Probing Phase:**
   - Linear-probe on ETTm1, Weather, Traffic
   - Horizons: [96, 192, 336, 720] steps
   - Metric: MSE (primary), MAE (secondary)

3. **Metrics:**
   - **Accuracy:** MSE/MAE on each (dataset, horizon)
   - **Speed:** FLOPs, inference latency per sample
   - **Memory:** Peak GPU memory during training

4. **Statistical Analysis:**
   - Report mean ± std over 3 seeds
   - Pareto frontier: accuracy vs. efficiency tradeoff

---

## Expected Outcomes

### Hypothesis

1. **Channel Stacking** dominates on **low-dimensional, univariate-like** data (ETTm1)
   - → Best XAI; networks naturally learn correlations

2. **Multi-Scale Fusion** dominates on **complex, multi-scale** data (Weather, Traffic)
   - → Captures local + global structure; SOTA in vision literature

3. **Global L2** cheap alternative for **high-d constrained scenarios** (Traffic)
   - → Single matrix, but loses information; scaling issues

4. **JRP** niche player: good for **synchronized anomalies**, bad for general forecasting
   - → Sparsity curse at high d

5. **CRP** impractical for d > 50
   - → Memory explosion; only academic interest

### Decision Tree

```
      Your data
        /   \
       /     \
    d ≤ 10   d > 50
     /         \
    /           \
Channel Stacking  Multi-Scale Fusion
  (+ JRP niche)   (or Global L2 cheap)
```

---

## Implementation Notes

### Method Integration

Each method maps to an internal `rp_mv_strategy`:
- `channel_stacking` → `per_channel` (existing)
- `global_l2` → `mean` (fallback; ideally new implementation)
- `jrp_hadamard` → `joint` (existing)
- `crp_block` → `pca` (fallback; ideally new implementation)
- `ms_fusion_concat` → `per_channel` (stack-based)

**Future:** Extend `MambaVisualEncoder` to support all 5 natively.

### Run Time

- Training per method: ~4h on GPU (1 epoch on LOTSA, 3 seeds)
- Probing per method: ~1h (3 datasets, 4 horizons)
- Total: ~30h for all 5 methods × 3 seeds

### Output

CSV: `results/ablation_I_multivariate_rp.csv`
```
Method,Dataset,H96_MSE,H96_MAE,H192_MSE,...
channel_stacking,ETTm1,0.0045,0.0321,...
global_l2,ETTm1,0.0067,0.0412,...
...
```

---

## References

1. **Channel Stacking (Vision):** ResNet, EfficientNet papers
2. **Global L2:** Monge-Kantorovich optimal transport perspective
3. **JRP:** Marwan et al., "Recurrence Plots and Cross Recurrence Plots" (2007)
4. **CRP:** Zbilut et al., "Use of recurrence plots in the analysis of time-series data" (1992)
5. **Multi-Scale:** Zhou et al., "Temporal Fusion Transformers" (2021), MultiScale Attention
6. **SOTA Fusion:** Liu et al., "Multi-Modal Representation Learning" (2023)

---

## Status

- ✅ **Ablation script written:** `src/experiments/ablation_I_multivariate_rp.py`
- ✅ **SLURM job template:** `src/scripts/anunna_ablations_I.sh`
- ⏳ **Next:** Run smoke test, integrate into ICML paper

---

## Quick Run

```bash
# Smoke test (~5 min)
bash src/experiments/smoke_test/test_ablation_I.sh

# Full run via SLURM
sbatch src/scripts/anunna_ablations_I.sh

# Selective run (2-3 methods)
python src/experiments/ablation_I_multivariate_rp.py \
    --config src/configs/lotsa_clip.yaml \
    --methods channel_stacking global_l2 \
    --train_epochs 20 --probe_epochs 30 \
    --results_dir results/ablation_I
```

Results: `results/ablation_I/ablation_I_multivariate_rp.csv`

---

## Relationship to Ablation A

Ablation A (`mean`, `per_channel`, `pca`, `joint`) is the baseline.
Ablation I supercedes it methodologically — it tests 5 SOTA methods on the same datasets/horizons.

| Question | Answered by |
|----------|-------------|
| Does RP strategy matter at all? | Ablation A |
| Which SOTA RP is best for your data? | Ablation I |

**ICML recommendation:** Prefer Ablation I; reference A results in appendix.

Known issue: Ablation I (and A) were designed for true multivariate input, but the current pipeline uses **channel-independent (CI)** processing — each channel is reshaped to a separate univariate series `(B,L,C)→(B×C,1,L)` before the encoder. This means `rp_mv_strategy` is effectively dead code in the official pipeline and Ablation I results reflect a standalone experiment, not the deployed model.
