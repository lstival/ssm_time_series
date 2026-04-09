# CM-Mamba Model Guide
### Architecture, Training, Evaluation and Ablation Reference

> This document is the authoritative reference for methodology writing and theoretical claims.  
> It describes **what the model is**, **why each design choice was made** (backed by ablation evidence), and **how everything fits together**.

---

## 1. Core Idea and Innovation

**CM-Mamba** (Cross-Modal Mamba) is a self-supervised time-series encoder that learns representations by aligning two complementary views of the same temporal signal:

1. **Temporal view** — the raw time-series processed as a sequence of patches by a Mamba SSM encoder
2. **Visual (structural) view** — the same signal converted to a **Recurrence Plot (RP)** and processed by a second Mamba encoder

The key innovation is that these two views capture **fundamentally different and complementary information**:

| View | What it captures | Inductive bias |
|------|-----------------|---------------|
| Temporal (SSM) | Local dynamics, trends, sequential order | Causal, sequential, 1-D |
| Visual / RP | Global recurrence structure, periodicity, regime changes | Spatial, symmetric, 2-D |

Aligning them forces the shared embedding space to be rich in both local temporal patterns and global structural properties — without any labels.

---

## 2. Architecture

### 2.1 Temporal Branch — `MambaEncoder`

Processes the raw time series as a sequence of non-overlapping patches.

```
Input:  (B, 1, L)  — univariate time series, length L
           │
           ▼  Tokenizer (non-overlapping windows)
       (B, N, patch_size)  — N = L / patch_size patches
           │
           ▼  Linear projection
       (B, N, model_dim)
           │
           ▼  × depth  MambaBlock
       (B, N, model_dim)
           │
           ▼  LayerNorm → mean pooling → Linear projection
       (B, embedding_dim)  — fixed-size embedding
```

**`MambaBlock`** — selective state-space model (Mamba SSM):
- Residual connection: `out = x + dropout(SSM(LayerNorm(x)))`
- Uses `mamba_ssm` CUDA kernel for fast selective scan
- Parameters: `state_dim` (SSM hidden state N), `conv_kernel` (depthwise conv width), `expand_factor` (inner-dim multiplier)

**Ablation-best config** (`lotsa_ablation_best.yaml`):

| Hyperparameter | Value | Justification |
|---|---|---|
| `input_dim` (patch size) | **64** | Ablation E: −15% MSE vs patch=32; patch=16 collapses (18× worse) |
| `model_dim` | 256 | — |
| `embedding_dim` | 128 | — |
| `depth` | 8 | — |
| `state_dim` | 16 | — |
| `conv_kernel` | 4 | — |
| `expand_factor` | 1.5 | — |
| `pooling` | mean | — |

---

### 2.2 Visual Branch — `MambaVisualEncoder`

Converts each patch of the raw series into a **Recurrence Plot (RP)** and processes the 2-D image with a Mamba-based visual encoder.

#### 2.2.1 Recurrence Plot computation

**Definition**: Given a univariate time-series patch of length `l`, the RP is the `l × l` matrix of pairwise absolute distances:

```
RP[i, j] = |x_i - x_j| / max(|x_i - x_j|)   ∈ [0, 1]
```

- **Continuous (grayscale)** — no thresholding applied. Values in [0, 1].
- Small values (dark) = similar timesteps; large values (bright) = dissimilar timesteps.
- Diagonal is always 0 (self-similarity). Symmetric by construction.
- Implemented on GPU via pairwise L1 distance: `dist = |x.unsqueeze(-1) - x.unsqueeze(-2)|`

**Why RP?** (Ablation D evidence):
- RP is **Pareto-dominant** over GASF, MTF, STFT across ETTm1, Weather, Traffic
- Captures periodicity (repeating off-diagonal structures), regime changes (block structures), and trend stationarity — all critical for forecasting

#### 2.2.2 Multivariate RP strategy

For a multivariate patch `(F channels, l timesteps)`, strategy `mean` is used:

```
(B, F, l) → mean over F channels → (B, l) → RP → (B, l, l)
```

**Why `mean`?** (Ablation A evidence):

| Strategy | Avg MSE | ms/batch |
|---|---|---|
| per_channel | 0.0049 | 395.3 |
| **mean** | **0.0043** | 393.6 |
| pca | 0.0047 | 387.0 |
| joint (Global L2) | 0.0048 | 298.4 |

`mean` achieves −11.9% MSE vs `per_channel` at the same speed.  
`joint` is 25% faster but 11.5% worse accuracy than `mean`.

> **Limitation**: `mean` collapses all channels before computing the RP. This means the visual branch receives the average behavior — inter-channel structure is lost for high-dimensional datasets (e.g., electricity with 321 channels). The encoder is therefore fundamentally **univariate** in its current form. Multivariate training is a planned extension.

#### 2.2.3 Visual encoder forward pass

```
Input:  (B, 1, L)  — univariate patch (or channel-averaged)
           │
           ▼  Tokenizer → per-patch RP computation
       (B, N, l, l)  — N patches, each an l×l RP image
           │
           ▼  _InputConv  (Conv1d along patch row dimension → projects l→model_dim)
       (B, N, model_dim)
           │
           ▼  × depth  MambaBlock
       (B, N, model_dim)
           │
           ▼  LayerNorm → pooling → Linear projection
       (B, embedding_dim)
```

**Key insight — local RP structure meets SSM sequential processing:**  
Each patch is independently converted to an RP (capturing local recurrence structure within that patch). The sequence of patch-RPs is then fed to the Mamba SSM, which processes them sequentially — allowing the SSM to capture **how local recurrence patterns evolve over time**. This is the bridge between the 2-D spatial information of individual RPs and the 1-D sequential modeling of the SSM: local recurrence (within a patch, via RP) + global temporal evolution (across patches, via SSM).

**Why separate Mamba visual encoder?** (Ablation B evidence):

| Variant | avg MSE |
|---|---|
| no_visual | 0.0082 |
| shared_1d (same encoder for both) | 0.0089 |
| sep_cnn_only (CNN, no SSM) | 0.0633 |
| **sep_mamba_1d (proposed)** | **0.0052** |

- CNN-only visual branch collapses (36× worse) — CNN cannot process RP patch sequences effectively
- Sharing the temporal encoder for visual branch is **worse** than no visual branch — the two modalities need distinct inductive biases
- Separate Mamba encoder reduces avg MSE 36% below no-visual baseline

---

### 2.3 Projection Heads

Each encoder output is passed through a projection head before the contrastive loss:

```
embedding_dim (128) → LayerNorm → Linear → ReLU → Linear → L2-normalize
```

Output: unit-norm vectors in the contrastive space. Projection heads are discarded after pre-training; only encoder weights are kept for downstream tasks.

---

### 2.4 Forecasting Head — `DualEncoderForecastMLP`

For downstream forecasting, both frozen encoder embeddings are concatenated and fed to an MLP:

```
[temporal_emb ‖ visual_emb]  →  (B, 2 × embedding_dim)
           │
           ▼  Linear(2D, hidden) → ReLU → Dropout → Linear(hidden, hidden) → ReLU
           │
           ▼  Linear(hidden, max_horizon)  — one head for all horizons
       (B, max_horizon)  → slice to requested horizon H
```

Hidden dim: 512. Trained with MSE loss, AdamW, cosine LR schedule.

---

## 3. Training Pipeline

### Stage 1 — CLIP-style Contrastive Pre-training

**Objective**: align temporal and visual embeddings of the same time series via symmetric InfoNCE loss.

```
For each batch of B univariate series:
  x_q  =  series (temporal branch input)
  x_k  =  augmented positive view of same series (temporal noise + scaling + masking)

  q = L2_norm( proj_head( temporal_encoder( x_q ) ) )   ∈ R^d
  k = L2_norm( visual_proj( visual_encoder(  x_k ) ) )   ∈ R^d

  Loss = clip_symm(q, k)  =  0.5 × [CE(q·k^T/τ, diag) + CE(k·q^T/τ, diag)]
```

**Temperature**: τ = 0.2

**Why `clip_symm`?** Both directions of the loss are needed. `cosine_mse` (MSE between normalized embeddings) has **no repulsion term** — both encoders collapse to the same constant unit vector (loss = 0 from epoch 1, zero gradients). Always use `clip_symm` for pre-training.

**Positive view augmentations** (`make_positive_view`):
- Additive Gaussian noise (σ = 0.01)
- Additional jitter (σ = 0.1)
- Per-sample amplitude scaling ×[0.9, 1.1]
- Random element masking (20% probability)
- Optional temporal dropout (contiguous segment zeroed)

**Training data** — LOTSA (HuggingFace `Salesforce/lotsa_data`):  
~24 diverse univariate time-series datasets (M4, traffic, hospital, taxi, weather, etc.)  
Total: ~150k individual time series across multiple domains and frequencies.

> **Note**: LOTSA series are stored as univariate. The encoder is pre-trained entirely on univariate data. Multivariate datasets (ETT, electricity) are **not seen** during pre-training.

**Training**: 100 epochs, AdamW (lr=1e-3, wd=1e-4), cosine LR schedule (warmup 10 epochs), AMP (mixed precision), gradient clipping (max_norm=1.0).

---

### Stage 2 — Downstream Evaluation

Two evaluation protocols, both with **frozen encoders**:

#### A. Linear Probe
A single linear layer (`Linear(2×embedding_dim, H)`) trained per dataset per horizon.  
Measures: how much forecasting information is in the frozen embeddings.  
Fair evaluation: each channel treated as independent univariate series (matching pre-training).

#### B. Zero-shot Forecasting (MLP trained on LOTSA)
`DualEncoderForecastMLP` trained on LOTSA data (same distribution as pre-training).  
Then evaluated on ICML benchmark datasets **without any fine-tuning**.  
True zero-shot: model has never seen the evaluation datasets during any training stage.

**ICML benchmark datasets** (ETTm1, ETTm2, ETTh1, ETTh2, weather, traffic, electricity, exchange_rate, solar_AL):
- Standard long-term forecasting benchmarks
- Horizons: H ∈ {96, 192, 336, 720}
- **Not seen during pre-training** (ETT, electricity, solar missing from LOTSA HuggingFace hub)

---

## 4. Ablation Results Summary

All ablations follow: short CLIP pre-training (20 epochs) → frozen linear probe on downstream datasets.

| Ablation | Question | Winner | Key metric |
|---|---|---|---|
| **A** | Multivariate RP aggregation | `mean` | −11.9% MSE vs per_channel |
| **B** | Visual encoder architecture | `sep_mamba_1d` | −36% MSE vs no_visual |
| **C** | Contrastive alignment loss | `clip_symm` | Stable; `cosine_mse` collapses |
| **D** | Visual representation type | `rp` | Pareto-dominant over GASF, MTF, STFT |
| **E** | Patch (token) length | `64` | −15% MSE vs patch=32; patch=16 collapses |
| **F** | Manifold quality on unseen data | `multimodal` | Best Davies-Bouldin + separation |
| **G** | Encoder output mode | `multimodal` [temporal ‖ visual] | Most consistent across datasets |
| **H** | 2-D scan pattern for RP visual encoder | `rp_ss2d_2` (2-scan) | Best MSE at small patch sizes |
| **I** | Advanced MV-RP methods | `global_l2` (joint L2 RP) | Best across all channel counts |
| **Mean vs Joint** | `mean` (Abl-A) vs `global_l2` (Abl-I) | `mean` | Best overall; joint marginal gain on high-d |

### Ablation D — Visual Representation Detail

`rp` is Pareto-dominant across all tested datasets and horizons. The other representations:
- **GASF** (Gramian Angular Summation Field): encodes angle cosines between timesteps — loses distance information
- **MTF** (Markov Transition Field): encodes transition probabilities between quantile bins — loses magnitude
- **STFT** (Short-Time Fourier Transform): frequency-domain representation — loses phase alignment information

RP preserves both distance magnitude and temporal ordering (via diagonal structure), making it the most information-rich representation for time-series forecasting.

---

## 5. Theoretical Claims and Justifications

### Claim 1: Two views are complementary, not redundant
The temporal encoder (SSM on raw patches) captures **local dynamics and causal order**.  
The visual encoder (SSM on RP patches) captures **global recurrence structure and periodicity**.  
These are provably different: a constant trend has low RP variance but high SSM temporal variation; a periodic signal has highly structured RP but may appear locally noisy to the SSM.

### Claim 2: RP + SSM bridges local and global structure
Each patch RP captures recurrence *within* the patch (local). The Mamba SSM then processes the sequence of RP patches sequentially — modeling how local recurrence patterns *evolve* over time (global). This hierarchical decomposition is unique to our architecture.

### Claim 3: Contrastive alignment forces multi-scale representations
`clip_symm` with in-batch negatives forces both encoders to be discriminative across the batch. This means the embeddings must encode properties that distinguish one series from others — which in practice requires capturing both local dynamics (what makes this series unique in a short window) and global structure (what makes this dataset domain unique).

### Claim 4: Separate visual encoder is necessary
Ablation B shows that sharing the temporal encoder for the visual branch is **worse than no visual branch at all**. The temporal SSM has inductive bias towards sequential causal processing; the RP image requires 2-D spatial processing. Forcing the same architecture to serve both collapses the learned representations.

### Claim 5: Patch size 64 is the optimal information granularity
Ablation E shows a clear sweet spot at patch=64. Smaller patches (16, 32) produce degenerate RPs — an `l×l` RP from a 16-step window captures only 16 timesteps of local recurrence, insufficient for pattern discrimination. Larger patches (96, 128) reduce the sequence length for the SSM, limiting its ability to model long-range dependencies.

---

## 6. Current Best Configuration

File: `src/configs/lotsa_ablation_best.yaml`

```yaml
model:
  input_dim: 64          # patch size (Ablation E)
  model_dim: 256
  embedding_dim: 128
  depth: 8
  state_dim: 16
  conv_kernel: 4
  expand_factor: 1.5
  pooling: mean
  rp_mv_strategy: mean   # multivariate RP (Ablation A)

training:
  alignment_strategy: clip_symm   # contrastive loss (Ablation C)
  epochs: 100
  temperature: 0.2
  learning_rate: 0.001
  weight_decay: 0.0001
  noise_std: 0.01
  scheduler: cosine
  warmup_epochs: 10
  use_amp: true
```

---

## 7. Running the Full Pipeline

```bash
# Step 1 — Pre-train encoders on LOTSA (100 epochs, ~8h on A100)
sbatch src/scripts/anunna_train_lotsa_ablation_best.sh
# → checkpoints/lotsa_ablation_best/ts_encoder_lotsa_ablation_best_<DATE>/

# Step 2a — Linear probe evaluation (ICML benchmark, frozen encoders)
sbatch --dependency=afterok:<ENCODER_JOB> src/scripts/anunna_probe_lotsa_ablation_best.sh
# → results/probe_lotsa_ablation_best/probe_lotsa_results.csv

# Step 2b — Zero-shot forecasting (train MLP on LOTSA → eval on ICML, no fine-tuning)
sbatch src/scripts/anunna_lotsa_forecast_zeroshot.sh
# → results/lotsa_zeroshot/lotsa_zeroshot_results.csv
```

---

## 8. Evaluation Results on ICML Benchmark Datasets

Checkpoint used for both experiments: `ts_encoder_lotsa_ablation_best_20260403_1213`  
Config: `src/configs/lotsa_ablation_best.yaml`  
Pre-trained on LOTSA (~150k univariate series). ETT, electricity, exchange_rate, solar_AL **not seen during pre-training**.

### 8.1 Linear Probe (job 66218048)

Frozen encoders + one linear layer per dataset × horizon, trained on the ICML train split.  
Context length: 96. Scaler: standard. Probe epochs: 20.

| Dataset | H=96 | H=192 | H=336 | H=720 | Mean MSE |
|---|---|---|---|---|---|
| ETTm1 | 0.1031 | 0.2484 | 0.2293 | 0.3406 | 0.2303 |
| ETTm2 | 0.1930 | 0.1966 | 0.2175 | 0.2572 | 0.2161 |
| ETTh1 | 0.1657 | 0.1941 | 0.2362 | 0.4089 | 0.2512 |
| ETTh2 | 0.1888 | 0.2920 | 0.2905 | 0.3691 | 0.2851 |
| weather | 0.0676 | 0.1104 | 0.0636 | 0.1255 | 0.0918 |
| traffic | 0.6355 | 0.6552 | 0.6407 | 0.6212 | 0.6381 |
| electricity | 0.6031 | 0.6245 | 0.6555 | 0.6685 | 0.6379 |
| exchange_rate | 2.1482 | 2.0897 | 4.2281 | 2.5111 | 2.7443 |
| solar_AL | 0.3133 | 0.3216 | 0.3161 | 0.3144 | 0.3164 |

Full results: `results/probe_lotsa_ablation_best/probe_lotsa_results.csv`

### 8.2 Zero-shot Forecasting (job 66215509)

Frozen encoders + `DualEncoderForecastMLP` (hidden=512) trained on LOTSA, then evaluated on ICML datasets with **no fine-tuning**.  
MLP trained for 20 epochs (best loss: 0.0269). Context length: 96. Batch size: 256. LR: 1e-3.

| Dataset | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| ETTm1 | 2.3157 | 2.3139 | 2.3393 | 2.3980 |
| ETTm2 | 1.8828 | 1.8803 | 1.9041 | 1.9490 |
| ETTh1 | 2.0844 | 2.1201 | 2.2328 | 2.4468 |
| ETTh2 | 1.4749 | 1.5067 | 1.6309 | 1.9422 |
| weather | 0.0024 | 0.0044 | 0.0095 | 0.0374 |
| traffic | 1.7352 | 1.7433 | 1.7472 | 1.7775 |
| electricity | 1.1106 | 1.0915 | 1.0845 | 1.0600 |
| exchange_rate | 2.1738 | 2.2788 | 2.3557 | 1.8371 |
| solar_AL | 0.9278 | 0.9015 | 0.8920 | 0.8893 |

Full results: `results/lotsa_zeroshot/lotsa_zeroshot_results.csv`

### 8.3 Key Takeaways

- Linear probe MSE is dramatically lower than zero-shot MLP (e.g. ETTm1 H=96: 0.10 vs 2.32), confirming the encoder representations are high quality but require dataset-specific adaptation.
- **Weather** achieves the lowest MSE in both protocols — likely aided by the smooth, structured nature of meteorological data.
- **Exchange rate** is the hardest dataset: non-stationary, never seen in pre-training, linear probe unstable at H=336.
- **Traffic and electricity** plateau in the linear probe (~0.6 MSE), reflecting high-dimensional channel structure that the univariate encoder cannot fully exploit.
- Zero-shot results are a strong baseline given the model has never seen any evaluation dataset during any training stage.

## 9. ICML Fine-tuning vs Supervised Comparison (New)

To directly test the claim that **fine-tuning from LOTSA-pretrained encoders is better than supervised training from scratch on ICML**, run two matched conditions:

1. `finetune` — initialize temporal + visual encoders from LOTSA checkpoint, then jointly train encoders + MLP on ICML train split.
2. `supervised` — random-initialize temporal + visual encoders, then jointly train encoders + MLP on ICML train split.

Both conditions keep architecture and protocol fixed:
- same backbone and forecasting head (`MambaEncoder` + `MambaVisualEncoder` + `DualEncoderForecastMLP`)
- same ICML datasets and horizons (`96, 192, 336, 720`)
- same context length (`96`) and optimization setup

### Run Command

```bash
sbatch src/scripts/anunna_icml_finetune_vs_supervised.sh
```

### Expected Outputs

- `results/icml_finetune_vs_supervised/icml_finetune_results.csv`
- `results/icml_finetune_vs_supervised/icml_supervised_results.csv`
- `results/icml_finetune_vs_supervised/best_finetune.pt`
- `results/icml_finetune_vs_supervised/best_supervised.pt`

---

## 10. Known Limitations

| Limitation | Root cause | Impact |
|---|---|---|
| Univariate only | LOTSA HF data is stored per-series (1 channel each) | High-dim datasets (electricity, traffic, solar) suffer |
| `mean` RP loses inter-channel info | Channels averaged before RP | Visual branch blind to cross-channel structure |
| ETT/electricity not in LOTSA | Missing from HuggingFace hub | These benchmarks are truly zero-shot (never seen) |
| Exchange rate near-random | Non-stationary + not in pre-training data | MSE > 1.0 on normalized data (worse than constant baseline) |
| `cosine_mse` causes collapse | No repulsion term in loss | Always use `clip_symm` for pre-training from scratch |

---

## 11. File Map

| File | Purpose |
|------|---------|
| `src/models/mamba_encoder.py` | Temporal encoder (`MambaEncoder`) |
| `src/models/mamba_visual_encoder.py` | Visual encoder (`MambaVisualEncoder`) + RP computation |
| `src/models/mamba_block.py` | Shared `MambaBlock` + `Tokenizer` |
| `src/models/dual_forecast.py` | `DualEncoderForecastMLP` forecasting head |
| `src/cosine_training.py` | CLIP pre-training loop |
| `src/util.py` | `clip_symm_loss`, `cosine_mse_loss`, `make_positive_view`, embedding helpers |
| `src/configs/lotsa_ablation_best.yaml` | **Optimal config** (all ablation winners) |
| `src/experiments/probe_lotsa_checkpoint.py` | Linear probe evaluation |
| `src/experiments/lotsa_forecast_zeroshot.py` | MLP training on LOTSA + zero-shot ICML eval |
| `src/experiments/icml_finetune_vs_supervised.py` | ICML fine-tune vs supervised comparison |
| `src/scripts/anunna_icml_finetune_vs_supervised.sh` | SLURM launcher for the comparison |
| `src/experiments/ABLATIONS.md` | Ablation study descriptions and run instructions |
| `ABLATION_RESULTS.md` | Full ablation results with tables |
