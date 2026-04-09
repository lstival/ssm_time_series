# CM-Mamba Ablation Studies (ICML Revision)

Six ablation studies addressing reviewer requests on the RP definition, architecture
justification, alignment strategy, visual representations, patch length, and manifold
quality on unseen data.

All scripts follow the same protocol: **CLIP pre-training on LOTSA → frozen linear
probe on downstream datasets → CSV results**.

---

## Ablation A — Multivariate RP Definition

**Script:** `ablation_A_mv_rp.py`
**SLURM:** `src/scripts/anunna_ablations_A.sh`
**Results:** `results/ablation_A/ablation_A_results.csv`

Compares four strategies for aggregating a Recurrence Plot across channels in a
multivariate time series patch:

| Strategy | Description |
|---|---|
| `per_channel` | RP computed independently per channel, then averaged **(default)** |
| `mean` | Channels averaged first → single-channel RP |
| `pca` | 1-component PCA projection → single-channel RP |
| `joint` | Multivariate RP using L2 distance in feature-dim state space |

**Evaluation datasets:** ETTm1, Weather, Traffic
**Horizons:** 96 / 192 / 336 / 720

```bash
python src/experiments/ablation_A_mv_rp.py \
    --config src/configs/lotsa_clip.yaml \
    --train_epochs 20 \
    --probe_epochs 30 \
    --results_dir results/ablation_A
```

---

## Ablation B — Visual Encoder Architecture

**Script:** `ablation_B_encoder_arch.py`
**SLURM:** `src/scripts/anunna_ablations_B.sh`
**Results:** `results/ablation_B/ablation_B_results.csv`

Justifies the use of a separate Mamba-based visual encoder by comparing four
architecture variants:

| Variant | Description |
|---|---|
| `no_visual` | Temporal encoder only; no visual branch |
| `shared_1d` | Single shared MambaEncoder for both temporal and visual branches |
| `sep_cnn_only` | Separate visual branch with CNN projection, no Mamba SSM blocks |
| `sep_mamba_1d` | Full architecture: separate Mamba visual encoder **(current)** |

`CNNOnlyVisualEncoder` is defined inline in the script (no Mamba SSM, Conv2D +
mean pooling only).

**Evaluation datasets:** ETTm1, ETTh1, Weather
**Horizons:** 96 / 192 / 336 / 720

```bash
python src/experiments/ablation_B_encoder_arch.py \
    --config src/configs/lotsa_clip.yaml \
    --train_epochs 20 \
    --probe_epochs 30 \
    --results_dir results/ablation_B
```

---

## Ablation C — Contrastive Alignment Strategy

**Script:** `ablation_C_alignment.py`
**SLURM:** `src/scripts/anunna_ablations_C.sh`
**Results:** `results/ablation_C/ablation_C_results.csv`

Justifies the CLIP-style symmetric NT-Xent loss by comparing four alignment
strategies:

| Variant | Description |
|---|---|
| `clip_symm` | Symmetric NT-Xent CLIP loss **(current)** |
| `cosine_mse` | MSE between L2-normalised temporal and visual embeddings |
| `concat_supervised` | Concatenated embeddings + supervised MLP head (no SSL) |
| `unimodal_temporal` | Temporal encoder only (lower bound) |

`clip_symm` must outperform `cosine_mse` to justify the contrastive formulation.
`concat_supervised` is the dangerous baseline — if it wins, the SSL contribution is
in question.

**Evaluation datasets:** ETTm1, Weather, Exchange
**Horizons:** 96 / 192 / 336 / 720

```bash
python src/experiments/ablation_C_alignment.py \
    --config src/configs/lotsa_clip.yaml \
    --train_epochs 20 \
    --probe_epochs 30 \
    --results_dir results/ablation_C
```

---

## Ablation D — Visual Representation Types

**Script:** `ablation_D_visual_repr.py`
**SLURM:** `src/scripts/anunna_ablations_D.sh`
**Results:** `results/ablation_D/ablation_D_results.csv`

Extends Table 5 (originally ETTh1-only) to five datasets and adds a computational
cost column for Pareto analysis. The claim is that RP is Pareto-dominant (lower or
equal MSE, lower cost) on at least 4/5 datasets.

| Representation | Description |
|---|---|
| `rp` | Recurrence Plot **(baseline)** |
| `gasf` | Gramian Angular Summation Field |
| `mtf` | Markov Transition Field |
| `stft` | Short-Time Fourier Transform spectrogram |

**Evaluation datasets:** ETTh1, ETTm1, Weather, Traffic, Solar
**Horizon:** 96 (primary) + ms/batch cost column

```bash
python src/experiments/ablation_D_visual_repr.py \
    --config src/configs/lotsa_clip.yaml \
    --train_epochs 20 \
    --probe_epochs 30 \
    --results_dir results/ablation_D
```

---

## Ablation E — Patch Length Impact

**Script:** `ablation_E_patch_length.py`
**SLURM:** `src/scripts/anunna_ablations_E.sh`
**Results:** `results/ablation_E/ablation_E_results.csv`

Evaluates how the token (patch) size `l` affects forecast quality and RP
informational content. Sweeps `l ∈ {16, 32, 64, 96, 128}` and detects the
sweet-spot where the RP is informationally rich and the token sequence is long
enough for the SSM.

**Evaluation datasets:** ETTm1 (periodic), Weather (irregular), Traffic (multi-scale)
**Horizons:** 96 / 192 / 336 / 720
**Output columns:** patch_size, dataset, horizon, MSE, MAE, ms/batch

```bash
python src/experiments/ablation_E_patch_length.py \
    --config src/configs/lotsa_clip.yaml \
    --train_epochs 20 \
    --probe_epochs 30 \
    --patch_sizes 16 32 64 96 128 \
    --results_dir results/ablation_E
```

---

## Ablation F — Manifold Quality on Unseen Data

**Script:** `ablation_F_manifold.py`
**SLURM:** `src/scripts/anunna_ablations_F.sh`
**Results:** `results/ablation_F/manifold_metrics.csv` + t-SNE PDF

Refactors Table 4: original used training-data embeddings (potential memorisation).
This re-runs the manifold analysis on **evaluation datasets never seen during
pre-training**, using a frozen checkpoint.

Three encoder modes compared on the same checkpoint:

| Mode | Description |
|---|---|
| `temporal_only` | MambaEncoder embeddings |
| `visual_only` | MambaVisualEncoder embeddings |
| `multimodal` | Concatenation `[temporal ‖ visual]` |

**Evaluation datasets:** Electricity, Solar, Weather, Exchange
**Outputs:** t-SNE plot (PDF/PNG) coloured by dataset + CSV with Silhouette score,
Davies-Bouldin index, Cohesion/Separation ratio per mode.

If clusters discriminate on unseen data → evidence of transfer.
If they collapse → fundamental limitation (report in Limitations section).

```bash
python src/experiments/ablation_F_manifold.py \
    --checkpoint_dir checkpoints/ts_encoder_lotsa \
    --config src/configs/lotsa_clip.yaml \
    --data_dir ICML_datasets \
    --results_dir results/ablation_F
```

> **Note:** `--checkpoint_dir` must point to an existing checkpoint trained on LOTSA.

---

## Ablation H — Visual Encoder Architecture for RP

**Script:** `ablation_H_visual_encoder_arch.py`
**SLURM:** `src/scripts/anunna_ablations_H.sh`
**Results:** `results/ablation_H/ablation_H_results.csv`

Determines which architecture processes the 2D structure of Recurrence Plots
best for small patch sizes l ∈ {16, 32, 64}.

| Variant | Inductive Bias |
|---|---|
| `cnn` | Local spatial filters (baseline) |
| `flatten_mamba` | Baseline destruction of 2D structure |
| `rp_ss2d_2` | Semantic 2-D dual scan (Proposed) |
| `ss2d_4` | Full VMamba-style 4-scan |

**Evaluation datasets:** ETTm1, Weather, Traffic
**Horizon:** 96 (primary ablation)
**Metrics:** MSE, MAE, Speed (ms/batch), Memory (MB), Parameters

```bash
python src/experiments/ablation_H_visual_encoder_arch.py \
    --config src/configs/lotsa_clip.yaml \
    --train_epochs 20 \
    --probe_epochs 30 \
    --results_dir results/ablation_H
```

---

## Findings Summary (April 2026)

Results for completed ablations. See `ABLATION_RESULTS.md` for full analysis and `ABLATION_LATEX_APPENDIX.md` for paper-ready LaTeX tables.

| Ablation | Selected setting | Key insight |
|---|---|---|
| A | `mean` (best MSE −11.9% vs default) | `joint` is faster but sacrifices accuracy |
| B | `sep_mamba_1d` | CNN-only visual branch collapses (36× worse); shared encoder worse than no visual branch |
| C | `cosine_mse` | `clip_symm` fails 2.7× on non-stationary data (exchange rate) |
| D | `rp` (pending re-run) | CUDA error fixed — smoke test submitted |
| E | patch `64` (best MSE −15% vs 32) | patch `16` collapses (18× worse on ETTm1) |
| F | `multimodal` | Best Davies-Bouldin + separation on unseen datasets |
| G | `multimodal` | More consistent than temporal-only across datasets |

Optimal config: **`src/configs/lotsa_ablation_best.yaml`** — SLURM job 66150812 (encoder) + 66150813 (probe, dependent).

---

## Running All Ablations (SLURM)

After LOTSA pre-training is complete, submit all eight jobs:

```bash
sbatch src/scripts/anunna_ablations_A.sh
sbatch src/scripts/anunna_ablations_B.sh
sbatch src/scripts/anunna_ablations_C.sh
sbatch src/scripts/anunna_ablations_D.sh
sbatch src/scripts/anunna_ablations_E.sh
sbatch src/scripts/anunna_ablations_F.sh
sbatch src/scripts/anunna_ablations_G.sh
sbatch src/scripts/anunna_ablations_H.sh
```

> A–E, G, H are independent and can run in parallel.
> F requires `--checkpoint_dir` pointing to a finished LOTSA checkpoint.

## Common Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `src/configs/lotsa_clip.yaml` | YAML config for model and training |
| `--train_epochs` | 20 | CLIP pre-training epochs per variant |
| `--probe_epochs` | 30 | Linear probe training epochs |
| `--data_dir` | `ICML_datasets/` | Directory with downstream CSV datasets |
| `--results_dir` | `results/ablation_X/` | Output directory for CSV + plots |
| `--seed` | 42 | Random seed |
