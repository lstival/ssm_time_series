# SSM Time Series Encoders

A modular, parameter-efficient framework for time series representation learning using Mamba-style state-space models. This project provides temporal and visual encoders designed for high-performance forecasting and representation learning.

---

## 🚀 Quick Start

### 1. Installation
The project is structured as a standard Python package. Install it in editable mode:
```bash
git clone <repo-url>
cd ssm_time_series
pip install -e .
```

### 2. Basic Usage (Importing)
Load a pre-trained model using the Hugging Face-style `from_pretrained` interface:

```python
import torch
from ssm_time_series.hf.cm_mamba import CM_MambaTemporal

# Load model from local directory or HF Hub
model = CM_MambaTemporal.from_pretrained("path/to/checkpoint")

# Input shape: (batch_size, sequence_length, features)
# Standard (B, T, F) format consistent with Transformers
x = torch.randn(8, 96, 32)
embeddings = model(x)
print(embeddings.shape)  # torch.Size([8, 128])
```

---

## 🤗 Hugging Face Hub

This project includes a full Hugging Face-compatible forecasting interface.

### Forecasting Model (Dual Encoder + Head)

```python
import torch
from ssm_time_series.hf.forecasting import CM_MambaForecastModel

model = CM_MambaForecastModel.from_pretrained("your-namespace/cm-mamba-tiny", trust_remote_code=True)
model.eval()

x = torch.randn(2, 256, 32)  # [B, T, F]
with torch.no_grad():
        preds = model(x)
print(preds.shape)  # [B, H, C]
```

### Encoder-Only Loading

```python
from pathlib import Path
from ssm_time_series.hf.forecasting import CM_MambaForecastConfig, CM_MambaForecastModel

config = CM_MambaForecastConfig.from_pretrained("your-namespace/cm-mamba-tiny", trust_remote_code=True)
encoders = CM_MambaForecastModel.from_checkpoint_encoder_only(
        config=config,
        checkpoint_path=Path("/path/to/best_model.pt"),
)
```

### Export to HF Format

```bash
python scripts/hf_export_cm_mamba.py \
    --config src/ssm_time_series/configs/hf/cm_mamba_tiny.yaml \
    --checkpoint /path/to/checkpoints/multi_horizon_forecast_dual_frozen_20251209_1049/all_datasets/best_model.pt \
    --output-dir /path/to/hf_export/cm-mamba-tiny
```

### Publish Step-by-Step

```bash
python scripts/hf_publish_step_by_step.py \
    --model-dir /path/to/hf_export/cm-mamba-tiny \
    --repo-id your-namespace/cm-mamba-tiny \
    --run-tests
```

---

## 📈 Evaluation & Visualization

The repository includes professional plotting tools for zero-shot forecasting results.

### Plotting Zero-Shot Forecasts
The script `plot_zeroshot_forecast.py` generates clean, publication-ready figures:

```bash
python src/ssm_time_series/evaluation/plot_zeroshot_forecast.py
```

**Key Features:**
- **Normalized vs. Real Values**: The plots show normalized data for consistency but use a "Reverse Transform" on the y-axis ticks to display real-world values.
- **Config-Driven**: Easily change datasets and model comparisons via YAML configs.
- **Publication Ready**: No titles (captions preferred), clean legend, and high-resolution output.

---

## 🏗️ Model Structure

The project features three main encoder types located in `ssm_time_series.models`:

1.  **Temporal Encoder (`MambaEncoder`)**: Processes multi-variate time series using a selective scan mechanism. It includes a built-in tokenizer for window-based aggregation.
2.  **Visual Encoder (`MambaVisualEncoder`)**: Interprets time series data through a visual lens (e.g., patched representations), utilizing a CLS-token approach for sequence-level embeddings.
3.  **Combined Encoder (`CM_MambaCombined`)**: A multi-modal wrapper that concatenates temporal and visual embeddings for a richer representation of the underlying dynamics.

---

## 🧠 How the Model Works

CM-Mamba is a compact state-space model (SSM) encoder designed for time-series data. It operates in three stages:

1. **Tokenization**: The input sequence $x \in \mathbb{R}^{B \times T \times F}$ is split into fixed-length windows (tokens) of size `token_size`. This yields $N$ tokens per series.
2. **SSM Encoding**: Each token is projected into a latent space and processed by stacked Mamba-style blocks that implement a selective scan, capturing long-range temporal dependencies efficiently.
3. **Pooling and Projection**: The sequence of hidden states is pooled (mean/last/cls) and projected to an embedding vector $z \in \mathbb{R}^{B \times D}$.

### Dual Encoder Variant

For CM-Mamba Tiny/Mini, the model uses **two encoders**:

- **Temporal Encoder**: Processes raw time-series tokens.
- **Visual Encoder**: Converts token windows into recurrence plots, then applies the same Mamba-style blocks.

The final embedding is the concatenation of both encoder outputs:

$$
z = [z_{\text{temporal}}; z_{\text{visual}}]
$$

### Forecasting Head

The forecasting model adds a lightweight MLP head that predicts a full horizon in one pass. Given embedding $z$, the head outputs:

$$
\hat{y} \in \mathbb{R}^{B \times H \times C}
$$

where $H$ is the maximum horizon and $C$ is the number of target features. A specific horizon can be selected by slicing the output.

---

## 📊 Dataloaders & Data Management

The data logic is centralized in `ssm_time_series.data`.

### Training Dataloader
Use the `TimeSeriesDataModule` to manage complex datasets (supporting `.csv`, `.npz`, and `.txt` formats):

```python
from ssm_time_series.data.loader import TimeSeriesDataModule

dm = TimeSeriesDataModule(
    data_dir="data/ICML_datasets",
    dataset_name="ETTm1.csv",
    batch_size=64,
    sample_size=96  # Context window size
)

# Get loaders for training, validation, and testing
loaders = dm.get_dataloaders()
for ds_loader in loaders:
    print(f"Dataset: {ds_loader.name}")
    # ds_loader.train, ds_loader.val, ds_loader.test
```

### Zero-Shot & Inference Support
For zero-shot evaluation or custom inference, use the `prepare_dataset` utility from `ssm_time_series.data.utils`:
```python
from ssm_time_series.data.utils import prepare_dataset

# Configuration-driven dataset loading
dataset = prepare_dataset(config_path="scripts/configs/mamba_encoder.yaml", config_data={"split": "test"})
```

---

## 🔧 Training the Model

Training scripts are located in the `scripts/` directory. They utilize YAML configurations for reproducibility.

### Supervised Forecasting
To train a model for supervised forecasting:
```bash
python scripts/chronos_supervised_training.py
```

### Contrastive Learning (Self-Supervised)
To pre-train an encoder using MoCo/Contrastive objectives:
```bash
python scripts/moco_training.py
```

> [!NOTE]
> Config files are stored in `src/ssm_time_series/configs/`. You should adjust parameters like `learning_rate`, `batch_size`, and `model_dim` there.

---

## 🧪 Evaluation & Benchmarking

The project includes a robust pipeline for zero-shot evaluation and benchmarking against standard datasets (ETT, PEMS, Weather, etc.).

### 1. Zero-Shot Forecasting
Evaluate trained models on unseen datasets using the zero-shot scripts in `src/ssm_time_series/evaluation/`:
```bash
# Standard evaluation
python src/ssm_time_series/evaluation/ICML_zeroshot_forecast.py --config src/ssm_time_series/configs/icml_zeroshot.yaml

# Dual-encoder (Temporal + Visual) evaluation
python src/ssm_time_series/evaluation/ICML_zeroshot_forecast_dual.py --config src/ssm_time_series/configs/icml_zeroshot_dual.yaml
```

### 2. Automated Reporting
Generate LaTeX tables for ICML-style results using scripts in `latex_generator/`:
```bash
python latex_generator/latex_multi_model_results.py
```

---

## 🎨 Visualization & Analytics

Powerful visualization tools are provided to interpret model performance and embedding quality.

### 1. Forecast Plotting
Generate forecast plots with a specialized dual-axis feature (plotting normalized values while displaying real-world scales):
```bash
python src/ssm_time_series/evaluation/plot_zeroshot_forecast.py
```
> [!TIP]
> This tool automatically identifies the best/worst samples and creates plots suitable for publication.

### 2. Embedding Visualization (t-SNE)
Analyze the latent space of the encoders:
```bash
# Generate t-SNE projections
python src/ssm_time_series/utils/visualization/tsne_clustering_metrics.py
```

---

## ⚙️ Project Architecture

```text
ssm_time_series/
├── latex_generator/    # LaTeX table generation for paper reporting
├── scripts/            # Main training entry points
├── src/
│   └── ssm_time_series/
│       ├── configs/    # Centralized YAML configurations
│       ├── data/       # Dataloaders and augmentation logic
│       ├── evaluation/ # Zero-shot pipelines and plotting
│       ├── hf/         # Hugging Face interface
│       ├── models/     # Mamba architecture definitions
│       └── utils/      # Visualization and NN helpers
├── pyproject.toml      # Package configuration
└── README.md
```

---
