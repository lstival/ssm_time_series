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

# Input shape: (batch_size, input_dim, sequence_length)
# Default input_dim is often 32 or 384 depending on configuration
x = torch.randn(8, 32, 96)
embeddings = model(x)
print(embeddings.shape)  # torch.Size([8, 128])
```

---

## 🏗️ Model Structure

The project features three main encoder types located in `ssm_time_series.models`:

1.  **Temporal Encoder (`MambaEncoder`)**: Processes multi-variate time series using a selective scan mechanism. It includes a built-in tokenizer for window-based aggregation.
2.  **Visual Encoder (`MambaVisualEncoder`)**: Interprets time series data through a visual lens (e.g., patched representations), utilizing a CLS-token approach for sequence-level embeddings.
3.  **Combined Encoder (`CM_MambaCombined`)**: A multi-modal wrapper that concatenates temporal and visual embeddings for a richer representation of the underlying dynamics.

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
> Config files are stored in `scripts/configs/`. You should adjust parameters like `learning_rate`, `batch_size`, and `model_dim` there.

---

## ⚙️ Project Architecture

```text
ssm_time_series/
├── scripts/            # Training and evaluation entry points
├── src/
│   └── ssm_time_series/
│       ├── data/       # Dataloaders and augmentation logic
│       ├── hf/         # Hugging Face from_pretrained interface
│       ├── models/     # Core Mamba architecture definitions
│       ├── training/   # Training loops and loss functions
│       └── utils/      # Modularized NN, Data, and Logging helpers
├── pyproject.toml      # Package configuration
└── README.md
```
