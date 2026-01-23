# CM-Mamba: Contrastive Mamba for Time Series

Official implementation for CM-Mamba, a high-performance architecture for time-series forecasting and feature extraction using Mamba blocks.

## Features
- **Temporal Encoder**: Mamba-based sequence modeling for 1D time series.
- **Visual Encoder**: Multi-scale visual representation using recurrence plots and Mamba.
- **Dual Encoder**: Hybrid architecture combining temporal and visual features.
- **Contrastive Learning**: Pre-training utilities using InfoNCE loss.
- **Hugging Face Integration**: Load models directly via `AutoModel.from_pretrained`.

## Installation

```bash
pip install -r requirements.txt
# Or install in editable mode
pip install -e .
```

## Quick Start

### Standalone Inference
See `examples/inference_example.py` for a full demonstration of temporal, visual, and dual encoder inference.

### Contrastive Pre-training
The model supports self-supervised pre-training. An example setup is provided in `examples/training_contrastive_example.py`.

### Custom Datasets
To use your own data, use the `SimpleTimeSeriesDataModule` provided in `cm_mamba.data.loader`.
Example: `examples/custom_dataset_example.py`.

## Hugging Face Hub
You can also load pre-trained weights from the Hugging Face Hub:

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("lstival/CM-Mamba", subfolder="cm-mamba-tiny", trust_remote_code=True)
x = torch.randn(1, 96, 32)
y = model(x)
print(y.shape) # [1, 720, 1]
```

## Repository Structure
- `src/cm_mamba/`: Core package source.
  - `models/`: Mamba and Encoder implementations.
  - `training/`: Training loops and losses.
  - `data/`: Data loading utilities.
  - `hf/`: Hugging Face interface.
- `examples/`: Guided scripts for different use cases.
