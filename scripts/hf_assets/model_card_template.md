---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: time-series-forecasting
---

# {{MODEL_ID}}

CM-Mamba is a compact Mamba-style state-space model for time series forecasting. This release provides a dual-encoder setup (temporal + visual) with a multi-horizon forecasting head.

## Model Details
- **Model Type**: Dual-encoder (temporal + visual) + multi-horizon forecasting head
- **Architecture**: Mamba-style SSM encoder with lightweight forecasting MLP
- **Intended Use**: Short to medium horizon forecasting for multivariate time series

## Usage

### Load the forecasting model

```python
import torch
from cm_mamba.hf.forecasting import CM_MambaForecastModel

model = CM_MambaForecastModel.from_pretrained("{{MODEL_ID}}", trust_remote_code=True)
model.eval()

x = torch.randn(2, 256, 32)  # [B, T, F]
with torch.no_grad():
    y = model(x)
print(y.shape)  # [B, H, C]
```

### Load encoders only

```python
model = CM_MambaForecastModel.from_pretrained("{{MODEL_ID}}", trust_remote_code=True)
encoders = model.get_encoder_only()
```

## Training Details
- **Checkpoint source**: {{CHECKPOINT_SOURCE}}
- **Horizons**: {{HORIZONS}}
- **Target features**: {{TARGET_FEATURES}}

## Limitations
- Forecasting accuracy depends on domain similarity and data preprocessing.
- The visual encoder uses recurrence plots, which can be sensitive to scaling.

## Citation
If you use this model, please cite the CM-Mamba paper or repository.
