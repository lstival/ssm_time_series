---
license: apache-2.0
library_name: transformers
pipeline_tag: time-series-forecasting
---

# CM-Mamba: Mamba-based Multimodal Contrastive Learning for Time Series Forecast

This repository contains two CM-Mamba forecasting variants exported to Hugging Face format:

- **cm-mamba-tiny** (depth=8)
- **cm-mamba-mini** (depth=16)

Each subfolder contains `config.json`, `pytorch_model.bin`, and a model card with usage examples.

## Model description

CM-Mamba is a compact state-space model (SSM) encoder for time-series forecasting. It tokenizes the input sequence into fixed windows, applies stacked Mamba-style SSM blocks to capture long-range dependencies, and pools the sequence into a fixed embedding. A lightweight multi-horizon MLP head predicts future values in a single forward pass.

**Dual-encoder variant**: CM-Mamba Tiny and Mini combine a temporal encoder with a visual encoder that uses recurrence plots. The final embedding is the concatenation of both encoders, which improves robustness across datasets.

**Paper status**: This work is submitted to ICML 2026.

## Quick start

```python
import torch
from ssm_time_series.hf.forecasting import CM_MambaForecastModel

model = CM_MambaForecastModel.from_pretrained("lstival/CM-Mamba", subfolder="cm-mamba-tiny", trust_remote_code=True)
model.eval()

x = torch.randn(2, 256, 32)  # [B, T, F]
with torch.no_grad():
    y = model(x)
print(y.shape)
```

## Detailed forecasting example

```python
import torch
from ssm_time_series.hf.forecasting import CM_MambaForecastModel

# Load the Tiny variant
model = CM_MambaForecastModel.from_pretrained(
    "lstival/CM-Mamba",
    subfolder="cm-mamba-tiny",
    trust_remote_code=True,
)
model.eval()

# Input: [B, T, F]
x = torch.randn(4, 256, 32)

# Full horizon prediction
with torch.no_grad():
    y = model(x)

print(y.shape)  # [B, H, C]

# If you want only the first 96 steps
with torch.no_grad():
    y_96 = model(x, horizon=96)
print(y_96.shape)  # [B, 96, C]
```

## Feature extraction

You can extract embeddings from the encoders directly:

```python
import torch
from ssm_time_series.hf.forecasting import CM_MambaForecastModel

model = CM_MambaForecastModel.from_pretrained(
    "lstival/CM-Mamba",
    subfolder="cm-mamba-mini",
    trust_remote_code=True,
)

temporal_encoder, visual_encoder = model.get_encoder_only()

x = torch.randn(2, 256, 32)
with torch.no_grad():
    t_emb = temporal_encoder(x)
    v_emb = visual_encoder(x)

print(t_emb.shape, v_emb.shape)
```

## Available variants

- `cm-mamba-tiny`
- `cm-mamba-mini`

## Results (Markdown table)
On zero-shot forecasting across 9 popular benchmarks, CM-Mamba variants achieve strong performance compared to recent baselines.
**Note**: **bold** = best, <u>underline</u> = second best.

| Metric | CM-Mamba-mini (MSE) | CM-Mamba-mini (MAE) | CM-Mamba-tiny (MSE) | CM-Mamba-tiny (MAE) | S-Mamba (MSE) | S-Mamba (MAE) | LightGTS (MSE) | LightGTS (MAE) | Timer (MSE) | Timer (MAE) | MOIRAI (MSE) | MOIRAI (MAE) | Chronos (MSE) | Chronos (MAE) | TimesFM (MSE) | TimesFM (MAE) | Time-MoE (MSE) | Time-MoE (MAE) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | **0.192** | **0.256** | <u>0.192</u> | <u>0.260</u> | 0.398 | 0.405 | 0.327 | 0.370 | 0.768 | 0.568 | 0.390 | 0.389 | 0.551 | 0.453 | 0.435 | 0.418 | 0.376 | 0.406 |
| ETTm2 | **0.198** | **0.281** | <u>0.199</u> | <u>0.283</u> | 0.288 | 0.332 | 0.247 | 0.316 | 0.315 | 0.356 | 0.276 | 0.320 | 0.293 | 0.331 | 0.347 | 0.360 | 0.315 | 0.365 |
| ETTh1 | <u>0.196</u> | **0.272** | **0.196** | <u>0.274</u> | 0.455 | 0.450 | 0.388 | 0.419 | 0.562 | 0.483 | 0.510 | 0.469 | 0.533 | 0.452 | 0.479 | 0.442 | 0.394 | 0.420 |
| ETTh2 | **0.195** | **0.276** | <u>0.195</u> | <u>0.277</u> | 0.381 | 0.405 | 0.348 | 0.395 | 0.370 | 0.400 | 0.354 | 0.377 | 0.392 | 0.397 | 0.400 | 0.403 | 0.403 | 0.415 |
| Traffic | <u>0.314</u> | 0.484 | **0.240** | 0.399 | 0.414 | **0.276** | 0.561 | <u>0.381</u> | 0.613 | 0.407 | - | - | 0.615 | 0.421 | - | - | - | - |
| Weather | <u>0.250</u> | 0.385 | 0.263 | 0.397 | 0.251 | 0.276 | **0.208** | **0.256** | 0.292 | 0.313 | 0.260 | <u>0.275</u> | 0.288 | 0.309 | - | - | 0.270 | 0.300 |
| Exchange | <u>0.275</u> | 0.433 | **0.261** | <u>0.405</u> | 0.367 | 0.408 | 0.347 | **0.396** | 0.392 | 0.425 | 0.385 | 0.417 | 0.370 | 0.412 | 0.390 | 0.417 | 0.432 | 0.454 |
| Solar | 0.311 | 0.497 | 0.255 | 0.394 | <u>0.240</u> | <u>0.273</u> | **0.191** | **0.271** | 0.771 | 0.604 | 0.714 | 0.704 | 0.393 | 0.319 | 0.500 | 0.397 | 0.411 | 0.428 |
| Electricity | 0.191 | **0.264** | 0.192 | 0.267 | **0.170** | <u>0.265</u> | 0.213 | 0.308 | 0.297 | 0.375 | <u>0.188</u> | 0.273 | - | - | - | - | - | - |


## Citation

```bibtex
@inproceedings{cm_mamba_2026,
    title={CM-Mamba: Compact State-Space Encoders for Time-Series Forecasting},
    author={Leandro Stival, Ricardo da Silva Torres},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2026},
    note={Submitted}
}
```
