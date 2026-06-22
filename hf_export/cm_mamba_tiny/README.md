---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: time-series-forecasting
tags:
- time-series
- mamba
- ssm
- zero-shot
- forecasting
---

# lstival/cm-mamba-tiny

**CM-Mamba** is a compact dual-encoder state-space model for zero-shot time series forecasting. The temporal encoder processes raw time series; the visual encoder processes recurrence-plot representations. A Mixture-of-Prompts (MoP) head combines both views for multi-horizon forecasting.

## Performance

| Benchmark | NRMSE | Notes |
|-----------|-------|-------|
| **GIFT-Eval** (39 subsets) | **0.4075** | Zero-shot, official evaluation |
| **TSLib / ICML-7** (ETT/weather/traffic/electricity) | **0.4743** | Zero-shot, H∈{96,192,336,720} |

Encoder: `simclr_enc_F_lasttoken` (SimCLR + last-token pooling, trained leak-free).

## Quick Start

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("lstival/cm-mamba-tiny", trust_remote_code=True)
model.eval()

# x: (batch, context_length=336, channels)
x = torch.randn(2, 336, 7)
with torch.no_grad():
    preds = model(x, horizon=96)   # → (2, 96, 7)

print(preds.shape)  # torch.Size([2, 96, 7])
```

Supported horizons: **96, 192, 336, 720**. Context length: **336** timesteps.

## Load Encoders Only

```python
model = AutoModel.from_pretrained("lstival/cm-mamba-tiny", trust_remote_code=True)
temporal_enc, visual_enc = model.get_encoder()
# temporal_enc: MambaEncoder → (B, emb_dim=96) embeddings
# visual_enc:   MambaVisualEncoder → (B, emb_dim=96) embeddings
```

## Architecture

- **Temporal encoder**: 5-layer Mamba SSM, model_dim=192, emb_dim=96, last-token pooling
- **Visual encoder**: same depth/dim, operates on recurrence-plot patches
- **MoP head**: 16 learnable prompts routing concat embeddings (dim=192) → hidden_dim=512
- **Skip head**: DLinear-style Linear(336→H) residual per horizon
- **Normalisation**: RevIN (instance normalisation, denormalised before output)
- **Channel mode**: Channel-independent (CI) — each channel processed separately

## How to Re-export / Update HuggingFace Weights

After training a new SSL encoder and MoP head:

```bash
# 1. Run the export script (uses the timeseries conda env)
/lustre/nobackup/WUR/AIN/stiva001/timeseries/bin/python3 \
    scripts/export_cm_mamba_mop_to_hf.py

# 2. Push to HuggingFace
huggingface-cli login
huggingface-cli upload lstival/cm-mamba-tiny hf_export/cm_mamba_tiny .
```

To swap checkpoints, edit `MOP_CKPT` at the top of `scripts/export_cm_mamba_mop_to_hf.py`.

## Citation

If you use this model, please cite the CM-Mamba paper or repository.
