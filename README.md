# Lightweight Mamba Encoder

This repo contains a compact PyTorch implementation of a Mamba-style encoder for
384-dimensional time-series inputs. The goal is to provide a parameter-efficient
module that converts an input sequence into a fixed-size embedding.

## Features

- Drop-in `MambaEncoder` module built with PyTorch only
- Simplified selective scan core for efficient temporal mixing
- Configurable depth, hidden size, and embedding dimension
- Minimal unit tests to validate shapes and error handling

## Quickstart

1. Install dependencies (PyTorch 2.1+ recommended):

   ```powershell
   pip install torch
   ```

2. Run the sample script:

   ```powershell
   python .\src\mamba_encoder.py
   ```

3. Execute the unit tests:

   ```powershell
   python -m unittest tests.test_mamba_encoder
   ```

## Usage in Your Project

```python
import torch
from mamba_encoder import MambaEncoder

encoder = MambaEncoder(model_dim=128, depth=2, embedding_dim=128)
sequence = torch.randn(8, 120, 384)  # (batch, time, features)
embedding = encoder(sequence)
print(embedding.shape)  # torch.Size([8, 128])
```

Adjust the constructor arguments to trade off parameter count versus model
capacity. Use `encoder.count_parameters()` to inspect the trainable parameter
budget.
