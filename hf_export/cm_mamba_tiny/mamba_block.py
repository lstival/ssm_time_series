"""Platform-aware Mamba block.

Selects the best available SSM implementation at import time:

* **Linux / macOS** with ``mamba-ssm`` installed → wraps the CUDA-optimised
  ``mamba_ssm.Mamba`` kernel from https://github.com/state-spaces/mamba.
* **Windows** (or any platform without ``mamba-ssm``) → falls back to the
  pure-PyTorch implementation in :mod:`_naive_mamba_block`.

Both paths expose an identical public interface::

    MambaBlock(d_model, *, state_dim, conv_kernel, expand_factor, dropout)
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn


def _mamba_ssm_available() -> bool:
    """Return True only on Unix platforms where mamba-ssm can be imported."""
    if sys.platform in ("win32", "cygwin"):
        return False
    try:
        import mamba_ssm  # noqa: F401
        return True
    except ImportError:
        return False


if _mamba_ssm_available():
    from mamba_ssm import Mamba as _MambaCore

    class MambaBlock(nn.Module):
        """Mamba block backed by the official CUDA-optimised ``mamba-ssm`` library."""

        def __init__(
            self,
            d_model: int = 128,
            *,
            state_dim: int = 16,
            conv_kernel: int = 4,
            expand_factor: float = 2.0,
            dropout: float = 0.0,
            input_dim: int = 32,
        ) -> None:
            super().__init__()
            if d_model <= 0:
                raise ValueError("d_model must be positive")
            self.d_model = d_model
            self.norm = nn.LayerNorm(d_model)
            self.mamba = _MambaCore(
                d_model=d_model,
                d_state=state_dim,
                d_conv=max(1, conv_kernel),
                expand=max(1, round(expand_factor)),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the block to a ``(batch, seq, channels)`` tensor."""
            if x.ndim != 3:
                raise ValueError("Expected input of shape (batch, seq, channels)")
            return x + self.dropout(self.mamba(self.norm(x)))

else:
    from ._naive_mamba_block import MambaBlock  # noqa: F401

try:
    from ._naive_mamba_block import hippo_legS_matrix  # noqa: F401
except ImportError:
    pass
