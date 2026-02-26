"""Platform-aware Mamba block.

Selects the best available SSM implementation at import time:

* **Linux / macOS** with ``mamba-ssm`` installed → wraps the CUDA-optimised
  ``mamba_ssm.Mamba`` kernel from https://github.com/state-spaces/mamba.
* **Windows** (or any platform without ``mamba-ssm``) → falls back to the
  pure-PyTorch implementation in :mod:`_naive_mamba_block`.

Both paths expose an identical public interface::

    MambaBlock(d_model, *, state_dim, conv_kernel, expand_factor, dropout)

so the rest of the codebase never has to branch on the platform.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _mamba_ssm_available() -> bool:
    """Return True only on Unix platforms where mamba-ssm can be imported."""
    if sys.platform in ("win32", "cygwin"):
        return False
    try:
        import mamba_ssm  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Official mamba-ssm wrapper  (Linux / macOS with mamba-ssm installed)
# ---------------------------------------------------------------------------

if _mamba_ssm_available():
    from mamba_ssm import Mamba as _MambaCore

    class MambaBlock(nn.Module):
        """Mamba block backed by the official CUDA-optimised ``mamba-ssm`` library.

        Adds a pre-normalisation layer and a residual connection around
        ``mamba_ssm.Mamba`` so the interface is identical to the naive version.

        Parameters
        ----------
        d_model:
            Number of channels after the input projection.
        state_dim:
            Internal SSM state dimension (``d_state`` in mamba-ssm).
        conv_kernel:
            Depthwise convolution width (``d_conv`` in mamba-ssm).
        expand_factor:
            Channel expansion ratio (rounded to the nearest integer for
            mamba-ssm's ``expand`` parameter).
        dropout:
            Dropout probability applied after the Mamba sub-layer.
        input_dim:
            Unused – accepted for API compatibility with the naive block.
        """

        def __init__(
            self,
            d_model: int = 128,
            *,
            state_dim: int = 16,
            conv_kernel: int = 4,
            expand_factor: float = 2.0,
            dropout: float = 0.0,
            input_dim: int = 32,  # API compat – not used internally
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


# ---------------------------------------------------------------------------
# Naive pure-PyTorch fallback  (Windows or mamba-ssm not installed)
# ---------------------------------------------------------------------------

else:
    try:
        from _naive_mamba_block import MambaBlock  # noqa: F401  (direct execution)
    except ImportError:
        from ._naive_mamba_block import MambaBlock  # noqa: F401  (package import)