"""Mamba block for HuggingFace CM-Mamba package.

Always uses the pure-PyTorch naive implementation so the model runs on CPU
without requiring CUDA or the mamba-ssm package.
"""

from __future__ import annotations

from ._naive_mamba_block import MambaBlock

__all__ = ["MambaBlock"]
