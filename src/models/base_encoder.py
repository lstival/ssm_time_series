"""Shared base encoder for Mamba-style encoders.

Provides:
- Mamba block stack
- pooling
- output projection
- parameter counting helper

Concrete encoders implement `forward_sequence()`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

try:
    from mamba_block import MambaBlock
except Exception:
    from .mamba_block import MambaBlock

Pooling = Literal["mean", "last", "cls"]


class BaseMambaEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        embedding_dim: int,
        depth: int,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        pooling: Pooling = "mean",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.model_dim = int(model_dim)
        self.embedding_dim = int(embedding_dim)
        self.pooling: Pooling = pooling

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    self.model_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.output_proj = nn.Linear(self.model_dim, self.embedding_dim, bias=False)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_sequence(x)
        pooled = self._pool_sequence(hidden)
        return self.output_proj(pooled)

    def _pool_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("Expected hidden of shape (B, S, D)")

        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "last":
            return hidden[:, -1, :]
        if self.pooling == "cls":
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    def count_parameters(self, trainable_only: bool = True) -> int:
        try:
            from torch.nn.parameter import UninitializedParameter  # type: ignore
        except Exception:  # pragma: no cover
            UninitializedParameter = ()  # type: ignore[assignment]

        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        total = 0
        for p in params:
            if isinstance(p, UninitializedParameter):
                continue
            total += p.numel()
        return total
