"""Lightweight Mamba-style encoder for 384-dim time series inputs.

This module implements a compact encoder inspired by the Mamba state-space
architecture (https://arxiv.org/abs/2312.00752). It is designed to keep the
parameter count low while still capturing temporal structure in fixed-length
(time-major) sequences. The core building block is a simplified selective scan
that mixes information along the sequence using inexpensive operations.

Example
-------
>>> import torch
>>> from .mamba_encoder import MambaEncoder, tokenize_sequence
>>> encoder = MambaEncoder()
>>> x = torch.randn(8, 128, 384)  # (batch, time, features)
>>> # create non-overlapping tokens of length 16 (result: 8 tokens)
>>> xt = tokenize_sequence(x, token_size=16, stride=None, method="mean")
>>> embedding = encoder(xt)        # encoder expects (B, seq, features)
>>> embedding.shape
torch.Size([8, 8, 128])  # if pooling='mean' / projection dims, etc.
"""

from __future__ import annotations
from typing import Literal, Optional
import math
import torch
from torch import nn
from .mamba_block import MambaBlock

Pooling = Literal["mean", "last", "cls"]


class Tokenizer:
    """
    Convert a (B, T, F) sequence into a sequence of tokens (B, N, F) by
    windowing along the time dimension.

    Usage:
        tk = Tokenizer(token_size=32, stride=None, method="values", pad=True)
        tokens = tk(x)  # where x is (B, T, F) or (B, F, T) — note: this implementation
                        # will swap axes at the start like the original function.

    Parameters
    - token_size: length of each token window (number of timesteps per token)
    - stride: step between token starts. If None, defaults to token_size (non-overlap).
    - method: how to aggregate values inside a token: "mean", "max", "first", or
              "values". "values" returns the raw window values with shape
              (batch, n_tokens, token_size, features).
    - pad: whether to pad the end with zeros so the final token is full length.
    """

    def __init__(
        self,
        *,
        token_size: int = 32,
        stride: Optional[int] = None,
        method: Literal["mean", "max", "first", "values"] = "values",
        pad: bool = True,
    ) -> None:
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        if method not in ("mean", "max", "first", "values"):
            raise ValueError(f"Unknown tokenization method: {method}")

        self.token_size = token_size
        self.stride = stride if stride is not None else token_size
        self.method = method
        self.pad = pad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Standard input: (batch, time, features)
        if x.ndim != 3:
            raise ValueError("Tokenizer expects input of shape (batch, time, features)")

        B, T, F = x.shape
        token_size = self.token_size
        stride = self.stride

        # compute required padding so that unfold will include the final (possibly partial) block
        if T <= token_size:
            n_steps = 1
            required_total = token_size
        else:
            n_steps = math.ceil((T - token_size) / stride) + 1
            required_total = token_size + stride * (n_steps - 1)
        pad_len = max(0, required_total - T)

        if pad_len > 0:
            if self.pad:
                pad_tensor = x.new_zeros((B, pad_len, F))
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                raise ValueError(
                    "Sequence length is shorter than token_size and pad=False; cannot form full tokens"
                )
        else:
            x_padded = x

        # Create the patches: (B, n_tokens, F, token_size)
        # We permute to (B, n_tokens, token_size, F) so that aggregation/reshape works as expected
        patches = x_padded.unfold(dimension=1, size=token_size, step=stride).contiguous()
        patches = patches.permute(0, 1, 3, 2)
        # patches is (B, n_tokens, token_size, F)

        if self.method == "mean":
            tokens = patches.mean(dim=2)
        elif self.method == "max":
            tokens, _ = patches.max(dim=2)
        elif self.method == "first":
            tokens = patches[:, :, 0, :]
        elif self.method == "values":
            tokens = patches
        else:
            # should be unreachable due to check in __init__
            raise ValueError(f"Unknown tokenization method: {self.method}")

        return tokens


def tokenize_sequence(
    x: torch.Tensor,
    *,
    token_size: int = 32,
    stride: Optional[int] = None,
    method: Literal["mean", "max", "first", "values"] = "values",
    pad: bool = True,
) -> torch.Tensor:
    return Tokenizer(token_size=token_size, stride=stride, method=method, pad=pad)(x)


class MambaEncoder(nn.Module):
    """Compact Mamba-style encoder for 384-dim time series inputs."""

    def __init__(
        self,
        *,
        input_dim: int = 128,  # Feature dimension (F)
        token_size: int = 32,  # Window size (T_token)
        model_dim: int = 768,
        depth: int = 6,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        embedding_dim: int = 128,
        pooling: Pooling = "mean",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if token_size <= 0:
            raise ValueError("token_size must be positive")

        self.input_dim = input_dim
        self.token_size = token_size
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.pooling: Pooling = pooling

        self.input_proj = nn.Linear(input_dim, model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    model_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a fixed-size embedding for a `(batch, seq, features)` tensor."""
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, seq, features)")
        seq_len = x.size(1)
        if seq_len < self.token_size:
            raise ValueError(
                f"Sequence length ({seq_len}) must be at least as large as the encoder token size ({self.token_size})."
            )
        # if x.size(-1) != self.input_dim:
        #     raise ValueError(f"Expected final dimension {self.input_dim}, got {x.size(-1)}")

        features = self.forward_sequence(x)
        pooled = self._pool_sequence(features, x)
        return self.output_proj(pooled)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        tokens = self.tokenizer(x)
        if tokens.ndim == 4:
            batch, windows, window_len, feat_dim = tokens.shape
            # Map patches to features: (batch, windows, window_len * feat_dim)
            tokens = tokens.reshape(batch, windows, window_len * feat_dim)
        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x):
        # We switch to 'mean' aggregation to match the trained weights which expect input_dim features per token
        tokens = tokenize_sequence(x, token_size=self.token_size, method="mean")
        return tokens

    def _pool_sequence(self, hidden: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "last":
            return hidden[:, -1, :]
        if self.pooling == "cls":
            # Expect the first timestep to act as a CLS token; fall back to mean if seq len == 1
            if hidden.size(1) == 1:
                return hidden[:, 0, :]
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)


def create_default_encoder(**overrides: object) -> MambaEncoder:
    """Factory that mirrors the defaults while allowing overrides."""
    return MambaEncoder(**overrides)


if __name__ == "__main__":
    feat_dim = 384
    tokens_dim = 32
    encoder = MambaEncoder(
        depth=6,
        input_dim=feat_dim,
        token_size=tokens_dim,
        pooling="mean",
        model_dim=128,
        embedding_dim=128,
        expand_factor=1.5
    )
    
    print(f"Trainable parameters: {encoder.count_parameters():,}")
    dummy = torch.randn(4, 96, 384)
    out = encoder(dummy)
    print("Output embedding shape:", out.shape)
    tokens = tokenize_sequence(dummy, token_size=tokens_dim)
    print("Output tokens shape:", tokens.shape)
