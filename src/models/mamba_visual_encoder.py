"""Lightweight Mamba-style encoder for 384-dim time series inputs.

This module implements a compact encoder inspired by the Mamba state-space
architecture (https://arxiv.org/abs/2312.00752). It is designed to keep the
parameter count low while still capturing temporal structure in fixed-length
(time-major) sequences. The core building block is a simplified selective scan
that mixes information along the sequence using inexpensive operations.

"""

from __future__ import annotations
from typing import Literal, Optional
import math
import torch
from torch import nn

try:
    from utils import time_series_2_recurrence_plot
    from mamba_block import MambaBlock
except:
    from .utils import time_series_2_recurrence_plot
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
        # Swap axes from (B, F, T) to (B, T, F) as default of tokenizes (preserve original behavior)
        # x = x.swapaxes(1, 2)
        if x.ndim != 3:
            raise ValueError("tokenize_sequence expects input of shape (batch, time, features)")

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

        # Create the patches for the number of tokens: (B, n_tokens, token_size, F)
        patches = x_padded.unfold(dimension=1, size=token_size, step=stride).contiguous().squeeze(2)

        if self.method == "values":
            tokens = patches
        elif self.method == "mean":
            tokens = patches.mean(dim=2)
        elif self.method == "max":
            tokens, _ = patches.max(dim=2)
        elif self.method == "first":
            tokens = patches[:, :, 0, :]
        else:
            # should be unreachable due to check in __init__
            raise ValueError(f"Unknown tokenization method: {self.method}")

        return tokens

# Backwards-compatible convenience: allow using tokenize_sequence(...) like before.
def tokenize_sequence(
    x: torch.Tensor,
    *,
    token_size: int = 32,
    stride: Optional[int] = None,
    method: Literal["mean", "max", "first", "values"] = "values",
    pad: bool = True,
) -> torch.Tensor:
    return Tokenizer(token_size=token_size, stride=stride, method=method, pad=pad)(x)

class _InputConv(nn.Module):
    """
    Accept input of shape (B, windows, window_size, window_size) and produce
    (B, windows, out_dim) by applying a Conv2d over each window and collapsing
    the spatial dimensions to a single vector per window.

    token_len here is the spatial window_size (H = W = token_len).
    """
    def __init__(self, token_len: int, out_dim: int):
        super().__init__()
        if token_len <= 0:
            raise ValueError("token_len must be positive")
        self.token_len = token_len
        self.out_dim = out_dim
        # single-channel input windows -> produce out_dim channels, kernel covers whole window
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_dim, kernel_size=(token_len, token_len), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # allow array-like inputs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        residual = x
        # expect (B, windows, H, W)
        if x.ndim != 4:
            raise ValueError("Expected input of shape (B, windows, window_size, window_size)")

        b, windows, h, w = x.shape
        if h != self.token_len or w != self.token_len:
            raise ValueError(f"Window size mismatch: got {h}x{w}, expected {self.token_len}x{self.token_len}")

        # treat each window as a single-channel image: (B*windows, 1, H, W)
        x = x.view(b * windows, 1, h, w)
        x = self.conv(x)  # -> (B*windows, out_dim, 1, 1)
        x = x.view(b, windows, -1)  # -> (B, windows, out_dim)
        return x

class MambaVisualEncoder(nn.Module):
    """Compact Mamba-style encoder for 384-dim time series inputs."""

    def __init__(
        self,
        *,
        input_dim: int = 32, #Token size default as 16
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

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.pooling: Pooling = pooling

        self.input_proj = _InputConv(token_len=self.input_dim, out_dim=model_dim)
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

    def _time_series_2_image(self, ts):
        # validation with the shape are correct (3 is the correct b,timetamps,tokens)
        if len(ts.shape) == 4:
            ts = ts.squeeze(1)
        x = time_series_2_recurrence_plot(ts)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_sequence(x)
        pooled = self._pool_sequence(features, x)
        return self.output_proj(pooled)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        tokens = self.tokenizer(x)
        img_from_patches = self._time_series_2_image(tokens)
        img_from_patches = torch.from_numpy(img_from_patches).float().to(x.device)

        if tokens.ndim == 4:
            batch, windows, window_len, feat_dim = tokens.shape
            tokens = tokens.reshape(batch, windows * window_len, feat_dim)
        x = self.input_proj(img_from_patches) # Change here to use pre_trained model to feature exctration

        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x):
        tokens = tokenize_sequence(x, token_size=self.input_dim)
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

def create_default_encoder(**overrides: object) -> MambaVisualEncoder:
    """Factory that mirrors the defaults while allowing overrides."""
    return MambaVisualEncoder(**overrides)


if __name__ == "__main__":
    tokens_dim = 32
    encoder = MambaVisualEncoder(
        depth=6,
        input_dim=tokens_dim,
        pooling="mean",
        model_dim=768,
        embedding_dim=128,
        expand_factor=1.5
    )
    
    # print(f"Trainable parameters: {encoder.count_parameters():,}")
    dummy = torch.randn(4, 1, 384)
    out = encoder(dummy)
    print("Output embedding shape:", out.shape)
    tokens = tokenize_sequence(dummy, token_size=tokens_dim)
    print("Output tokens shape:", tokens.shape)

