"""Lightweight Mamba-style encoder for 384-dim time series inputs.

This module implements a compact encoder inspired by the Mamba state-space
architecture (https://arxiv.org/abs/2312.00752). It is designed to keep the
parameter count low while still capturing temporal structure in fixed-length
(time-major) sequences. The core building block is a simplified selective scan
that mixes information along the sequence using inexpensive operations.

Example
-------
>>> import torch
>>> from mamba_encoder import MambaEncoder, tokenize_sequence
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

try:
    from chronos import ChronosPipeline  # optional; used for pretrained time-series tokenization/embedding
except Exception:  # pragma: no cover
    ChronosPipeline = None  # type: ignore[assignment]

try:
    from mamba_block import MambaBlock
except Exception:
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
        x = x.swapaxes(1, 2)
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
        input_dim: int = 32, #Token size default as 16
        model_dim: int = 768,
        depth: int = 6,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        embedding_dim: int = 128,
        pooling: Pooling = "mean",
        dropout: float = 0.05,
        # Optional Hugging Face pretrained time-series tokenizer/embedding.
        # Uses ChronosPipeline.embed() to produce a token sequence, then runs the local Mamba blocks.
        # This keeps I/O the same: input remains (B, *, *), output remains (B, embedding_dim).
        use_hf_tokenizer: bool = True,
        hf_model_id: str = "amazon/chronos-t5-small",
        freeze_hf: bool = True,
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

        self.use_hf_tokenizer = bool(use_hf_tokenizer)
        self.hf_model_id = str(hf_model_id)
        self.freeze_hf = bool(freeze_hf)
        self._hf_pipeline = None

        # Legacy path: window tokenizer produces tokens with feature dim == input_dim.
        self.input_proj = nn.Linear(input_dim, model_dim, bias=False)

        # HF path: Chronos embeddings have unknown last-dim at construction time.
        # LazyLinear will infer in_features on first forward.
        self.hf_input_proj = nn.LazyLinear(model_dim, bias=False)
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
        features = self.forward_sequence(x)
        pooled = self._pool_sequence(features, x)
        return self.output_proj(pooled)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        if self.use_hf_tokenizer and ChronosPipeline is not None:
            tokens = self._hf_tokens(x)
            x = self.hf_input_proj(tokens)
        else:
            tokens = self.tokenizer(x)
            if tokens.ndim == 4:
                batch, windows, window_len, feat_dim = tokens.shape
                tokens = tokens.reshape(batch, windows * window_len, feat_dim)
            x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x):
        tokens = tokenize_sequence(x, token_size=self.input_dim)
        return tokens

    def _hf_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return a token sequence from a pretrained Hugging Face time-series model.

        Currently uses ChronosPipeline.embed(), which returns a (B, S, D) embedding sequence.
        """
        if self._hf_pipeline is None:
            self._hf_pipeline = self._load_hf_pipeline(device=x.device)

        # Map multivariate input to a single univariate series for Chronos.
        # This keeps the external shape contract unchanged while matching Chronos' expected input.
        series = self._to_univariate_series(x)
        series_cpu = series.detach().to(dtype=torch.float32, device="cpu")
        emb, _ = self._hf_pipeline.embed(series_cpu)

        # Move back to the module's device; keep dtype aligned with the local compute.
        return emb.to(device=x.device, dtype=x.dtype)

    def _load_hf_pipeline(self, *, device: torch.device):
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        pipeline = ChronosPipeline.from_pretrained(self.hf_model_id, device_map=device, dtype=dtype)
        model = getattr(pipeline, "model", None)
        if self.freeze_hf and model is not None:
            model.requires_grad_(False)
            model.eval()
        return pipeline

    @staticmethod
    def _to_univariate_series(x: torch.Tensor) -> torch.Tensor:
        """Convert (B, A, B) shaped input into a (B, T) univariate series.

        The existing codebase frequently uses (B, F, T) with F=1 for univariate.
        We handle both (B, 1, T) and (B, T, 1); otherwise we average over the feature axis
        assuming (B, F, T).
        """
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, *, *)")
        if x.size(1) == 1:
            return x[:, 0, :]
        if x.size(2) == 1:
            return x[:, :, 0]
        return x.mean(dim=1)

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
        # `LazyLinear` holds `UninitializedParameter` until the first forward pass.
        # Unit tests (and some callers) may query parameter counts before forward,
        # so we skip uninitialized parameters rather than raising.
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


def create_default_encoder(**overrides: object) -> MambaEncoder:
    """Factory that mirrors the defaults while allowing overrides."""
    return MambaEncoder(**overrides)


if __name__ == "__main__":
    tokens_dim = 32
    encoder = MambaEncoder(
        depth=6,
        input_dim=tokens_dim,
        pooling="mean",
        model_dim=128,
        embedding_dim=128,
        expand_factor=1.5
    )
    
    print(f"Trainable parameters: {encoder.count_parameters():,}")
    dummy = torch.randn(4, 1, 384)
    out = encoder(dummy)
    print("Output embedding shape:", out.shape)
    tokens = tokenize_sequence(dummy, token_size=tokens_dim)
    print("Output tokens shape:", tokens.shape)

