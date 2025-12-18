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

from typing import Literal

import torch
from torch import nn

try:
    from base_encoder import BaseMambaEncoder
    from tokenizers import ChronosConfig, ChronosPatchTokenizer, Tokenizer, as_bft, tokenize_sequence
except Exception:
    from .base_encoder import BaseMambaEncoder
    from .tokenizers import ChronosConfig, ChronosPatchTokenizer, Tokenizer, as_bft, tokenize_sequence

Pooling = Literal["mean", "last", "cls"]
class MambaEncoder(BaseMambaEncoder):
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
        super().__init__(
            model_dim=model_dim,
            embedding_dim=embedding_dim,
            depth=depth,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand_factor=expand_factor,
            pooling=pooling,
            dropout=dropout,
        )
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.use_hf_tokenizer = bool(use_hf_tokenizer) and ChronosPatchTokenizer.is_available()
        self.hf_model_id = str(hf_model_id)
        self.freeze_hf = bool(freeze_hf)
        self._chronos = (
            ChronosPatchTokenizer(config=ChronosConfig(model_id=self.hf_model_id, freeze=self.freeze_hf))
            if self.use_hf_tokenizer
            else None
        )

        # Patch tokenizer used for both the legacy path and the HF (Chronos) path.
        # This tokenizer assumes (B, F, T) input by default to match existing training code.
        self.patch_tokenizer = Tokenizer(
            token_size=self.input_dim,
            stride=None,
            method="values",
            pad=True,
            input_layout="bft",
        )

        # Legacy path: window tokenizer produces tokens with feature dim == input_dim.
        self.input_proj = nn.Linear(input_dim, model_dim, bias=False)

        # HF path: Chronos embeddings have unknown last-dim at construction time.
        # LazyLinear will infer in_features on first forward.
        self.hf_input_proj = nn.LazyLinear(model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a fixed-size embedding for a 3D time-series tensor.

        This module accepts either `(B, F, T)` (common in this repo) or `(B, T, F)`.
        Internally tokenization always windows along the time axis.
        """
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, *, *)")
        return super().forward(x)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        if self.use_hf_tokenizer and self._chronos is not None:
            tokens = self._chronos_tokens(x)
            x = self.hf_input_proj(tokens)
        else:
            tokens = self.tokenizer(x)
            if tokens.ndim == 4:
                # `Tokenizer(method="values")` yields (B, N, patch_len, F).
                # Convert each patch to a token vector of length `patch_len`.
                # For multivariate inputs we average over features to keep the token size stable.
                if tokens.size(-1) == 1:
                    tokens = tokens[..., 0]
                else:
                    tokens = tokens.mean(dim=-1)
            x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Tokenizer class (patching/windowing) for the legacy path.
        x_bft = as_bft(x)
        return self.patch_tokenizer(x_bft)

    def _chronos_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self._chronos is None:
            raise RuntimeError("ChronosPatchTokenizer is not configured")

        x_bft = as_bft(x)
        patches = self.patch_tokenizer(x_bft)
        if patches.ndim != 4:
            raise ValueError("Expected patch_tokenizer(method='values') to return (B, N, patch_len, F)")

        # Convert to univariate (B, N, P).
        if patches.size(-1) == 1:
            series_patches = patches[..., 0]
        else:
            series_patches = patches.mean(dim=-1)

        return self._chronos.embed(series_patches, device=x.device, dtype=x.dtype)

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
        expand_factor=1.5,
        use_hf_tokenizer=False,
    )
    
    print(f"Trainable parameters: {encoder.count_parameters():,}")
    dummy = torch.randn(4, 10, 384)
    out = encoder(dummy)
    print("Output embedding shape:", out.shape)
    tokens = tokenize_sequence(dummy, token_size=tokens_dim)
    print("Output tokens shape:", tokens.shape)

