"""Lightweight Mamba-style encoder for 384-dim time series inputs.

This module implements a compact encoder inspired by the Mamba state-space
architecture (https://arxiv.org/abs/2312.00752). It is designed to keep the
parameter count low while still capturing temporal structure in fixed-length
(time-major) sequences. The core building block is a simplified selective scan
that mixes information along the sequence using inexpensive operations.

"""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn.functional as F
from torch import nn

try:
    from base_encoder import BaseMambaEncoder
    from tokenizers import HFVisionTokenizer, Tokenizer, VisionConfig, tokenize_sequence
except Exception:
    from .base_encoder import BaseMambaEncoder
    from .tokenizers import HFVisionTokenizer, Tokenizer, VisionConfig, tokenize_sequence

try:
    from utils import time_series_2_recurrence_plot
except Exception:
    from .utils import time_series_2_recurrence_plot

Pooling = Literal["mean", "last", "cls"]
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

class MambaVisualEncoder(BaseMambaEncoder):
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
        # Optional Hugging Face pretrained vision backbone.
        # If enabled and `transformers` is available, each 2D patch/window is embedded
        # into a token vector and then projected to `model_dim` before the local Mamba blocks.
        # This keeps the output shape the same: (B, embedding_dim).
        use_hf_vision: bool = True,
        # Default to a lightweight convnet-style backbone to avoid large CPU allocations.
        hf_vision_model_id: str = "google/mobilenet_v2_1.0_224",
        freeze_hf: bool = True,
        # Process windows in chunks to cap peak memory.
        hf_max_batch_windows: int = 64,
        # Avoid upscaling tiny patches (e.g., 32x32 -> 224x224), which can blow up memory.
        hf_resize_to_model: bool = False,
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
        self.use_hf_vision = bool(use_hf_vision) and HFVisionTokenizer.is_available()
        self.hf_vision_model_id = str(hf_vision_model_id)
        self.freeze_hf = bool(freeze_hf)
        self.hf_max_batch_windows = int(hf_max_batch_windows)
        self.hf_resize_to_model = bool(hf_resize_to_model)
        self._vision = (
            HFVisionTokenizer(
                config=VisionConfig(
                    model_id=self.hf_vision_model_id,
                    freeze=self.freeze_hf,
                    max_batch_windows=self.hf_max_batch_windows,
                    resize_to_model=self.hf_resize_to_model,
                )
            )
            if self.use_hf_vision
            else None
        )

        # Patch tokenizer used to window raw time series into fixed-length patches.
        # This visual encoder expects raw 3D input as (B, T, F) by default.
        self.patch_tokenizer = Tokenizer(
            token_size=self.input_dim,
            stride=None,
            method="values",
            pad=True,
            input_layout="btf",
        )

        self.input_proj = _InputConv(token_len=self.input_dim, out_dim=model_dim)

        # HF path: vision backbone hidden size depends on model; use LazyLinear to avoid hard-coding.
        self.hf_input_proj = nn.LazyLinear(model_dim, bias=False)

    def _time_series_2_image(self, ts):
        # validation with the shape are correct (3 is the correct b,timetamps,tokens)
        if len(ts.shape) == 4:
            ts = ts.squeeze(1)
        x = time_series_2_recurrence_plot(ts)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        # Two supported inputs:
        # - (B, windows, H, W): already windowed 2D patches (preferred for HF vision).
        # - (B, T, F): raw time series; we tokenize and convert to recurrence-plot windows.
        if x.ndim == 4:
            patches = x
        elif x.ndim == 3:
            # Support the common univariate (B, F, T) case by swapping to (B, T, F).
            if x.size(1) == 1 and x.size(2) != 1:
                x = x.swapaxes(1, 2)

            # Patch first using Tokenizer, then convert EACH patch to a recurrence-plot image.
            ts_patches = self.patch_tokenizer(x)  # (B, windows, patch_len, F)
            if ts_patches.ndim != 4:
                raise ValueError("Expected patch_tokenizer(method='values') to return (B, windows, patch_len, F)")

            b, windows, patch_len, feat_dim = ts_patches.shape
            flat = ts_patches.reshape(b * windows, patch_len, feat_dim)

            # `time_series_2_recurrence_plot` supports (n_samples, n_channels, length).
            flat_cfl = flat.permute(0, 2, 1).contiguous()
            img_from_patches = time_series_2_recurrence_plot(flat_cfl)

            # If multichannel, collapse to single-channel by averaging across channels.
            if img_from_patches.ndim == 4:
                img_from_patches = img_from_patches.mean(axis=1)

            patches = torch.from_numpy(img_from_patches).float().to(x.device)
            patches = patches.reshape(b, windows, patch_len, patch_len)
        else:
            raise ValueError("Expected input of shape (B, T, F) or (B, windows, H, W)")

        if self.use_hf_vision and self._vision is not None:
            if self._vision is None:
                raise RuntimeError("HFVisionTokenizer is not configured")
            vision_tokens = self._vision.embed(patches)
            x = self.hf_input_proj(vision_tokens)
        else:
            # Legacy lightweight projection: single conv over each window.
            x = self.input_proj(patches)

        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_tokenizer(x)



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
    dummy = torch.randn(4, 10, 96)
    out = encoder(dummy)
    print("Output embedding shape:", out.shape)
    tokens = tokenize_sequence(dummy, token_size=tokens_dim)
    print("Output tokens shape:", tokens.shape)

