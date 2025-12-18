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
import torch.nn.functional as F
from torch import nn

try:
    from utils import time_series_2_recurrence_plot
    from mamba_block import MambaBlock
except:
    from .utils import time_series_2_recurrence_plot
    from .mamba_block import MambaBlock

try:
    from transformers import AutoConfig, AutoModel  # optional; used for pretrained visual tokenization/embedding
except Exception:  # pragma: no cover
    AutoConfig = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]

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
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.pooling: Pooling = pooling

        self.use_hf_vision = bool(use_hf_vision)
        self.hf_vision_model_id = str(hf_vision_model_id)
        self.freeze_hf = bool(freeze_hf)
        self.hf_max_batch_windows = int(hf_max_batch_windows)
        self.hf_resize_to_model = bool(hf_resize_to_model)
        self._hf_vision_model = None
        self._hf_vision_config = None

        self.input_proj = _InputConv(token_len=self.input_dim, out_dim=model_dim)

        # HF path: vision backbone hidden size depends on model; use LazyLinear to avoid hard-coding.
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
        # Two supported inputs:
        # - (B, windows, H, W): already windowed 2D patches (preferred for HF vision).
        # - (B, T, F): raw time series; we tokenize and convert to recurrence-plot windows.
        if x.ndim == 4:
            patches = x
        elif x.ndim == 3:
            tokens = self.tokenizer(x)
            if tokens.ndim == 4:
                batch, windows, window_len, feat_dim = tokens.shape
                tokens = tokens.reshape(batch, windows * window_len, feat_dim)
            img_from_patches = self._time_series_2_image(tokens)
            patches = torch.from_numpy(img_from_patches).float().to(x.device)
        else:
            raise ValueError("Expected input of shape (B, T, F) or (B, windows, H, W)")

        if self.use_hf_vision and AutoModel is not None:
            vision_tokens = self._hf_visual_tokens(patches)
            x = self.hf_input_proj(vision_tokens)
        else:
            # Legacy lightweight projection: single conv over each window.
            x = self.input_proj(patches)

        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x):
        tokens = tokenize_sequence(x, token_size=self.input_dim)
        return tokens

    def _load_hf_vision(self, *, device: torch.device):
        if AutoConfig is None or AutoModel is None:
            raise RuntimeError("transformers is required for use_hf_vision=True")

        config = AutoConfig.from_pretrained(self.hf_vision_model_id)
        model = AutoModel.from_pretrained(self.hf_vision_model_id)
        model.to(device)
        if self.freeze_hf:
            model.requires_grad_(False)
            model.eval()

        self._hf_vision_model = model
        self._hf_vision_config = config

    def _hf_visual_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed each 2D patch/window into a token vector using a pretrained HF vision model.

        Input: patches of shape (B, windows, H, W) (single-channel).
        Output: token embeddings of shape (B, windows, D).
        """
        if patches.ndim != 4:
            raise ValueError("Expected patches of shape (B, windows, H, W)")

        device = patches.device
        if self._hf_vision_model is None or self._hf_vision_config is None:
            self._load_hf_vision(device=device)

        model = self._hf_vision_model
        config = self._hf_vision_config
        assert model is not None
        assert config is not None

        b, windows, h, w = patches.shape

        if self.hf_max_batch_windows <= 0:
            raise ValueError("hf_max_batch_windows must be positive")

        # Flatten windows so we can chunk the forward pass.
        flat = patches.reshape(b * windows, 1, h, w)
        flat = flat.repeat(1, 3, 1, 1)

        # Basic scaling to [0, 1] if needed.
        if flat.dtype.is_floating_point and flat.detach().max() > 1.5:
            flat = flat / 255.0

        # Optionally resize to the model's expected image size.
        image_size = getattr(config, "image_size", None)
        if self.hf_resize_to_model and isinstance(image_size, int) and (h != image_size or w != image_size):
            flat = F.interpolate(flat, size=(image_size, image_size), mode="bilinear", align_corners=False)

        # Run the backbone in small chunks to keep peak memory low.
        embeddings = []
        context = torch.inference_mode() if self.freeze_hf else torch.enable_grad()
        with context:
            for start in range(0, flat.size(0), self.hf_max_batch_windows):
                chunk = flat[start : start + self.hf_max_batch_windows]
                outputs = model(pixel_values=chunk)

                # Prefer pooler_output when available (common for convnets).
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                    # ViT-like models: hidden[:, 0] is often CLS; convnets may not have CLS.
                    if hidden.ndim == 3 and hidden.size(1) > 1:
                        emb = hidden[:, 0, :]
                    else:
                        emb = hidden.mean(dim=1)
                else:
                    raise RuntimeError(
                        "HF vision model did not return pooler_output or last_hidden_state; choose a compatible backbone"
                    )

                embeddings.append(emb)

        flat_emb = torch.cat(embeddings, dim=0)
        return flat_emb.reshape(b, windows, -1)

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
    dummy = torch.randn(4, 10, 96)
    out = encoder(dummy)
    print("Output embedding shape:", out.shape)
    tokens = tokenize_sequence(dummy, token_size=tokens_dim)
    print("Output tokens shape:", tokens.shape)

