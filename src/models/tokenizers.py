"""Tokenizers and patching utilities shared by encoders.

This file centralizes:
- Time-series patching (`Tokenizer`, `tokenize_sequence`, `as_bft`)
- Optional pretrained tokenization/embedding:
  - Chronos (`ChronosPatchTokenizer`)
  - Hugging Face vision backbones (`HFVisionTokenizer`)

The goal is to keep encoder modules focused on model composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math

import torch
import torch.nn.functional as F

# -----------------------------
# Basic patch/tokenize utilities
# -----------------------------

InputLayout = Literal["bft", "btf"]
TokenizeMethod = Literal["mean", "max", "first", "values"]


class Tokenizer:
    """Window a sequence into fixed-length tokens/patches.

    Contract:
    - Input is a 3D tensor with layout controlled by `input_layout`.
    - For `method="values"`, output is `(B, N, P, F)`.
    """

    def __init__(
        self,
        *,
        token_size: int = 32,
        stride: Optional[int] = None,
        method: TokenizeMethod = "values",
        pad: bool = True,
        input_layout: InputLayout = "bft",
    ) -> None:
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        if method not in ("mean", "max", "first", "values"):
            raise ValueError(f"Unknown tokenization method: {method}")

        self.token_size = int(token_size)
        self.stride = int(stride) if stride is not None else int(token_size)
        self.method = method
        self.pad = bool(pad)
        self.input_layout = input_layout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Tokenizer expects a 3D tensor")

        # Normalize to (B, T, F) for windowing.
        if self.input_layout == "bft":
            x = x.swapaxes(1, 2)
        elif self.input_layout == "btf":
            pass
        else:  # pragma: no cover
            raise ValueError(f"Unknown input_layout: {self.input_layout}")

        bsz, t_len, feat = x.shape
        token_size = self.token_size
        stride = self.stride

        # Pad so unfold includes final window.
        if t_len <= token_size:
            n_steps = 1
            required_total = token_size
        else:
            n_steps = math.ceil((t_len - token_size) / stride) + 1
            required_total = token_size + stride * (n_steps - 1)
        pad_len = max(0, required_total - t_len)

        if pad_len > 0:
            if not self.pad:
                raise ValueError(
                    "Sequence length is shorter than token_size and pad=False; cannot form full tokens"
                )
            pad_tensor = x.new_zeros((bsz, pad_len, feat))
            x = torch.cat([x, pad_tensor], dim=1)

        # (B, T, F) -> (B, N, F, token_size) -> (B, N, token_size, F)
        patches = x.unfold(dimension=1, size=token_size, step=stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()

        if self.method == "values":
            return patches
        if self.method == "mean":
            return patches.mean(dim=2)
        if self.method == "max":
            tokens, _ = patches.max(dim=2)
            return tokens
        if self.method == "first":
            return patches[:, :, 0, :]

        raise ValueError(f"Unknown tokenization method: {self.method}")


def tokenize_sequence(
    x: torch.Tensor,
    *,
    token_size: int = 32,
    stride: Optional[int] = None,
    method: TokenizeMethod = "values",
    pad: bool = True,
    input_layout: InputLayout = "bft",
) -> torch.Tensor:
    return Tokenizer(
        token_size=token_size,
        stride=stride,
        method=method,
        pad=pad,
        input_layout=input_layout,
    )(x)


def as_bft(x: torch.Tensor, *, max_feature_dim: int = 64) -> torch.Tensor:
    """Best-effort normalization to (B, F, T)."""

    if x.ndim != 3:
        raise ValueError("Expected a 3D tensor")
    if x.size(1) <= x.size(2) and x.size(1) <= max_feature_dim:
        return x
    return x.transpose(1, 2)


# -----------------------------
# Optional pretrained tokenizers
# -----------------------------

try:
    from chronos import ChronosPipeline  # type: ignore
except Exception:  # pragma: no cover
    ChronosPipeline = None  # type: ignore

try:
    from transformers import AutoConfig, AutoModel  # type: ignore
except Exception:  # pragma: no cover
    AutoConfig = None  # type: ignore
    AutoModel = None  # type: ignore


@dataclass
class ChronosConfig:
    model_id: str = "amazon/chronos-t5-small"
    freeze: bool = True


class ChronosPatchTokenizer:
    """Embed univariate patches using ChronosPipeline.embed()."""

    def __init__(self, *, config: ChronosConfig) -> None:
        self.config = config
        self._pipeline = None

    @staticmethod
    def is_available() -> bool:
        return ChronosPipeline is not None

    def _load(self, *, device: torch.device):
        if ChronosPipeline is None:
            raise RuntimeError("chronos is required for ChronosPatchTokenizer")

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        pipeline = ChronosPipeline.from_pretrained(self.config.model_id, device_map=device, dtype=dtype)
        model = getattr(pipeline, "model", None)
        if self.config.freeze and model is not None:
            model.requires_grad_(False)
            model.eval()
        return pipeline

    def embed(self, series_patches: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if series_patches.ndim != 3:
            raise ValueError("Expected series_patches of shape (B, N, P)")

        if self._pipeline is None:
            self._pipeline = self._load(device=device)

        bsz, n_patches, patch_len = series_patches.shape
        series_cpu = series_patches.reshape(bsz * n_patches, patch_len).detach().to(dtype=torch.float32, device="cpu")
        emb, _ = self._pipeline.embed(series_cpu)
        emb = emb.to(device=device, dtype=dtype)

        if emb.ndim != 3:
            raise ValueError("ChronosPipeline.embed() returned an unexpected tensor rank")

        _, s_tokens, d_model = emb.shape
        return emb.reshape(bsz, n_patches, s_tokens, d_model).reshape(bsz, n_patches * s_tokens, d_model)


@dataclass
class VisionConfig:
    model_id: str = "google/mobilenet_v2_1.0_224"
    freeze: bool = True
    max_batch_windows: int = 64
    resize_to_model: bool = False


class HFVisionTokenizer:
    """Embed single-channel image patches with a pretrained HF vision model."""

    def __init__(self, *, config: VisionConfig) -> None:
        self.config = config
        self._model = None
        self._cfg = None

    @staticmethod
    def is_available() -> bool:
        return AutoConfig is not None and AutoModel is not None

    def _load(self, *, device: torch.device):
        if AutoConfig is None or AutoModel is None:
            raise RuntimeError("transformers is required for HFVisionTokenizer")

        cfg = AutoConfig.from_pretrained(self.config.model_id)
        model = AutoModel.from_pretrained(self.config.model_id)
        model.to(device)
        if self.config.freeze:
            model.requires_grad_(False)
            model.eval()
        return model, cfg

    def embed(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("Expected patches of shape (B, windows, H, W)")

        device = patches.device
        if self._model is None or self._cfg is None:
            self._model, self._cfg = self._load(device=device)

        model = self._model
        cfg = self._cfg
        assert model is not None
        assert cfg is not None

        bsz, windows, h, w = patches.shape
        if self.config.max_batch_windows <= 0:
            raise ValueError("max_batch_windows must be positive")

        flat = patches.reshape(bsz * windows, 1, h, w)
        flat = flat.repeat(1, 3, 1, 1)

        if flat.dtype.is_floating_point and flat.detach().max() > 1.5:
            flat = flat / 255.0

        image_size = getattr(cfg, "image_size", None)
        if self.config.resize_to_model and isinstance(image_size, int) and (h != image_size or w != image_size):
            flat = F.interpolate(flat, size=(image_size, image_size), mode="bilinear", align_corners=False)

        embeddings = []
        context = torch.inference_mode() if self.config.freeze else torch.enable_grad()
        with context:
            for start in range(0, flat.size(0), self.config.max_batch_windows):
                chunk = flat[start : start + self.config.max_batch_windows]
                outputs = model(pixel_values=chunk)

                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
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
        return flat_emb.reshape(bsz, windows, -1)
