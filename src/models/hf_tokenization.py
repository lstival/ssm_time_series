"""Optional Hugging Face / Chronos tokenization helpers.

These helpers are intentionally kept out of the encoder modules so the encoder
files remain focused on the model definition.

All heavy dependencies are optional:
- `chronos` for ChronosPipeline
- `transformers` for AutoModel/AutoConfig
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

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
    """Embed univariate patches using ChronosPipeline.embed().

    Inputs:
    - `series_patches`: (B, N, P) float tensor

    Output:
    - tokens: (B, N*S, D) where S is Chronos token length per patch.
    """

    def __init__(self, *, config: ChronosConfig) -> None:
        self.config = config
        self._pipeline = None

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

        # Chronos runs on CPU in this codebase to avoid device_map issues; move back afterward.
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
    """Embed single-channel image patches (recurrence plots) with a pretrained HF vision model.

    Input: patches (B, windows, H, W)
    Output: tokens (B, windows, D)
    """

    def __init__(self, *, config: VisionConfig) -> None:
        self.config = config
        self._model = None
        self._cfg = None

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


# ---------------------------------------------------------------------------
# Compatibility shim
#
# This module is deprecated; implementations are consolidated in `tokenizers.py`.
# We keep this file for older imports, but ensure the exported symbols resolve
# to the consolidated implementations.
# ---------------------------------------------------------------------------

try:
    from tokenizers import ChronosConfig as _ChronosConfig
    from tokenizers import ChronosPatchTokenizer as _ChronosPatchTokenizer
    from tokenizers import VisionConfig as _VisionConfig
    from tokenizers import HFVisionTokenizer as _HFVisionTokenizer
except Exception:  # pragma: no cover
    from .tokenizers import ChronosConfig as _ChronosConfig
    from .tokenizers import ChronosPatchTokenizer as _ChronosPatchTokenizer
    from .tokenizers import VisionConfig as _VisionConfig
    from .tokenizers import HFVisionTokenizer as _HFVisionTokenizer

ChronosConfig = _ChronosConfig
ChronosPatchTokenizer = _ChronosPatchTokenizer
VisionConfig = _VisionConfig
HFVisionTokenizer = _HFVisionTokenizer

__all__ = [
    "ChronosConfig",
    "ChronosPatchTokenizer",
    "VisionConfig",
    "HFVisionTokenizer",
]
