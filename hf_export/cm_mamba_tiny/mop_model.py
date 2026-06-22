"""MoP (Mixture of Prompts) forecasting model for HuggingFace CM-Mamba."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from .mamba_encoder import MambaEncoder
from .mamba_visual_encoder import MambaVisualEncoder


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unknown RevIN mode: {mode}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dims = tuple(range(1, x.ndim - 1))
        self.mean = x.mean(dim=dims, keepdim=True).detach()
        self.stdev = (x.var(dim=dims, keepdim=True, unbiased=False) + self.eps).sqrt().detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        return x * self.stdev + self.mean


# ---------------------------------------------------------------------------
# ModuleOfPrompts
# ---------------------------------------------------------------------------

class ModuleOfPrompts(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_prompts: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.prompt_keys = nn.Parameter(torch.randn(num_prompts, embedding_dim) * 0.02)
        self.prompt_values = nn.Parameter(torch.randn(num_prompts, embedding_dim) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scale = z.shape[-1] ** 0.5
        scores = torch.matmul(z, self.prompt_keys.t()) / scale
        weights = F.softmax(scores, dim=-1)
        blended = torch.matmul(weights, self.prompt_values)
        return self.proj(torch.cat([z, blended], dim=-1))


# ---------------------------------------------------------------------------
# HuggingFace config + model
# ---------------------------------------------------------------------------

class CM_MambaMoPConfig(PretrainedConfig):
    """Config for the CM-Mamba + MoP forecasting model."""

    model_type = "cm_mamba_mop"

    def __init__(
        self,
        *,
        # encoder architecture (matches simclr_enc_F_lasttoken)
        input_dim: int = 64,
        model_dim: int = 192,
        embedding_dim: int = 96,
        depth: int = 5,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand_factor: float = 1.5,
        dropout: float = 0.05,
        pooling: str = "last",
        # MoP head
        num_prompts: int = 16,
        hidden_dim: int = 512,
        # forecast
        horizons: Optional[Sequence[int]] = None,
        target_features: int = 1,
        context_length: int = 336,
        # normalisation
        norm_mode: str = "revin",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.embedding_dim = int(embedding_dim)
        self.depth = int(depth)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)
        self.expand_factor = float(expand_factor)
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.num_prompts = int(num_prompts)
        self.hidden_dim = int(hidden_dim)
        self.horizons = sorted({int(h) for h in (horizons or [96, 192, 336, 720])})
        self.target_features = int(target_features)
        self.context_length = int(context_length)
        self.norm_mode = str(norm_mode)
        self.auto_map = {
            "AutoConfig": "mop_model.CM_MambaMoPConfig",
            "AutoModel": "mop_model.CM_MambaMoPModel",
        }


class CM_MambaMoPModel(PreTrainedModel):
    """CM-Mamba dual-encoder with Mixture-of-Prompts forecasting head.

    Input:  x of shape (B, T, C)  — batch, context time-steps, channels.
    Output: predictions of shape (B, H, C) for horizon H.

    The model runs channel-independently (CI): each channel is processed
    separately through the temporal and visual encoders, then the MoP head.
    """

    config_class = CM_MambaMoPConfig
    base_model_prefix = "cm_mamba_mop"

    def __init__(self, config: CM_MambaMoPConfig) -> None:
        super().__init__(config)

        enc_kwargs = dict(
            input_dim=config.input_dim,
            model_dim=config.model_dim,
            depth=config.depth,
            state_dim=config.state_dim,
            conv_kernel=config.conv_kernel,
            expand_factor=config.expand_factor,
            embedding_dim=config.embedding_dim,
            pooling=config.pooling,
            dropout=config.dropout,
        )
        self.encoder = MambaEncoder(**enc_kwargs)
        self.visual_encoder = MambaVisualEncoder(**enc_kwargs)

        # MoP input = temporal_emb + visual_emb concatenated
        mop_input_dim = config.embedding_dim * 2
        self.mop = ModuleOfPrompts(
            embedding_dim=mop_input_dim,
            num_prompts=config.num_prompts,
            hidden_dim=config.hidden_dim,
        )
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(config.hidden_dim, h * config.target_features)
            for h in config.horizons
        })
        # DLinear-style skip connection (learned direct mapping context → horizon)
        self.skip_heads = nn.ModuleDict({
            str(h): nn.Linear(config.context_length, h * config.target_features)
            for h in config.horizons
        })
        if config.norm_mode == "revin":
            self.revin = RevIN(num_features=config.target_features)
        else:
            self.revin = None

    # ------------------------------------------------------------------
    def _encode(self, x_ci: torch.Tensor) -> torch.Tensor:
        """Encode (B*C, 1, T) → (B*C, emb_dim*2)."""
        ze = self.encoder(x_ci)
        zv = self.visual_encoder(x_ci)
        return torch.cat([ze, zv], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """Forecast from context window.

        Args:
            x: (B, T, C) context tensor.
            horizon: One of the trained horizons. Defaults to the largest horizon.

        Returns:
            Predictions of shape (B, H, C).
        """
        B, T, C = x.shape
        h_key = str(horizon if horizon in self.config.horizons else self.config.horizons[-1])
        H = int(h_key)

        # Channel-independent: (B, T, C) → (B*C, 1, T)
        x_ci = x.permute(0, 2, 1).reshape(B * C, 1, T)

        # RevIN normalise on channel dim
        if self.revin is not None:
            x_norm = self.revin(x_ci.permute(0, 2, 1), "norm").permute(0, 2, 1)
        else:
            x_norm = x_ci

        # Encode
        z = self._encode(x_norm)                    # (B*C, emb*2)
        z_prompted = self.mop(z)                     # (B*C, hidden_dim)

        # Forecast heads
        pred = self.heads[h_key](z_prompted)         # (B*C, H*target_features)
        skip = self.skip_heads[h_key](x_norm[:, 0, :])  # (B*C, H*target_features)
        pred = pred + skip
        pred = pred.reshape(B * C, H, self.config.target_features)  # (B*C, H, 1)

        # RevIN denormalise
        if self.revin is not None:
            pred = self.revin(pred, "denorm")

        # (B*C, H, 1) → (B, H, C)
        pred = pred.squeeze(-1).reshape(B, C, H).permute(0, 2, 1)
        return pred

    def get_encoder(self) -> Tuple[MambaEncoder, MambaVisualEncoder]:
        """Return (temporal_encoder, visual_encoder)."""
        return self.encoder, self.visual_encoder
