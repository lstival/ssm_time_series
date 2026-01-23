"""CM_Mamba: Professional Hugging Face interface for SSM encoders."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from ssm_time_series.models.mamba_encoder import MambaEncoder
from ssm_time_series.models.mamba_visual_encoder import MambaVisualEncoder


class CM_MambaConfig(PretrainedConfig):
    """Configuration class for CM_Mamba models."""
    model_type = "cm_mamba"

    def __init__(
        self,
        input_dim: int = 128,
        token_size: int = 32,
        model_dim: int = 768,
        embedding_dim: int = 128,
        depth: int = 6,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        dropout: float = 0.05,
        pooling: str = "mean",
        encoder_type: str = "temporal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.token_size = token_size
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.expand_factor = expand_factor
        self.dropout = dropout
        self.pooling = pooling
        self.encoder_type = encoder_type


class CM_MambaPreTrainedModel(PreTrainedModel):
    """Base class for CM_Mamba models."""
    config_class = CM_MambaConfig
    base_model_prefix = "cm_mamba"

    def _init_weights(self, module):
        """Standard weight initialization."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class CM_MambaTemporal(CM_MambaPreTrainedModel):
    """Temporal SSM Encoder."""

    def __init__(self, config: CM_MambaConfig, **kwargs):
        super().__init__(config)
        self.encoder = MambaEncoder(
            input_dim=config.input_dim,
            token_size=config.token_size,
            model_dim=config.model_dim,
            depth=config.depth,
            state_dim=config.state_dim,
            conv_kernel=config.conv_kernel,
            expand_factor=config.expand_factor,
            embedding_dim=config.embedding_dim,
            pooling=config.pooling,
            dropout=config.dropout,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "CM_MambaTemporal":
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if device is not None:
            model.to(device)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a time series tensor.

        Args:
            x: Input tensor with shape [B, T, F].

        Returns:
            Encoded embeddings with shape [B, D].
        """
        return self.encoder(x)


class CM_MambaVisual(CM_MambaPreTrainedModel):
    """Visual SSM Encoder."""

    def __init__(self, config: CM_MambaConfig, **kwargs):
        super().__init__(config)
        self.encoder = MambaVisualEncoder(
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "CM_MambaVisual":
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if device is not None:
            model.to(device)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a time series tensor.

        Args:
            x: Input tensor with shape [B, T, F].

        Returns:
            Encoded embeddings with shape [B, D].
        """
        return self.encoder(x)


class CM_MambaCombined(nn.Module):
    """Combined model that uses both temporal and visual encoders."""

    def __init__(
        self,
        temporal_encoder: CM_MambaTemporal,
        visual_encoder: CM_MambaVisual,
    ) -> None:
        super().__init__()
        self.temporal = temporal_encoder
        self.visual = visual_encoder
        self.embedding_dim = self.temporal.config.embedding_dim + self.visual.config.embedding_dim

    @classmethod
    def from_pretrained(
        cls,
        temporal_repo_or_path: str,
        visual_repo_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> CM_MambaCombined:
        """Load both encoders and combine them."""
        temporal = CM_MambaTemporal.from_pretrained(temporal_repo_or_path, **kwargs)
        visual = CM_MambaVisual.from_pretrained(visual_repo_or_path, **kwargs)
        if device:
            temporal.to(device)
            visual.to(device)
        return cls(temporal, visual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate embeddings from both encoders.

        Args:
            x: Input tensor with shape [B, T, F].

        Returns:
            Concatenated embeddings with shape [B, D_t + D_v].
        """
        t_feat = self.temporal(x)
        v_feat = self.visual(x)
        return torch.cat([t_feat, v_feat], dim=-1)
