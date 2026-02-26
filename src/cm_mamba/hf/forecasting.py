"""Hugging Face forecasting interface for CM-Mamba models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from cm_mamba.models.classifier import MultiHorizonForecastMLP
from cm_mamba.models.dual_forecast import DualEncoderForecastMLP
from cm_mamba.models.mamba_encoder import MambaEncoder
from cm_mamba.models.mamba_visual_encoder import MambaVisualEncoder


class CM_MambaForecastConfig(PretrainedConfig):
    """Configuration for CM-Mamba forecasting models.

    Attributes:
        input_dim: Input feature dimension (F).
        token_size: Token window size.
        model_dim: Hidden model dimension.
        embedding_dim: Encoder embedding dimension.
        depth: Number of Mamba blocks.
        state_dim: State dimension for Mamba blocks.
        conv_kernel: Convolution kernel size.
        expand_factor: Mamba expansion factor.
        dropout: Encoder dropout.
        pooling: Pooling strategy (mean/last/cls).
        encoder_type: "temporal" or "visual" for single-encoder models.
        use_dual_encoder: Whether to use both temporal and visual encoders.
        horizons: Forecast horizons (sorted unique list).
        target_features: Number of target features.
        mlp_hidden_dim: Hidden dimension of the forecasting MLP head.
        head_dropout: Dropout for the forecasting head.
        freeze_encoder: Whether to freeze encoder weights.
    """

    model_type = "cm_mamba_forecast"

    def __init__(
        self,
        *,
        input_dim: int = 32,
        token_size: int = 32,
        model_dim: int = 128,
        embedding_dim: int = 128,
        depth: int = 8,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand_factor: float = 1.5,
        dropout: float = 0.05,
        pooling: str = "mean",
        encoder_type: str = "temporal",
        use_dual_encoder: bool = False,
        horizons: Optional[Sequence[int]] = None,
        target_features: int = 1,
        mlp_hidden_dim: int = 512,
        head_dropout: float = 0.1,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.token_size = int(token_size)
        self.model_dim = int(model_dim)
        self.embedding_dim = int(embedding_dim)
        self.depth = int(depth)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)
        self.expand_factor = float(expand_factor)
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.encoder_type = str(encoder_type)
        self.use_dual_encoder = bool(use_dual_encoder)
        self.horizons = sorted({int(h) for h in (horizons or [96])})
        self.target_features = int(target_features)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.freeze_encoder = bool(freeze_encoder)
        self.auto_map = {
            "AutoConfig": "forecasting.CM_MambaForecastConfig",
            "AutoModel": "forecasting.CM_MambaForecastModel",
        }


class CM_MambaForecastModel(PreTrainedModel):
    """Forecasting model with optional dual encoders.

    Input shape: [B, T, F]
    Output shape: [B, H, C] where H is max horizon and C is target_features.
    """

    config_class = CM_MambaForecastConfig
    base_model_prefix = "cm_mamba_forecast"

    def __init__(self, config: CM_MambaForecastConfig) -> None:
        super().__init__(config)
        self.encoder_type = config.encoder_type
        self.use_dual_encoder = config.use_dual_encoder

        self.encoder = self._build_temporal_encoder(config)
        self.visual_encoder: Optional[nn.Module] = None
        if self.use_dual_encoder:
            self.visual_encoder = self._build_visual_encoder(config)
            head_input_dim = int(self.encoder.embedding_dim) + int(self.visual_encoder.embedding_dim)
            self.head = DualEncoderForecastMLP(
                input_dim=head_input_dim,
                hidden_dim=config.mlp_hidden_dim,
                horizons=config.horizons,
                target_features=config.target_features,
                dropout=config.head_dropout,
            )
        else:
            head_input_dim = int(self.encoder.embedding_dim)
            self.head = MultiHorizonForecastMLP(
                input_dim=head_input_dim,
                hidden_dim=config.mlp_hidden_dim,
                horizons=config.horizons,
                target_features=config.target_features,
                dropout=config.head_dropout,
            )

        if config.freeze_encoder:
            self._freeze_encoders()

    def _build_temporal_encoder(self, config: CM_MambaForecastConfig) -> MambaEncoder:
        return MambaEncoder(
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

    def _build_visual_encoder(self, config: CM_MambaForecastConfig) -> MambaVisualEncoder:
        return MambaVisualEncoder(
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

    def _freeze_encoders(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.visual_encoder is not None:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        self.encoder.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()

    def get_encoder_only(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        """Return encoder module(s) without the forecast head."""
        if self.visual_encoder is None:
            return self.encoder
        return self.encoder, self.visual_encoder

    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        """Run forecasting inference.

        Args:
            x: Input tensor with shape [B, T, F].
            horizon: Optional horizon to slice outputs.

        Returns:
            Forecast predictions with shape [B, H, C].
        """
        if self.visual_encoder is None:
            embeddings = self.encoder(x)
        else:
            temporal = self.encoder(x)
            visual = self.visual_encoder(x)
            embeddings = torch.cat([temporal, visual], dim=-1)
        return self.head(embeddings, horizon=horizon)

    @staticmethod
    def _extract_state_dict(payload: Dict[str, torch.Tensor] | Dict[str, object]) -> Dict[str, torch.Tensor]:
        if "model_state_dict" in payload:
            return payload["model_state_dict"]  # type: ignore[return-value]
        if "state_dict" in payload:
            return payload["state_dict"]  # type: ignore[return-value]
        return payload  # type: ignore[return-value]

    def load_from_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        device: Optional[Union[str, torch.device]] = None,
        load_encoder: bool = True,
        load_head: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Load model weights from a training checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Optional device for loading the checkpoint.
            load_encoder: Whether to load encoder weights.
            load_head: Whether to load forecasting head weights.

        Returns:
            Tuple of (missing_keys, unexpected_keys) from ``load_state_dict``.
        """
        map_location = device if device is not None else "cpu"
        payload = torch.load(checkpoint_path, map_location=map_location)
        state_dict = self._extract_state_dict(payload)

        filtered: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder.") and load_encoder:
                filtered[key] = value
            elif key.startswith("visual_encoder.") and load_encoder:
                filtered[key] = value
            elif key.startswith("head.") and load_head:
                filtered[key] = value
            elif not (key.startswith("encoder.") or key.startswith("visual_encoder.") or key.startswith("head.")):
                if load_encoder and load_head:
                    filtered[key] = value

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        return list(missing), list(unexpected)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        config: CM_MambaForecastConfig,
        checkpoint_path: Path,
        device: Optional[Union[str, torch.device]] = None,
        load_encoder: bool = True,
        load_head: bool = True,
    ) -> "CM_MambaForecastModel":
        """Create a model and load weights from a checkpoint."""
        model = cls(config)
        model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            load_encoder=load_encoder,
            load_head=load_head,
        )
        if device is not None:
            model.to(device)
        return model

    @classmethod
    def from_checkpoint_encoder_only(
        cls,
        *,
        config: CM_MambaForecastConfig,
        checkpoint_path: Path,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        """Load encoder weights only and return the encoder module(s)."""
        model = cls.from_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
            load_encoder=True,
            load_head=False,
        )
        return model.get_encoder_only()


@dataclass
class CM_MambaForecastExportSpec:
    """Configuration container for HF export scripts."""

    config: CM_MambaForecastConfig
    checkpoint_path: Path
    output_dir: Path
    model_card_template: Optional[Path] = None
    model_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "CM_MambaForecastExportSpec":
        model_cfg = payload.get("model") or {}
        forecast_cfg = payload.get("forecast") or {}
        paths_cfg = payload.get("paths") or {}

        config = CM_MambaForecastConfig(
            input_dim=int(model_cfg.get("input_dim", 32)),
            token_size=int(model_cfg.get("token_size", 32)),
            model_dim=int(model_cfg.get("model_dim", 128)),
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            depth=int(model_cfg.get("depth", 8)),
            state_dim=int(model_cfg.get("state_dim", 16)),
            conv_kernel=int(model_cfg.get("conv_kernel", 4)),
            expand_factor=float(model_cfg.get("expand_factor", 1.5)),
            dropout=float(model_cfg.get("dropout", 0.05)),
            pooling=str(model_cfg.get("pooling", "mean")),
            encoder_type=str(model_cfg.get("encoder_type", "temporal")),
            use_dual_encoder=bool(model_cfg.get("use_dual_encoder", False)),
            horizons=forecast_cfg.get("horizons", [96]),
            target_features=int(forecast_cfg.get("target_features", 1)),
            mlp_hidden_dim=int(forecast_cfg.get("mlp_hidden_dim", 512)),
            head_dropout=float(forecast_cfg.get("head_dropout", 0.1)),
            freeze_encoder=bool(forecast_cfg.get("freeze_encoder", True)),
        )

        checkpoint_value = paths_cfg.get("checkpoint")
        output_value = paths_cfg.get("output_dir")
        model_card_template = paths_cfg.get("model_card_template")

        if checkpoint_value is None or output_value is None:
            raise ValueError("paths.checkpoint and paths.output_dir must be provided")

        checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
        output_dir = Path(str(output_value)).expanduser().resolve()
        template_path = (
            Path(model_card_template).expanduser().resolve() if model_card_template else None
        )
        model_id = paths_cfg.get("model_id")

        if not str(checkpoint_path):
            raise ValueError("paths.checkpoint is required in the export config")
        if not str(output_dir):
            raise ValueError("paths.output_dir is required in the export config")

        return cls(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            model_card_template=template_path,
            model_id=str(model_id) if model_id is not None else None,
        )
