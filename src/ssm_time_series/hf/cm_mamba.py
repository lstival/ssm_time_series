"""CM_Mamba: Hugging Face-style interface for trained encoders."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from ssm_time_series.models.mamba_encoder import MambaEncoder
from ssm_time_series.models.mamba_visual_encoder import MambaVisualEncoder


class CM_MambaBase(nn.Module):
    """Base class for CM_Mamba models with from_pretrained support."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> CM_MambaBase:
        """
        Load a CM_Mamba model from a directory or Hugging Face Hub.

        Args:
            pretrained_model_name_or_path: Path to directory or HF repo ID.
            device: Device to load the model on.
            **kwargs: Additional arguments for model initialization or HF download.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model_path = Path(pretrained_model_name_or_path)

        # 1. Handle Hugging Face Hub if path doesn't exist locally
        if not model_path.exists():
            if hf_hub_download is None:
                raise ImportError(
                    "huggingface_hub is required to download models from the Hub. "
                    "Install it with `pip install huggingface_hub`."
                )
            
            # We expect a config.json and a model.pt (or similar)
            repo_id = str(pretrained_model_name_or_path)
            try:
                config_file = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
                weight_file = hf_hub_download(repo_id=repo_id, filename="pytorch_model.pt", **kwargs)
            except Exception as e:
                # Try fallback filename for weights if pytorch_model.pt fails
                try:
                    weight_file = hf_hub_download(repo_id=repo_id, filename="model.pt", **kwargs)
                except:
                    raise RuntimeError(f"Could not download model files from HF Hub: {e}")
            
            config_path = Path(config_file)
            weight_path = Path(weight_file)
        else:
            # Local path
            config_path = model_path / "config.json"
            # Try a few common weight filenames
            weight_path = None
            for fname in ["pytorch_model.pt", "model.pt", "best_model.pt"]:
                p = model_path / fname
                if p.exists():
                    weight_path = p
                    break
            
            if not config_path.exists():
                raise FileNotFoundError(f"config.json not found in {model_path}")
            if weight_path is None:
                raise FileNotFoundError(f"No weight file found in {model_path}")

        # 2. Load config
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 3. Instantiate model
        model = cls(config, **kwargs)
        
        # 4. Load weights
        # Set weights_only=True for security to avoid untrusted code execution
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        # Check if state_dict is wrapped in "model_state_dict" (common in this project's checkpoints)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        # Fix keys if they were saved with a prefix (e.g., from a wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_state_dict[k[8:]] = v
            elif k.startswith("visual_encoder."):
                new_state_dict[k[15:]] = v
            else:
                new_state_dict[k] = v
        
        model.encoder.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        return model


class CM_MambaTemporal(CM_MambaBase):
    """Temporal encoder using Mamba architecture."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        # Extract params from config, handling overrides
        input_dim = config.get("input_dim", 32)
        model_dim = config.get("model_dim", 128)
        embedding_dim = config.get("embedding_dim", 128)
        depth = config.get("depth", 6)
        state_dim = config.get("state_dim", 16)
        conv_kernel = config.get("conv_kernel", 3)
        expand_factor = config.get("expand_factor", 1.5)
        dropout = config.get("dropout", 0.05)
        pooling = config.get("pooling", "mean")

        self.encoder = MambaEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            depth=depth,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand_factor=expand_factor,
            embedding_dim=embedding_dim,
            pooling=pooling,
            dropout=dropout,
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, features, seq) or (batch, seq, features)
               Note: MambaEncoder internally handles axis swaps via tokenizer if configured.
               Usually it expects (batch, 1, seq_len * features) or similar.
               Following the project's logic, it often expects (batch, seq, features).
        """
        # If input is (batch, features, seq), transpose to (batch, seq, features) 
        # as encoders in this project typically expect that for tokenization.
        if x.ndim == 3 and x.size(1) < x.size(2) and x.size(1) in [1, 3, 384]:
            x = x.transpose(1, 2)
            
        return self.encoder(x)


class CM_MambaVisual(CM_MambaBase):
    """Visual encoder using Mamba architecture."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        input_dim = config.get("input_dim", 3)
        model_dim = config.get("model_dim", 128)
        embedding_dim = config.get("embedding_dim", 128)
        depth = config.get("depth", 6)
        state_dim = config.get("state_dim", 16)
        conv_kernel = config.get("conv_kernel", 3)
        expand_factor = config.get("expand_factor", 1.5)
        dropout = config.get("dropout", 0.05)
        pooling = config.get("pooling", "cls")

        self.encoder = MambaVisualEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            depth=depth,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand_factor=expand_factor,
            embedding_dim=embedding_dim,
            pooling=pooling,
            dropout=dropout,
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.ndim == 3 and x.size(1) < x.size(2) and x.size(1) in [1, 3, 384]:
            x = x.transpose(1, 2)
        return self.encoder(x)


class CM_MambaCombined(nn.Module):
    """Combined model that uses both temporal and visual encoders."""

    def __init__(
        self, 
        temporal_encoder: CM_MambaTemporal, 
        visual_encoder: CM_MambaVisual
    ) -> None:
        super().__init__()
        self.temporal = temporal_encoder
        self.visual = visual_encoder
        self.embedding_dim = self.temporal.embedding_dim + self.visual.embedding_dim

    @classmethod
    def from_pretrained(
        cls,
        temporal_repo_or_path: Union[str, Path],
        visual_repo_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> CM_MambaCombined:
        """Load both encoders and combine them."""
        temporal = CM_MambaTemporal.from_pretrained(temporal_repo_or_path, device=device, **kwargs)
        visual = CM_MambaVisual.from_pretrained(visual_repo_or_path, device=device, **kwargs)
        return cls(temporal, visual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate embeddings from both encoders."""
        # Ensure we don't transpose twice if we pass the same x to both
        t_feat = self.temporal(x)
        v_feat = self.visual(x)
        return torch.cat([t_feat, v_feat], dim=-1)
