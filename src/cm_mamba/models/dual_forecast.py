"""Dual encoder forecasting models."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class DualEncoderForecastMLP(nn.Module):
    """MLP that takes concatenated outputs from both encoder and visual_encoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizons: List[int],
        target_features: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not horizons:
            raise ValueError("At least one horizon must be provided")
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = int(target_features)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = max(self.horizons)

        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.prediction_head = nn.Linear(
            self.hidden_dim,
            self.max_horizon * self.target_features,
        )

    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        shared_out = self.shared_layers(x)
        output = self.prediction_head(shared_out)
        output = output.view(batch_size, self.max_horizon, self.target_features)
        if horizon is None:
            return output
        if horizon not in self.horizons:
            raise ValueError(f"Requested horizon {horizon} not configured; available {self.horizons}")
        return output[:, :horizon, :]

    def get_max_horizon(self) -> int:
        return self.max_horizon


class DualEncoderForecastRegressor(nn.Module):
    """Forecast regressor using both encoder and visual_encoder with frozen weights."""
    
    def __init__(
        self,
        encoder: nn.Module,
        visual_encoder: nn.Module,
        head: DualEncoderForecastMLP,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.visual_encoder = visual_encoder  
        self.head = head
        
        if freeze_encoders:
            # Freeze encoders
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
                
        self.encoder.eval()
        self.visual_encoder.eval()

    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        # Encode with both encoders
        encoder_embedding = self.encoder(x)
        visual_embedding = self.visual_encoder(x)
        
        # Concatenate embeddings
        combined_embedding = torch.cat([encoder_embedding, visual_embedding], dim=-1)
        
        # Pass through forecast head
        return self.head(combined_embedding, horizon=horizon)
