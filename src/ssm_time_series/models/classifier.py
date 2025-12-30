import torch
import torch.nn as nn
from typing import List, Optional

class MultiHorizonForecastMLP(nn.Module):
    """MLP that predicts up to the maximum configured horizon in one pass."""

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
            raise ValueError(
                f"Requested horizon {horizon} not configured; available {self.horizons}"
            )
        return output[:, :horizon, :]

    def get_max_horizon(self) -> int:
        return self.max_horizon


class ForecastRegressor(nn.Module):
	"""Frozen encoder followed by a trainable forecasting head."""

	def __init__(
		self,
		*,
		encoder: nn.Module,
		head: MultiHorizonForecastMLP,
		freeze_encoder: bool = True,
	) -> None:
		super().__init__()
		self.encoder = encoder
		self.head = head
		self.freeze_encoder = freeze_encoder

		if freeze_encoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
			self.encoder.eval()

	def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
		with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
			embeddings = self.encoder(x)
		return self.head(embeddings, horizon)

	def train(self, mode: bool = True):
		super().train(mode)
		if self.freeze_encoder:
			self.encoder.eval()
		return self

	@property
	def max_horizon(self) -> int:
		return self.head.get_max_horizon()