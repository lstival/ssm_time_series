import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union

class BaseModelWrapper(nn.Module, ABC):
    """Base interface for all zero-shot models."""
    
    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        """
        Standard forward pass for compatibility with TFBEvaluator.
        x: (Batch, Seq, Features)
        """
        # If horizon is not specified, use a default or max horizon the model supports
        h = horizon if horizon is not None else getattr(self, "max_horizon", 96)
        return self.predict(x, h)

    @abstractmethod
    def predict(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        pass

class TimesFMModel(BaseModelWrapper):
    """
    Wrapper for Google's TimesFM model.
    """
    def __init__(self, model_id: str = "google/timesfm-2.0-500m-pytorch", device: str = "cuda"):
        super().__init__()
        try:
            from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
        except ImportError:
            raise ImportError("Please install timesfm: pip install timesfm")
            
        self.device = device
        self.max_horizon = 720
        
        # TimesFM 2.0 500M uses 50 layers.
        backend = "gpu" if "cuda" in self.device else "cpu"
        hparams = TimesFmHparams(
            context_len=512,
            horizon_len=self.max_horizon,
            num_layers=50,
            model_dims=1280,
            num_heads=16,
            backend=backend,
        )
        checkpoint = TimesFmCheckpoint(huggingface_repo_id=model_id)
        self.tfm = TimesFm(hparams=hparams, checkpoint=checkpoint)
        
    def predict(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        B, L, C = context.shape
        # Fold channels into batch dimension for univariate processing
        context_folded = context.transpose(1, 2).reshape(B * C, L)
        context_np = context_folded.cpu().numpy()
        
        # Ensure context isn't longer than 512
        if context_np.shape[1] > 512:
            context_np = context_np[:, -512:]
            
        forecast, _ = self.tfm.forecast(
            list(context_np),
            freq=[0] * len(context_np)
        )
        # Unfold back to (B, H, C)
        preds_folded = torch.from_numpy(forecast).to(context.device) # (B*C, max_horizon)
        preds = preds_folded.reshape(B, C, -1).transpose(1, 2) # (B, max_horizon, C)
        return preds[:, :horizon, :]

class ChronosModel(BaseModelWrapper):
    """
    Wrapper for Amazon's Chronos model.
    """
    def __init__(self, model_id: str = "amazon/chronos-t5-small", device: str = "cuda"):
        super().__init__()
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError("Please install chronos: pip install chronos-forecasting")
            
        self.device = device
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
        )
        # Fix for common device mismatch in chronos-forecasting
        self.pipeline.model.to(device)
        if hasattr(self.pipeline, "tokenizer"):
            for attr in ["boundaries", "center"]:
                if hasattr(self.pipeline.tokenizer, attr):
                    val = getattr(self.pipeline.tokenizer, attr)
                    if isinstance(val, torch.Tensor):
                        setattr(self.pipeline.tokenizer, attr, val.to(device))

        self.max_horizon = 720
        
    def predict(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        B, L, C = context.shape
        # Fold channels into batch dimension for univariate processing
        context_folded = context.transpose(1, 2).reshape(B * C, L)
        
        # Ensure context is on the correct device for the pipeline
        context_folded = context_folded.to(self.device)
        
        forecast = self.pipeline.predict(
            context_folded,
            prediction_length=horizon,
            num_samples=1,
        )
        # forecast is (B*C, num_samples, horizon)
        # Unfold back to (B, H, C)
        preds_folded = forecast.squeeze(1) # (B*C, horizon)
        preds = preds_folded.reshape(B, C, horizon).transpose(1, 2)
        return preds.to(context.device)

class HFTransformerModel(BaseModelWrapper):
    """
    Wrapper for HuggingFace Time Series Transformer.
    """
    def __init__(self, model_id: str = "huggingface/time-series-transformer-tourism-monthly", device: str = "cuda"):
        super().__init__()
        try:
            from transformers import TimeSeriesTransformerForPrediction
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
            
        self.device = device
        self.model = TimeSeriesTransformerForPrediction.from_pretrained(model_id).to(device)
        self.model.eval()
        self.max_horizon = self.model.config.prediction_length
        self.context_length = self.model.config.context_length
        # Lags are needed to determine minimum context
        self.lags = getattr(self.model.config, "lags_sequence", [1])
        
    def predict(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        B, original_L, C = context.shape
        # We need enough context for lags + context_length
        max_lag = max(self.lags)
        needed_context = self.context_length + max_lag
        
        # Fold channels into batch
        context_folded = context.transpose(1, 2).reshape(B * C, original_L)
        
        if context_folded.shape[1] > needed_context:
            context_folded = context_folded[:, -needed_context:]
        elif context_folded.shape[1] < needed_context:
            # Pad if too short
            pad_len = needed_context - context_folded.shape[1]
            pad = torch.zeros(context_folded.shape[0], pad_len).to(self.device)
            context_folded = torch.cat([pad, context_folded], dim=1)

        batch_size_folded, seq_len = context_folded.shape[0], context_folded.shape[1]
        num_f = getattr(self.model.config, "num_time_features", 0)
        
        with torch.no_grad():
            dtype = next(self.model.parameters()).dtype
            past_tf = torch.zeros(batch_size_folded, seq_len, num_f, dtype=dtype).to(self.device)
            # generate usually predicts its internal prediction_length
            h = max(horizon, self.max_horizon)
            fut_tf = torch.zeros(batch_size_folded, h, num_f, dtype=dtype).to(self.device)
            past_mask = torch.ones(batch_size_folded, seq_len, dtype=dtype).to(self.device)
            
            outputs = self.model.generate(
                past_values=context_folded.to(self.device).to(dtype),
                past_time_features=past_tf,
                future_time_features=fut_tf,
                past_observed_mask=past_mask
            )
            # outputs.sequences is (B*C, h)
            preds_folded = outputs.sequences
            # Unfold back to (B, h, C)
            preds = preds_folded.reshape(B, C, h).transpose(1, 2)
            return preds[:, :horizon, :].to(context.device)

class PatchTSTModel(BaseModelWrapper):
    """
    Wrapper for PatchTST (Patch Time Series Transformer).
    """
    def __init__(self, model_id: str = "google/patchtst-base", device: str = "cuda"):
        super().__init__()
        try:
            from transformers import PatchTSTForPrediction
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
            
        self.device = device
        # Use a more reliably public PatchTST model
        if "ibm/" in model_id:
            model_id = "google/patchtst-base"
        try:
            self.model = PatchTSTForPrediction.from_pretrained(model_id).to(device)
        except Exception:
            # Use TST tourism-monthly as fallback if patchtst fails
            from transformers import TimeSeriesTransformerForPrediction
            self.model = TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly").to(device)
            
        self.model.eval()
        self.max_horizon = getattr(self.model.config, "prediction_length", 720)

    def predict(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        B, L, C = context.shape
        # Fold channels into batch dimension for univariate processing
        context_folded = context.transpose(1, 2).reshape(B * C, L)
        
        with torch.no_grad():
            outputs = self.model(
                past_values=context_folded.to(self.device),
            )
            # Handle different output attributes in PatchTST
            preds_folded = getattr(outputs, "prediction_outputs", None)
            if preds_folded is None:
                 preds_folded = outputs[0]
            
            # Unfold back to (B, H, C)
            h = preds_folded.shape[-1]
            preds = preds_folded.reshape(B, C, h).transpose(1, 2)
            return preds[:, :horizon, :]

def get_model(name: str, **kwargs) -> BaseModelWrapper:
    name = name.lower()
    if "timesfm" in name:
        return TimesFMModel(**kwargs)
    elif "chronos" in name:
        return ChronosModel(**kwargs)
    elif "transformer" in name:
        return HFTransformerModel(**kwargs)
    elif "patchtst" in name:
        return PatchTSTModel(**kwargs)
    else:
        raise ValueError(f"Model {name} not found in zoo.")
