"""Shared helpers for training scripts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from ssm_time_series.models.mamba_encoder import MambaEncoder
from ssm_time_series.data.loader import TimeSeriesDataModule
from ssm_time_series.models.mamba_visual_encoder import MambaVisualEncoder


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    device: str
    model: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    logging: Dict[str, Any]


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    required = {"experiment_name", "seed", "device", "model", "data", "training", "logging"}
    missing = required.difference(payload.keys())
    if missing:
        raise KeyError(f"Missing keys in configuration: {sorted(missing)}")
    return ExperimentConfig(
        experiment_name=str(payload["experiment_name"]),
        seed=int(payload.get("seed", 42)),
        device=str(payload.get("device", "auto")),
        model=dict(payload["model"]),
        data=dict(payload["data"]),
        training=dict(payload["training"]),
        logging=dict(payload["logging"]),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_device(device_request: str) -> torch.device:
    device_request = device_request.lower()
    if device_request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_request in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_request == "cpu":
        return torch.device("cpu")
    return torch.device(device_request)


def build_encoder_from_config(model_cfg: Dict[str, Any]) -> MambaEncoder:
    input_dim = int(model_cfg.get("input_dim", 32))
    model_dim = int(model_cfg.get("model_dim", 128))
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    depth = int(model_cfg.get("depth", 6))
    state_dim = int(model_cfg.get("state_dim", model_cfg.get("d_state", 16)))
    conv_kernel = int(model_cfg.get("conv_kernel", model_cfg.get("d_conv", 3)))
    expand_factor = float(model_cfg.get("expand_factor", model_cfg.get("mlp_ratio", 1.5)))
    dropout = float(model_cfg.get("dropout", 0.05))
    pooling = str(model_cfg.get("pooling", "mean")).lower()

    if pooling not in {"mean", "last", "cls"}:
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    expand_factor = max(1.0, float(expand_factor))

    return MambaEncoder(
        input_dim=input_dim,
        model_dim=model_dim,
        depth=depth,
        state_dim=state_dim,
        conv_kernel=max(1, conv_kernel),
        expand_factor=expand_factor,
        embedding_dim=embedding_dim,
        pooling=pooling,
        dropout=dropout,
    )

def build_visual_encoder_from_config(model_cfg: Dict[str, Any]) -> MambaVisualEncoder:
    cfg_get = model_cfg.get
    input_channels = int(cfg_get("input_dim", 3))
    model_dim = int(cfg_get("model_dim", 128))
    embedding_dim = int(cfg_get("embedding_dim", model_dim))
    depth = int(cfg_get("depth", 6))
    state_dim = int(cfg_get("state_dim", cfg_get("d_state", 16)))
    conv_kernel = max(1, int(cfg_get("conv_kernel", cfg_get("d_conv", 3))))
    expand = float(cfg_get("expand_factor", cfg_get("mlp_ratio", 1.5)))
    dropout = float(cfg_get("dropout", 0.05))
    pooling = str(cfg_get("pooling", "cls")).lower()

    if pooling not in {"mean", "last", "cls"}:
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    expand = max(1.0, expand)

    return MambaVisualEncoder(
        input_dim=input_channels,
        model_dim=model_dim,
        depth=depth,
        state_dim=state_dim,
        conv_kernel=conv_kernel,
        expand_factor=expand,
        embedding_dim=embedding_dim,
        pooling=pooling,
        dropout=dropout,
    )

def build_optimizer(model: torch.nn.Module, training_cfg: Dict[str, Any]) -> Optimizer:
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    name = str(training_cfg.get("optimizer", "adamw")).lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(training_cfg.get("momentum", 0.9))
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: Optimizer, training_cfg: Dict[str, Any], epochs: int) -> Optional[Any]:
    name = str(training_cfg.get("scheduler", "none")).lower()
    min_lr = float(training_cfg.get("min_lr", 1e-6))
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if name == "step":
        step_size = int(training_cfg.get("step_size", 30))
        gamma = float(training_cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "plateau":
        gamma = float(training_cfg.get("gamma", 0.1))
        patience = int(training_cfg.get("patience", 10))
        return ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=patience, min_lr=min_lr)
    raise ValueError(f"Unsupported scheduler: {name}")


def prepare_dataloaders(config: ExperimentConfig, root: Path) -> Tuple[DataLoader, Optional[DataLoader]]:
    data_cfg = config.data
    data_dir = Path(data_cfg.get("data_dir", "")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()

    module = TimeSeriesDataModule(
        # dataset_name=data_cfg.get("dataset_name", "ETTh1.csv"),
        data_dir=str(data_dir),
        batch_size=int(data_cfg.get("batch_size", 128)),
        val_batch_size=int(data_cfg.get("val_batch_size", 256)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        normalize=bool(data_cfg.get("normalize", True)),
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        train=True,
        val=True,
        test=False,
    )
    module.setup()
    train_loader, val_loader = module.get_dataloaders()
    return train_loader, val_loader


def infer_feature_dim(train_loader: DataLoader) -> int:
    sample = next(iter(train_loader))
    features = sample[0]
    if features.ndim < 2:
        raise RuntimeError("Expected time-series batch with at least 2 dims")
    return features.shape[-1]
