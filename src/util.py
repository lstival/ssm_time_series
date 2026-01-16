"""Tiny helpers shared by the training scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from typing import Dict, List, Optional
from dataloaders.cronos_loader import build_chronos_dataloaders
from time_series_loader import TimeSeriesDataModule

from down_tasks.forecast_utils import (
	compute_multi_horizon_loss,
	compute_multi_horizon_metrics,
)

from models.classifier import (
    ForecastRegressor
	)

BatchType = Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], torch.Tensor]

class MoCoProjectionHead(nn.Module):
    """Projection head for MoCo model."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def train_epoch_dataset(
    model: ForecastRegressor,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    horizons: List[int],
    reverse_normalize: bool = False,
) -> Dict[int, float]:
    model.train()
    running_losses = {h: 0.0 for h in horizons}
    steps = 0
    max_horizon = model.max_horizon

    for batch in tqdm(dataloader, desc="Train", leave=False):
        if len(batch) == 5:
            seq_x, seq_y, _, _, norm_stats = batch
        else:
            seq_x, seq_y, _, _ = batch
            norm_stats = None

        seq_x = seq_x.to(device).float().transpose(1, 2)
        seq_y = seq_y.to(device).float()

        if seq_y.size(1) < max_horizon:
            raise ValueError(
                f"Target sequence length {seq_y.size(1)} smaller than required max horizon {max_horizon}"
            )

        targets = seq_y[:, -max_horizon:, :]

        if reverse_normalize and norm_stats is not None and norm_stats.numel() > 0:
            norm_stats = norm_stats.to(device).float()
            mins = norm_stats[:, 0, :]
            maxs = norm_stats[:, 1, :]
            ranges = (maxs - mins).clamp_min(1e-6)
        else:
            mins = maxs = ranges = None

        optimizer.zero_grad(set_to_none=True)
        predictions = model(seq_x)

        if reverse_normalize and ranges is not None:
            predictions = predictions * ranges.unsqueeze(1) + mins.unsqueeze(1)
            targets = targets * ranges.unsqueeze(1) + mins.unsqueeze(1)

        total_loss, loss_values = compute_multi_horizon_loss(predictions, targets, horizons, criterion)
        total_loss.backward()
        optimizer.step()

        for horizon, value in loss_values.items():
            running_losses[horizon] += value
        steps += 1

    return {h: running_losses[h] / max(1, steps) for h in horizons}


def evaluate_dataset(
    model: ForecastRegressor,
    dataloader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    horizons: List[int],
    reverse_normalize: bool = False,
) -> Optional[Dict[int, Dict[str, float]]]:
    if dataloader is None:
        return None

    model.eval()
    running_mse = {h: 0.0 for h in horizons}
    running_mae = {h: 0.0 for h in horizons}
    counts = {h: 0 for h in horizons}
    max_horizon = model.max_horizon

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            if len(batch) == 5:
                seq_x, seq_y, _, _, norm_stats = batch
            else:
                seq_x, seq_y, _, _ = batch
                norm_stats = None

            seq_x = seq_x.to(device).float().transpose(1, 2)
            seq_y = seq_y.to(device).float()

            if seq_y.size(1) < max_horizon:
                raise ValueError(
                    f"Target sequence length {seq_y.size(1)} smaller than required max horizon {max_horizon}"
                )

            targets = seq_y[:, -max_horizon:, :]

            if reverse_normalize and norm_stats is not None and norm_stats.numel() > 0:
                norm_stats = norm_stats.to(device).float()
                mins = norm_stats[:, 0, :]
                maxs = norm_stats[:, 1, :]
                ranges = (maxs - mins).clamp_min(1e-6)
            else:
                ranges = mins = None

            predictions = model(seq_x)

            if reverse_normalize and ranges is not None:
                predictions = predictions * ranges.unsqueeze(1) + mins.unsqueeze(1)
                targets = targets * ranges.unsqueeze(1) + mins.unsqueeze(1)

            batch_metrics = compute_multi_horizon_metrics(predictions, targets, horizons)

            target_features = targets.size(2)
            batch_size = targets.size(0)
            for horizon, metrics in batch_metrics.items():
                elements = batch_size * horizon * target_features
                running_mse[horizon] += metrics["mse"] * elements
                running_mae[horizon] += metrics["mae"] * elements
                counts[horizon] += elements

    results: Dict[int, Dict[str, float]] = {}
    for horizon in horizons:
        if counts[horizon] == 0:
            results[horizon] = {"mse": float("nan"), "mae": float("nan")}
        else:
            results[horizon] = {
                "mse": running_mse[horizon] / counts[horizon],
                "mae": running_mae[horizon] / counts[horizon],
            }
    return results


def load_encoder_checkpoint(
	encoder: nn.Module,
	checkpoint_path: Path,
	device: torch.device,
) -> Dict[str, object]:
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

	payload = torch.load(checkpoint_path, map_location=device)
	if not isinstance(payload, dict):
		raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")

	candidates = (
		payload.get("model_state_dict"),
		payload.get("encoder_state_dict"),
		payload.get("state_dict"),
		payload.get("encoder"),
		payload.get("model"),
	)
	state_dict = next((item for item in candidates if isinstance(item, dict)), None)
	state_dict = state_dict or payload

	missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"Warning: missing encoder weights: {sorted(missing)}")
	if unexpected:
		print(f"Warning: unexpected encoder weights: {sorted(unexpected)}")

	return payload


def default_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_datasets(value: Optional[Union[str, Sequence[str]]]) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    return [str(item).strip() for item in value if str(item).strip()]


def build_time_series_dataloaders(
    *,
    data_dir: str | Path,
    filename: Optional[str] = None,
    dataset_name: Optional[str] = None,
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    train: bool = True,
    val: bool = True,
    test: bool = False,
    dataset_type: str = "icml",
    datasets: Optional[Union[str, Sequence[str]]] = None,
    val_split: Optional[float] = None,
    seed: int = 42,
    cronos_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val dataloaders for ICML or Chronos datasets."""

    kind = dataset_type.lower()
    if kind == "icml":
        module = TimeSeriesDataModule(
            dataset_name=dataset_name or "",
            data_dir=str(data_dir),
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            normalize=normalize,
            filename=filename,
            train=train,
            val=val,
            test=test,
        )
        loaders = module.get_dataloaders()
        train_loader = loaders[0]
        val_loader = loaders[1] if len(loaders) > 1 else None
        return train_loader, val_loader

    if kind == "cronos":
        dataset_list = _parse_datasets(datasets or dataset_name)
        if not dataset_list:
            raise ValueError("Cronos loader requires at least one dataset name.")
        split = val_split if val_split is not None else val_ratio
        extra = dict(cronos_kwargs or {})
        dtype = extra.pop("torch_dtype", torch.float32)
        if isinstance(dtype, str):
            if not hasattr(torch, dtype):
                raise ValueError(f"Unknown torch dtype string: {dtype}")
            dtype = getattr(torch, dtype)
        train_loader, val_loader = build_chronos_dataloaders(
            dataset_list,
            val_split=split,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=extra.pop("shuffle_train", True),
            shuffle_val=extra.pop("shuffle_val", False),
            drop_last=extra.pop("drop_last", False),
            pad_value=extra.pop("pad_value", 0.0),
            torch_dtype=dtype,
            repo_id=extra.pop("repo_id", "autogluon/chronos_datasets"),
            target_dtype=extra.pop("target_dtype", "float32"),
            normalize_mode=extra.pop("normalize_mode", None),
            seed=extra.pop("seed", seed),
            load_kwargs=extra.pop("load_kwargs", None),
        )
        return train_loader, val_loader

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def extract_sequence(batch: BatchType) -> torch.Tensor:
    """Normalize different batch structures to a single tensor."""
    if isinstance(batch, dict):
        if "target" not in batch:
            raise KeyError("Batch dictionary missing 'target' key.")
        return batch["target"]
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Received empty batch tuple.")
        return batch[0]
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def prepare_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Ensure the sequence tensor has shape (batch, time, features)."""
    if seq.ndim == 0:
        seq = seq.unsqueeze(0)
    if seq.ndim == 1:
        seq = seq.unsqueeze(0)
    if seq.ndim == 2:
        seq = seq.unsqueeze(-1)
    if seq.ndim != 3:
        raise ValueError(f"Expected sequence tensor with 3 dimensions, got shape {tuple(seq.shape)}")
    return seq.contiguous()


def reshape_multivariate_series(seq: torch.Tensor) -> torch.Tensor:
    """Convert (batch, time, features) tensor into per-feature univariate channels."""
    if seq.ndim != 3:
        raise ValueError(f"Expected tensor with shape (batch, time, features); received {tuple(seq.shape)}")
    batch, length, features = seq.shape
    if features <= 1:
        return seq.transpose(1, 2)
    reshaped = seq.permute(0, 2, 1).reshape(batch * features, 1, length)
    return reshaped


def random_time_mask(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    if drop_prob <= 0.0:
        return x
    mask = torch.rand(x.shape[:2], device=x.device).unsqueeze(-1)
    keep = (mask > drop_prob).float()
    return x * keep


def random_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def random_scaling(x: torch.Tensor, low: float = 0.9, high: float = 1.1) -> torch.Tensor:
    scales = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(low, high)
    return x * scales


def create_contrastive_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    view1 = random_scaling(random_time_mask(random_jitter(x.clone())))
    view2 = random_scaling(random_time_mask(random_jitter(x.clone())))
    return view1, view2


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss1 + loss2)


def clip_contrastive_loss(
    xq: torch.Tensor, xk: torch.Tensor, *, temperature: float = 0.07
) -> torch.Tensor:
    """CLIP-style symmetric cross-entropy loss for aligned pairs."""

    if not torch.is_tensor(xq) or not torch.is_tensor(xk):
        raise TypeError("xq and xk must be torch.Tensors")
    if xq.dim() != 2 or xk.dim() != 2:
        raise ValueError("xq and xk must have shape (batch, dim)")
    if xq.shape[0] != xk.shape[0]:
        raise ValueError("xq and xk must share the same batch size")

    q = F.normalize(xq, dim=1)
    k = F.normalize(xk, dim=1)

    logits = torch.matmul(q, k.transpose(0, 1)) / float(temperature)
    targets = torch.arange(logits.size(0), device=logits.device, dtype=torch.long).to(xq.device)

    loss_q = F.cross_entropy(logits, targets)
    loss_k = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (loss_q + loss_k)


def forward_contrastive_batch(
    model: nn.Module,
    batch: BatchType,
    *,
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> torch.Tensor:
    seq_x = prepare_sequence(extract_sequence(batch)).to(device).float()
    view1, view2 = create_contrastive_views(seq_x)
    with autocast(enabled=use_amp and device.type == "cuda"):
        z1 = model(view1)
        z2 = model(view2)
        loss = info_nce_loss(z1, z2, temperature)
    return loss


def run_contrastive_training(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    temperature: float,
    device: torch.device,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = None,
    checkpoint_dir: Optional[Path] = None,
    save_best_only: bool = True,
    save_last: bool = True,
    initial_epoch: int = 0,
    best_metric: Optional[float] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Train the contrastive encoder with tqdm progress bars."""

    model = model.to(device)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    if amp_enabled and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    best_metric = float("inf") if best_metric is None else float(best_metric)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if initial_epoch >= epochs:
        print(f"Requested epochs already completed ({initial_epoch}/{epochs}).")
        return

    def _run_epoch(loader: DataLoader, train_mode: bool, desc: str) -> float:
        if loader is None:
            return float("nan")

        running = 0.0
        steps = 0
        model.train(mode=train_mode)
        grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()

        with grad_ctx:
            progress = tqdm(loader, desc=desc, leave=False)
            for batch in progress:
                loss = forward_contrastive_batch(
                    model,
                    batch,
                    device=device,
                    temperature=temperature,
                    use_amp=amp_enabled,
                )

                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        if max_grad_norm is not None:
                            scaler.unscale_(optimizer)
                            clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if max_grad_norm is not None:
                            clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()

                running += float(loss.item())
                steps += 1
                avg_loss = running / max(1, steps)
                progress.set_postfix(loss=f"{avg_loss:.4f}")

        return running / max(1, steps)

    for epoch in range(initial_epoch, epochs):
        train_loss = _run_epoch(train_loader, True, f"Train {epoch + 1}/{epochs}")
        metric = train_loss

        if val_loader is not None:
            val_loss = _run_epoch(val_loader, False, f"Val {epoch + 1}/{epochs}")
            metric = val_loss
            print(f"Epoch {epoch + 1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} | train={train_loss:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metric)
            else:
                scheduler.step()

        is_new_best = metric < best_metric
        if is_new_best:
            best_metric = metric

        if checkpoint_dir is not None:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metric": metric,
                "best_metric": best_metric,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if amp_enabled else None,
            }

            if save_best_only:
                if is_new_best:
                    torch.save(state, checkpoint_dir / "best.pt")
            else:
                torch.save(state, checkpoint_dir / f"epoch_{epoch + 1}.pt")

            if save_last:
                torch.save(state, checkpoint_dir / "last.pt")


def prepare_run_directory(base_dir: str | Path, prefix: str) -> Path:
    """Create and return ``<base>/<prefix>_<timestamp>``."""
    root = Path(base_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = root / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_and_save(
    optimizer: Optimizer,
    *,
    models: Dict[str, nn.Module],
    epoch: int,
    epochs: int,
    train_loss: float,
    val_loss: Optional[float],
    checkpoint_dir: Path,
    best_loss: float,
) -> float:
    """Log metrics and persist checkpoints for the provided models."""

    if checkpoint_dir is None:
        return best_loss

    current_lr = optimizer.param_groups[0].get("lr", float("nan"))
    if val_loss is not None:
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )
        metric = val_loss
    else:
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        metric = train_loss

    if metric != metric:  # NaN check
        return best_loss

    is_best = metric < best_loss
    updated_best = metric if is_best else best_loss

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": metric,
        }

        best_path = checkpoint_dir / f"{name}_best.pt"
        last_path = checkpoint_dir / f"{name}_last.pt"

        if is_best:
            torch.save(state, best_path)
            print(f"Saved best checkpoint to {best_path}")

        torch.save(state, last_path)

    return updated_best


def run_clip_training(
    *,
    encoder: nn.Module,
    visual_encoder: nn.Module,
    projection_head: nn.Module,
    visual_projection_head: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int = 2,
    noise_std: float = 0.01,
    optimizer: Optional[Optimizer] = None,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    experiment: Optional[Any] = None,  # comet_ml.Experiment
) -> None:
    """Training loop that optimizes a CLIP-style contrastive objective."""

    encoder.to(device)
    visual_encoder.to(device)
    projection_head.to(device)
    visual_projection_head.to(device)

    params = list(encoder.parameters())
    params += list(visual_encoder.parameters())
    params += list(projection_head.parameters())
    params += list(visual_projection_head.parameters())

    if optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=1e-3)

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)

    best_metric = float("inf") if best_loss is None else float(best_loss)
    start_epoch = max(0, int(initial_epoch))
    if start_epoch >= epochs:
        print(f"Requested epochs already completed ({start_epoch}/{epochs}).")
        return

    for epoch in range(start_epoch, epochs):
        encoder.train()
        visual_encoder.train()
        projection_head.train()
        visual_projection_head.train()

        epoch_loss = 0.0
        batches = 0

        total = len(train_loader) if hasattr(train_loader, "__len__") else None
        desc = f"Epoch {epoch + 1}/{epochs}"

        with tqdm(train_loader, desc=desc, total=total) as pbar:
            for batch in pbar:
                # Chronos loader may pad variable-length series; keep original size by trimming.
                if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
                    padded = batch["target"].to(device).float()
                    lengths = batch["lengths"].to(device)

                    q_proj_list = []
                    k_proj_list = []
                    for i in range(int(padded.size(0))):
                        L = int(lengths[i].item())
                        if L < 2:
                            continue
                        seq_i = padded[i, :L].unsqueeze(0)
                        seq_i = prepare_sequence(seq_i)
                        x_q_i = reshape_multivariate_series(seq_i)
                        noise = noise_std * torch.randn_like(x_q_i)
                        x_k_i = make_positive_view(x_q_i + noise)

                        q_encoded_i = encoder(x_q_i)
                        k_encoded_i = visual_encoder(x_k_i)
                        q_proj_list.append(F.normalize(projection_head(q_encoded_i), dim=1))
                        k_proj_list.append(F.normalize(visual_projection_head(k_encoded_i), dim=1))

                    if not q_proj_list:
                        continue
                    q_proj = torch.cat(q_proj_list, dim=0)
                    k_proj = torch.cat(k_proj_list, dim=0)
                    loss = clip_contrastive_loss(q_proj, k_proj)
                else:
                    seq = prepare_sequence(extract_sequence(batch)).to(device).float()
                    x_q = reshape_multivariate_series(seq)
                    noise = noise_std * torch.randn_like(x_q)
                    x_k = make_positive_view(x_q + noise)

                    q_encoded = encoder(x_q)
                    k_encoded = visual_encoder(x_k)

                    q_proj = F.normalize(projection_head(q_encoded), dim=1)
                    k_proj = F.normalize(visual_projection_head(k_encoded), dim=1)

                    loss = clip_contrastive_loss(q_proj, k_proj)

                optimizer.zero_grad(set_to_none=True)

                loss.backward()
                optimizer.step()

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                batches += 1
                pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{(epoch_loss / batches):.4f}")

        train_loss = epoch_loss / batches if batches > 0 else float("nan")

        val_loss = None
        if val_loader is not None:
            encoder.eval()
            visual_encoder.eval()
            projection_head.eval()
            visual_projection_head.eval()

            val_epoch_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    if isinstance(val_batch, dict) and "target" in val_batch and "lengths" in val_batch:
                        padded = val_batch["target"].to(device).float()
                        lengths = val_batch["lengths"].to(device)
                        q_proj_list = []
                        k_proj_list = []
                        for i in range(int(padded.size(0))):
                            L = int(lengths[i].item())
                            if L < 2:
                                continue
                            seq_i = padded[i, :L].unsqueeze(0)
                            seq_i = prepare_sequence(seq_i)
                            x_q_i = reshape_multivariate_series(seq_i)
                            noise = noise_std * torch.randn_like(x_q_i)
                            x_k_i = make_positive_view(x_q_i + noise)

                            q_encoded_i = encoder(x_q_i)
                            k_encoded_i = visual_encoder(x_k_i)
                            q_proj_list.append(F.normalize(projection_head(q_encoded_i), dim=1))
                            k_proj_list.append(F.normalize(visual_projection_head(k_encoded_i), dim=1))
                        if not q_proj_list:
                            continue
                        val_q_proj = torch.cat(q_proj_list, dim=0)
                        val_k_proj = torch.cat(k_proj_list, dim=0)
                        val_batch_loss = clip_contrastive_loss(val_q_proj, val_k_proj).item()
                    else:
                        val_seq = prepare_sequence(extract_sequence(val_batch)).to(device).float()
                        val_x_q = reshape_multivariate_series(val_seq)
                        val_noise = noise_std * torch.randn_like(val_x_q)
                        val_x_k = make_positive_view(val_x_q + val_noise)

                        val_q_encoded = encoder(val_x_q)
                        val_k_encoded = visual_encoder(val_x_k)

                        val_q_proj = F.normalize(projection_head(val_q_encoded), dim=1)
                        val_k_proj = F.normalize(visual_projection_head(val_k_encoded), dim=1)

                        val_batch_loss = clip_contrastive_loss(val_q_proj, val_k_proj).item()
                    val_epoch_loss += val_batch_loss
                    val_batches += 1

            if val_batches > 0:
                val_loss = val_epoch_loss / val_batches

            encoder.train()
            visual_encoder.train()
            projection_head.train()
            visual_projection_head.train()
        
        # Log metrics to Comet ML
        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            experiment.log_metric("learning_rate", current_lr, step=epoch + 1)

        models_to_save = {
            "time_series": encoder,
            "visual_encoder": visual_encoder,
            "time_series_projection": projection_head,
            "visual_projection": visual_projection_head,
        }

        best_metric = log_and_save(
            optimizer,
            models=models_to_save,
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint_dir=checkpoint_dir,
            best_loss=best_metric,
        )


def build_projection_head(encoder: nn.Module, *, output_dim: Optional[int] = None) -> nn.Module:
    """Infer encoder output dimension and create a projection head."""

    input_dim: Optional[int] = None
    hidden_dim: Optional[int] = None

    output_proj = getattr(encoder, "output_proj", None)
    if isinstance(output_proj, nn.Linear):
        input_dim = int(output_proj.out_features)
        hidden_dim = int(output_proj.in_features)

    if input_dim is None or hidden_dim is None:
        final_norm = getattr(encoder, "final_norm", None)
        weight = getattr(final_norm, "weight", None)
        if isinstance(weight, torch.Tensor):
            hidden_dim = int(weight.shape[0])
        embedding_dim = getattr(encoder, "embedding_dim", None)
        if isinstance(embedding_dim, int):
            input_dim = int(embedding_dim)

    if input_dim is None or hidden_dim is None:
        raise RuntimeError("Unable to infer encoder dimensions for projection head construction")

    projection_dim = int(output_dim) if output_dim is not None else input_dim
    return MoCoProjectionHead(input_dim, hidden_dim, projection_dim)


def mask_time_series(
    x: torch.Tensor,
    mask_prob: float = 0.75,
    mask_value: float = 0.0,
    exact_fraction: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Return a masked version of a batch of time-series shaped (B, 1, 384).

    By default each element is independently masked with probability `mask_prob`
    (so ~mask_prob fraction masked). If `exact_fraction` is True each sample
    will have exactly floor(mask_prob * L) positions masked (randomly chosen).
    """
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.dim() != 3:
        raise ValueError("x must have shape (B, C, L)")
    B, C, L = x.shape
    
    device = x.device
    if seed is not None:
        # optional deterministic behavior
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    else:
        gen = None

    if exact_fraction:
        k = max(1, int((mask_prob) * L))
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        for i in range(B):
            if gen is None:
                perm = torch.randperm(L, device=device)
            else:
                perm = torch.randperm(L, generator=gen, device=device)
            mask[i, perm[:k]] = True
        mask = mask.unsqueeze(1)
    else:
        if gen is None:
            mask = torch.rand((B, C, L), device=device) < float(mask_prob)
        else:
            mask = torch.rand((B, C, L), generator=gen, device=device) < float(mask_prob)

    return x.masked_fill(mask, float(mask_value))


def make_positive_view(
    x: torch.Tensor,
    *,
    noise_std: float = 0.1,
    jitter_std: float = 0.1,
    mask_prob: float = 0.20,
    scale_range: tuple[float, float] = (0.9, 1.1),
    time_dropout_prob: float = 0.1,
    max_dropout_ratio: float = 0.2,
    permute_segments: int = 1,
) -> torch.Tensor:
    """Create an augmented positive view from input time-series tensor.

    Expected input shape: (batch, channels, length).
    Augmentations applied (in order):
      - additive gaussian noise
      - small jitter (additional gaussian)
      - per-(sample,channel) scaling
      - random element masking
      - optional random temporal dropout (contiguous segment zeroed)
      - optional permutation of contiguous segments

    Returns a new tensor (does not modify input in-place).
    """
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.dim() != 3:
        raise ValueError("x must have shape (batch, channels, length)")

    out = x.clone()

    # additive noise + jitter
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    if jitter_std > 0:
        out = out + jitter_std * torch.randn_like(out)

    # per-sample, per-channel scaling
    b, c, L = out.shape
    device = out.device
    scale_min, scale_max = scale_range
    if not (scale_min == 1.0 and scale_max == 1.0):
        scales = torch.empty((b, c, 1), device=device).uniform_(scale_min, scale_max)
        out = out * scales

    # random element masking
    if mask_prob > 0:
        mask = torch.rand_like(out) < float(mask_prob)
        out = out.masked_fill(mask, 0.0)

    # time dropout: zero out a contiguous segment per sample with some probability
    if time_dropout_prob > 0 and L > 0:
        for i in range(b):
            if torch.rand(1, device=device).item() < float(time_dropout_prob):
                max_len = max(1, int(L * float(max_dropout_ratio)))
                drop_len = torch.randint(1, max_len + 1, (1,), device=device).item()
                start = torch.randint(0, L - drop_len + 1, (1,), device=device).item()
                out[i, :, start : start + drop_len] = 0.0

    # permutation of segments: split into K segments and shuffle their order
    if permute_segments and L > 1:
        K = int(permute_segments)
        K = max(2, min(K, L))  # at least 2 segments, at most L
        sizes = [L // K] * K
        for idx in range(L % K):
            sizes[idx] += 1
        segments = torch.split(out, sizes, dim=2)
        perm = torch.randperm(K, device=device).tolist()
        out = torch.cat([segments[p] for p in perm], dim=2)

    return out


def simple_interpolation(time_series, target_size):
    """
    Interpolate a 2D time series of shape (time_steps, variables) to (target_size, variables).
    Accepts numpy arrays, torch tensors, or list-like inputs. Returns same type as input.
    """

    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")

    is_torch = isinstance(time_series, torch.Tensor)
    if is_torch:
        device = time_series.device
        dtype = time_series.dtype
        arr = time_series.detach().cpu().numpy()
    else:
        arr = np.asarray(time_series)

    if arr.ndim != 2:
        raise ValueError("time_series must be 2D with shape (time_steps, variables)")

    T, V = arr.shape
    if T == target_size:
        return time_series.clone() if is_torch else arr.copy()

    if T == 1:
        # If only one timestamp, repeat it
        out = np.repeat(arr, target_size, axis=0)
    else:
        old_pos = np.linspace(0.0, 1.0, T)
        new_pos = np.linspace(0.0, 1.0, target_size)
        out = np.empty((target_size, V), dtype=arr.dtype)
        for v in range(V):
            out[:, v] = np.interp(new_pos, old_pos, arr[:, v])

    if is_torch:
        return torch.from_numpy(out).to(device=device, dtype=dtype)
    return out

if __name__ == "__main__":
    # Example usage for ICML and Chronos datasets.
    class SimpleEncoder(nn.Module):
        """A tiny encoder that averages over time and projects to an embedding."""
        def __init__(self, in_features: int, embed_dim: int = 64):
            super().__init__()
            self.proj = nn.Linear(in_features, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, F) -> average over time -> (B, F) -> linear -> (B, embed_dim)
            x = x.mean(dim=1)
            return self.proj(x)


    def example_icml():
        # Adjust these to point at your ICML-style dataset
        data_dir = "C:\WUR\ssm_time_series\ICML_datasets"
        train_loader, val_loader = build_time_series_dataloaders(
            data_dir=data_dir,
            # dataset_name="your_icml_dataset",  # replace with real name
            batch_size=64,
            val_batch_size=128,
            num_workers=2,
            pin_memory=False,
            dataset_type="icml",
        )

        batch = next(iter(train_loader))
        seq = prepare_sequence(extract_sequence(batch))
        print("ICML batch shape (B,T,F):", seq.shape)

        model = SimpleEncoder(in_features=seq.shape[-1], embed_dim=64).to(default_device())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # run a tiny training run (1 epoch) to demonstrate API
        run_contrastive_training(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            temperature=0.1,
            device=default_device(),
            use_amp=False,
        )


    def example_cronos():
        # Example Cronos usage. Replace dataset names with valid Chronos dataset ids.
        data_dir = "unused_for_cronos"  # not required by Chronos loader, kept for API compatibility
        train_loader, val_loader = build_time_series_dataloaders(
            data_dir=data_dir,
            datasets="m4_daily,m4_hourly",  # comma-separated list or sequence
            batch_size=64,
            val_batch_size=128,
            num_workers=2,
            pin_memory=False,
            dataset_type="cronos",
            cronos_kwargs={"repo_id": "autogluon/chronos_datasets"},
        )

        batch = next(iter(train_loader))
        seq = prepare_sequence(extract_sequence(batch))
        print("Cronos batch shape (B,T,F):", seq.shape)

        model = SimpleEncoder(in_features=seq.shape[-1], embed_dim=64).to(default_device())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        run_contrastive_training(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            train_loader=train_loader,
            val_loader=None,
            epochs=1,
            temperature=0.1,
            device=default_device(),
            use_amp=False,
        )

    # example_icml()
    # example_cronos()
    