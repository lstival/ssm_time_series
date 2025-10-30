"""Tiny helpers shared by the training scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloaders.cronos_loader import build_chronos_dataloaders
from time_series_loader import TimeSeriesDataModule


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
            train_ratio=train_ratio,
            val_ratio=val_ratio,
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
) -> None:
    """Train the contrastive encoder with tqdm progress bars."""

    model = model.to(device)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    best_metric = float("inf")

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    for epoch in range(epochs):
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

        if checkpoint_dir is not None:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metric": metric,
            }

            if save_best_only:
                if metric < best_metric:
                    best_metric = metric
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
        # compute segment boundaries (as even as possible)
        sizes = [L // K] * K
        for idx in range(L % K):
            sizes[idx] += 1
        boundaries = []
        pos = 0
        for s in sizes:
            boundaries.append((pos, pos + s))
            pos += s
        perm = torch.randperm(K, device=device).tolist()
        permuted = torch.empty_like(out)
        for si, pj in enumerate(perm):
            start_src, end_src = boundaries[pj]
            start_dst, end_dst = boundaries[si]
            permuted[:, :, start_dst:end_dst] = out[:, :, start_src:end_src]
        out = permuted

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
    