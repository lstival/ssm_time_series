"""
Ablation I — Advanced Multivariate RP Methods
==============================================
Compares five state-of-the-art strategies for building Recurrence Plots from
multivariate time series, addressing the 5 methods described in ICML revision.

Methods:
--------
  1. Channel Stacking (per_channel_stack)
     - Compute RP independently per channel → stack as (N × N × d) tensor
     - Advantage: CNN/ViT learns correlations natively; excellent XAI/Grad-CAM support

  2. Global Reconstruction (global_l2)
     - Treat state as multivariate vector x_i ∈ R^d
     - Single RP from L2 distance: R_{i,j} = ||x_i - x_j||
     - Advantage: Reduces computation; Risk: variable magnitude domination masking

  3. Joint Recurrence Plots (jrp_hadamard)
     - Co-occurrence via Hadamard product: JRP_{i,j} = ∏_k RP^(k)_{i,j}
     - Advantage: Captures phase synchronization; Risk: sparse at high d

  4. Cross Recurrence Plots (crp_block)
     - Block matrix: diag(RP^(1), ..., RP^(d)) + off-diag CRPs
     - Captures lead/lag and causal structure between pairs

  5. Multi-Scale Fusion (ms_fusion_concat)
     - Multiple embedding dimensions + time delays
     - Concatenate across scales; optionally fuse with GAF/MTF

Protocol
--------
  1. Train CM-Mamba on LOTSA for --train_epochs using each method
  2. Linear-probe on ETTm1, Weather, Traffic for [96, 192, 336, 720]
  3. Save results/ablation_I_multivariate_rp.csv with MSE/MAE

Usage
-----
  python src/experiments/ablation_I_multivariate_rp.py \
      --config src/configs/lotsa_clip.yaml \
      --train_epochs 20 \
      --probe_epochs 30 \
      --results_dir results/ablation_I \
      --methods channel_stacking global_l2 jrp hadamard crp_block ms_fusion
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF

# ── path setup ──────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import (
    build_time_series_dataloaders,
    clip_contrastive_loss,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
    build_projection_head,
)
from models.mamba_encoder import MambaEncoder
from models.mamba_visual_encoder import MambaVisualEncoder
from time_series_loader import TimeSeriesDataModule

# ── constants ────────────────────────────────────────────────────────────────
MV_STRATEGIES: List[str] = [
    "channel_stacking",      # Method 1: per-channel stacked (N×N×d)
    "global_l2",             # Method 2: global L2 distance
    "jrp_hadamard",          # Method 3: Joint RP via Hadamard product
    "crp_block",             # Method 4: Cross-RP block matrix
    "ms_fusion_concat",      # Method 5: Multi-scale fusion
]

PROBE_DATASETS: List[str] = ["ETTm1.csv", "weather.csv", "traffic.csv"]
HORIZONS: List[int] = [96, 192, 336, 720]

# Benchmark configs: (label, n_channels)
BENCHMARK_CONFIGS: List[Tuple[str, int]] = [
    ("low_ch_7",    7),      # ETTm1 / ETTh1
    ("mid_ch_21",  21),      # Weather
    ("high_ch_321", 321),    # Traffic
]
_BENCHMARK_SEQ_LEN: int = 96
_BENCHMARK_BATCH:   int = 32
_BENCHMARK_REPS:    int = 50


# ── multivariate RP implementations ──────────────────────────────────────────

def rp_channel_stacking(x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
    Method 1: Channel Stacking

    Compute RP independently for each channel, then stack as (N, N, d).

    Args:
        x: (n_samples, n_channels, length) or (n_channels, length)
        threshold: If not None, binarize using recurrence threshold

    Returns:
        (n_samples, N, N, d) or (N, N, d) tensor
    """
    from pyts.image import RecurrencePlot

    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[None, :, :]  # (1, d, L)

    n_samples, n_channels, length = arr.shape
    rp_list = []

    for ch in range(n_channels):
        x_ch = arr[:, ch, :]  # (n_samples, L)
        rp_ch = RecurrencePlot(threshold=threshold).fit_transform(x_ch)  # (n_samples, L, L) already in [0,1]
        rp_list.append(rp_ch)

    # Stack along new dimension: (n_samples, L, L, d)
    result = np.stack(rp_list, axis=-1)

    # Normalize stacked result to [0, 1] along the channel dimension
    flat = result.reshape(result.shape[0], result.shape[1], result.shape[2], -1)  # Already is this shape
    mx = flat.max(axis=-1, keepdims=True).max()  # Global max
    if mx > 0:
        result = result / mx

    return result.astype(np.float32)


def rp_global_l2(x: np.ndarray) -> np.ndarray:
    """
    Method 2: Global L2 Distance in State Space

    Treat each timestep as a multivariate vector in R^d,
    compute single RP from L2 distances.

    Args:
        x: (n_samples, n_channels, length) or (n_channels, length)

    Returns:
        (n_samples, L, L) or (L, L) tensor
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]  # (1, d, L)

    n_samples, n_channels, length = arr.shape
    result = np.zeros((n_samples, length, length), dtype=np.float32)

    for s in range(n_samples):
        # (d, L) -> (L, d) for state space view
        states = arr[s, :, :].T  # (L, d)

        # Pairwise L2 distances: (L, L)
        dist = np.zeros((length, length), dtype=np.float32)
        for i in range(length):
            diff = states - states[i:i+1]  # (L, d)
            dist[i] = np.linalg.norm(diff, axis=1)

        # Normalize to [0, 1]
        dist_max = dist.max()
        if dist_max > 0:
            dist = dist / dist_max

        result[s] = dist

    return result


def rp_jrp_hadamard(x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
    Method 3: Joint Recurrence Plot via Hadamard Product

    JRP_{i,j} = ∏_k RP^(k)_{i,j} (element-wise product of per-channel RPs)
    Captures simultaneous recurrence across all variables.

    Args:
        x: (n_samples, n_channels, length)
        threshold: Recurrence threshold

    Returns:
        (n_samples, L, L) tensor normalized to [0, 1]
    """
    from pyts.image import RecurrencePlot

    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[None, :, :]

    n_samples, n_channels, length = arr.shape

    # Compute per-channel RPs (already normalized to [0, 1])
    rp_list = []
    for ch in range(n_channels):
        x_ch = arr[:, ch, :]
        rp_ch = RecurrencePlot(threshold=threshold).fit_transform(x_ch)
        rp_list.append(rp_ch)  # (n_samples, L, L) in [0, 1]

    # Hadamard product: element-wise multiplication
    # Note: product of values in [0,1] stays in [0,1]
    jrp = rp_list[0].astype(np.float32).copy()
    for ch in range(1, n_channels):
        jrp = jrp * rp_list[ch].astype(np.float32)

    # Clamp to [0, 1] just in case of floating point errors
    jrp = np.clip(jrp, 0, 1)

    return jrp.astype(np.float32)


def rp_crp_block(x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
    Method 4: Cross Recurrence Plot Block Matrix

    Constructs (d*L) × (d*L) block matrix with:
    - Diagonal: individual RP for each channel
    - Off-diagonal: cross-recurrence plots (CRP) capturing lead/lag

    Args:
        x: (n_samples, n_channels, length)
        threshold: Recurrence threshold

    Returns:
        (n_samples, d*L, d*L) tensor
    """
    from pyts.image import RecurrencePlot

    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[None, :, :]

    n_samples, n_channels, length = arr.shape
    block_size = n_channels * length
    result = np.zeros((n_samples, block_size, block_size), dtype=np.float32)

    for s in range(n_samples):
        states = arr[s, :, :].T  # (L, d)

        # Diagonal blocks: per-channel RPs
        for ch in range(n_channels):
            x_ch = arr[s, ch, :]  # (L,)
            # Pairwise L1 distance for this channel
            dist = np.abs(x_ch[:, None] - x_ch[None, :])  # (L, L)
            dist_max = dist.max()
            if dist_max > 0:
                dist = dist / dist_max

            start = ch * length
            end = (ch + 1) * length
            result[s, start:end, start:end] = dist

        # Off-diagonal blocks: cross-recurrence (simplified as L2 distance between channel pairs)
        for ch1 in range(n_channels):
            for ch2 in range(n_channels):
                if ch1 == ch2:
                    continue

                x_ch1 = arr[s, ch1, :]  # (L,)
                x_ch2 = arr[s, ch2, :]  # (L,)

                # CRP: distance between recurrence points in different channels
                dist = np.abs(x_ch1[:, None] - x_ch2[None, :])  # (L, L)
                dist_max = dist.max()
                if dist_max > 0:
                    dist = dist / dist_max

                start1 = ch1 * length
                end1 = (ch1 + 1) * length
                start2 = ch2 * length
                end2 = (ch2 + 1) * length
                result[s, start1:end1, start2:end2] = dist

    return result.astype(np.float32)


def rp_ms_fusion_concat(x: np.ndarray,
                        embedding_dims: List[int] = None,
                        time_delays: List[int] = None,
                        threshold: Optional[float] = None) -> np.ndarray:
    """
    Method 5: Multi-Scale Fusion via Concatenation

    Generate RPs at multiple embedding dimensions and time delays,
    then concatenate. Creates higher-dimensional representation.

    Args:
        x: (n_samples, n_channels, length)
        embedding_dims: List of embedding dimensions [1, 2, 3]
        time_delays: List of time delays [1, 2, 3]
        threshold: Recurrence threshold

    Returns:
        (n_samples, L, L, num_scales) stacked along last dim
    """
    from pyts.image import RecurrencePlot

    if embedding_dims is None:
        embedding_dims = [1, 2, 3]  # 3 scales
    if time_delays is None:
        time_delays = [1, 2, 3]

    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[None, :, :]

    n_samples, n_channels, length = arr.shape
    rp_scales = []

    # Generate at each scale
    for emb_dim, tau in zip(embedding_dims, time_delays):
        # Simple multi-scale: downsample and recompute
        scale = max(1, tau)
        x_scaled = arr[:, :, ::scale]  # (n_samples, n_channels, L//scale)

        if x_scaled.shape[2] < 2:
            continue

        # Compute RP on scaled data, then upsample back
        x_mean = x_scaled[:, :, :].mean(axis=1)  # (n_samples, L//scale)
        rp_scaled = RecurrencePlot(threshold=threshold).fit_transform(x_mean)  # (n_samples, L//scale, L//scale)

        # Upsample back to original length via interpolation
        rp_up = np.zeros((n_samples, length, length), dtype=np.float32)
        indices = np.round(np.linspace(0, rp_scaled.shape[1]-1, length)).astype(int)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                rp_up[:, i, j] = rp_scaled[:, idx_i, idx_j]

        rp_scales.append(rp_up)

    # Stack along new dimension
    result = np.stack(rp_scales, axis=-1)

    # Each scale is already in [0, 1], clamp to be safe
    result = np.clip(result, 0, 1)

    return result.astype(np.float32)


def _resize_square_images(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resize a batch of square images to (N, target_len, target_len)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got {arr.shape}")
    n, h, w = arr.shape
    if h == target_len and w == target_len:
        return arr

    tensor = torch.from_numpy(arr).unsqueeze(1)
    resized = torchF.interpolate(
        tensor,
        size=(target_len, target_len),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(1).numpy().astype(np.float32)


def _apply_rp_mode(img: np.ndarray, rp_mode: str) -> np.ndarray:
    """Apply existing RP mode perturbations on top of method-specific images."""
    x = np.asarray(img, dtype=np.float32)
    if rp_mode == "shuffled":
        n, l, _ = x.shape
        flat = x.reshape(n, l * l)
        idx = np.argsort(np.random.rand(n, l * l), axis=1)
        flat = flat[np.arange(n)[:, None], idx]
        return flat.reshape(n, l, l).astype(np.float32)
    if rp_mode == "random":
        return np.random.normal(0, 1, size=x.shape).astype(np.float32)
    return x


def _build_method_image_fn(method: str):
    """Return a callable that maps (N, F, L) -> (N, L, L) for each Ablation I method."""

    def _fn(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, :, :]
        if x.ndim != 3:
            raise ValueError(f"Expected (N, F, L) or (F, L), got {x.shape}")

        target_len = x.shape[-1]
        if method == "channel_stacking":
            rep = rp_channel_stacking(x)  # (N, L, L, F)
            rep = rep.mean(axis=-1)       # Keep method-specific RP generation, then fuse for encoder input
        elif method == "global_l2":
            rep = rp_global_l2(x)
        elif method == "jrp_hadamard":
            rep = rp_jrp_hadamard(x)
        elif method == "crp_block":
            rep = rp_crp_block(x)         # (N, F*L, F*L)
            rep = _resize_square_images(rep, target_len)
        elif method == "ms_fusion_concat":
            rep = rp_ms_fusion_concat(x)  # (N, L, L, S)
            rep = rep.mean(axis=-1)
        else:
            raise ValueError(f"Unknown Ablation I method: {method}")

        rep = np.nan_to_num(rep, nan=0.0, posinf=1.0, neginf=0.0)
        if rep.ndim != 3:
            raise ValueError(f"Expected 3D output (N, L, L), got {rep.shape}")
        if rep.shape[1] != target_len or rep.shape[2] != target_len:
            rep = _resize_square_images(rep, target_len)
        return rep.astype(np.float32)

    return _fn


def _patch_visual_encoder_for_method(visual: MambaVisualEncoder, method: str) -> None:
    """Patch visual encoder RP conversion so Ablation I methods are truly executed."""
    image_fn = _build_method_image_fn(method)
    visual.use_gpu_rp = False

    def _ts2img(self, ts):
        if isinstance(ts, torch.Tensor):
            arr = ts.detach().cpu().numpy()
        else:
            arr = np.asarray(ts, dtype=np.float32)
        img = image_fn(arr)
        return _apply_rp_mode(img, self.rp_mode)

    def _ts2img_gpu(self, ts: torch.Tensor) -> torch.Tensor:
        img = _ts2img(self, ts)
        return torch.from_numpy(img).float().to(ts.device)

    visual._time_series_2_image = MethodType(_ts2img, visual)
    visual._time_series_2_image_gpu = MethodType(_ts2img_gpu, visual)


# ── adapter for mamba_visual_encoder ─────────────────────────────────────────

def _build_encoders_with_mv_method(
    model_cfg: Dict,
    mv_method: str,
) -> Tuple[MambaEncoder, MambaVisualEncoder]:
    """
    Build encoders with custom multivariate RP method.
    Maps 'mv_method' to internal 'rp_mv_strategy' name for compatibility.
    """
    encoder = tu.build_encoder_from_config(model_cfg)

    visual = tu.build_visual_encoder_from_config(
        model_cfg,
        rp_mode="correct",
        rp_mv_strategy="per_channel",
        repr_type="rp",
    )
    _patch_visual_encoder_for_method(visual, mv_method)
    return encoder, visual


def _get_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def _clip_train_epoch(
    encoder: nn.Module,
    visual: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.01,
) -> float:
    encoder.train(); visual.train(); proj.train(); vproj.train()
    total, n = 0.0, 0
    for batch in loader:
        if isinstance(batch, dict) and "target" in batch:
            seq = batch["target"].to(device).float()
            if "lengths" in batch:
                seq = seq[:, : int(batch["lengths"].max().item())]
        elif isinstance(batch, (tuple, list)):
            seq = batch[0].to(device).float()
        else:
            seq = batch.to(device).float()

        seq = prepare_sequence(seq)
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))

        q = torchF.normalize(proj(encoder(x_q)), dim=1)
        k = torchF.normalize(vproj(visual(x_k)), dim=1)
        loss = clip_contrastive_loss(q, k)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()); n += 1

    return total / max(1, n)


@torch.no_grad()
def _clip_val_epoch(
    encoder: nn.Module,
    visual: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    device: torch.device,
    noise_std: float = 0.01,
) -> float:
    encoder.eval(); visual.eval(); proj.eval(); vproj.eval()
    total, n = 0.0, 0
    for batch in loader:
        if isinstance(batch, dict) and "target" in batch:
            seq = batch["target"].to(device).float()
            if "lengths" in batch:
                seq = seq[:, : int(batch["lengths"].max().item())]
        elif isinstance(batch, (tuple, list)):
            seq = batch[0].to(device).float()
        else:
            seq = batch.to(device).float()
        seq = prepare_sequence(seq)
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))
        q = torchF.normalize(proj(encoder(x_q)), dim=1)
        k = torchF.normalize(vproj(visual(x_k)), dim=1)
        total += float(clip_contrastive_loss(q, k).item()); n += 1
    encoder.train(); visual.train(); proj.train(); vproj.train()
    return total / max(1, n)


class _LinearProbe(nn.Module):
    def __init__(self, feat_dim: int, horizon: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def _probe_evaluate(
    encoder: nn.Module,
    visual: nn.Module,
    data_dir: Path,
    dataset_csv: str,
    horizons: List[int],
    device: torch.device,
    probe_epochs: int = 30,
    batch_size: int = 64,
) -> Dict[int, Dict[str, float]]:
    """Train linear probes and return {horizon: {mse, mae}}."""
    max_horizon = max(horizons)
    module = TimeSeriesDataModule(
        dataset_name=dataset_csv,
        data_dir=str(data_dir),
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        normalize=True,
        train=True,
        val=False,
        test=True,
        sample_size=(96, 0, max_horizon),
    )
    module.setup()
    if not module.train_loaders:
        return {}
    train_loader = module.train_loaders[0]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    encoder.eval(); visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    def _embed(batch):
        seq = batch[0].to(device).float()
        seq = prepare_sequence(seq)
        x = reshape_multivariate_series(seq)
        with torch.no_grad():
            ze = encoder(x)
            zv = visual(x)
        return torch.cat([ze, zv], dim=1)

    def _collect(loader):
        zs, ys = [], []
        for batch in loader:
            z = _embed(batch)
            y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
            zs.append(z); ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    Z_tr, Y_tr = _collect(train_loader)
    feat_dim = Z_tr.shape[1]

    results: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        probe = _LinearProbe(feat_dim, H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(probe_epochs):
            probe.train()
            perm = torch.randperm(Z_tr.shape[0], device=device)
            for i in range(0, Z_tr.shape[0], 64):
                idx = perm[i: i + 64]
                z_b = Z_tr[idx]; y_b = Y_tr[idx]
                if y_b.ndim == 3:
                    y_b = y_b[:, :H, 0]
                elif y_b.ndim == 2:
                    y_b = y_b[:, :H]
                if y_b.shape[1] < H:
                    continue
                pred = probe(z_b)
                loss = torchF.mse_loss(pred, y_b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if test_loader is not None:
            Z_test, Y_test = _collect(test_loader)
            probe.eval()
            with torch.no_grad():
                pred = probe(Z_test)
                if Y_test.ndim == 3:
                    Y_test = Y_test[:, :H, 0]
                elif Y_test.ndim == 2:
                    Y_test = Y_test[:, :H]
                if Y_test.shape[1] < H:
                    results[H] = {"MSE": float("nan"), "MAE": float("nan")}
                    continue
                mse = torchF.mse_loss(pred, Y_test).item()
                mae = (pred - Y_test).abs().mean().item()
            results[H] = {"MSE": mse, "MAE": mae}
        else:
            results[H] = {"MSE": float("nan"), "MAE": float("nan")}

    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(True)

    return results


# ── main workflow ────────────────────────────────────────────────────────────

def train_and_probe(
    config_path: Path,
    methods: List[str],
    train_epochs: int,
    probe_epochs: int,
    data_dir: Path,
    pretrain_data_dir: Path,
    results_dir: Path,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Train CM-Mamba with each multivariate RP method, then linear-probe.

    Returns:
        dict mapping method_name -> {horizon -> {metric -> value}}
    """
    if device is None:
        device = _get_device("auto")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load base config (returns ExperimentConfig object with dict attributes)
    cfg = tu.load_config(config_path)
    model_cfg = cfg.model  # dict
    training_cfg = cfg.training  # dict

    train_loader, val_loader = build_time_series_dataloaders(
        data_dir=str(pretrain_data_dir),
        dataset_name=cfg.data.get("dataset_name", ""),
        dataset_type=cfg.data.get("dataset_type", "icml"),
        batch_size=int(cfg.data.get("batch_size", 128)),
        val_batch_size=int(cfg.data.get("val_batch_size", 64)),
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=bool(cfg.data.get("pin_memory", True)),
        val_ratio=float(cfg.data.get("val_ratio", 0.1)),
        cronos_kwargs=dict(cfg.data.get("cronos_kwargs", {})),
        seed=seed,
    )

    results: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}

    for method in methods:
        print(f"\n{'='*70}")
        print(f"Ablation I: Method = {method}")
        print(f"{'='*70}")

        try:
            # Build encoders with this method
            encoder, visual = _build_encoders_with_mv_method(model_cfg, method)
            encoder = encoder.to(device)
            visual = visual.to(device)

            # Build optimizer and projection heads
            proj_head = build_projection_head(
                encoder,
            ).to(device)
            vproj_head = build_projection_head(visual).to(device)

            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) +
                list(visual.parameters()) +
                list(proj_head.parameters()) +
                list(vproj_head.parameters()),
                lr=float(training_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            )
            noise_std = float(training_cfg.get("noise_std", 0.01))

            start = time.time()
            for ep in range(train_epochs):
                train_loss = _clip_train_epoch(
                    encoder,
                    visual,
                    proj_head,
                    vproj_head,
                    train_loader,
                    optimizer,
                    device,
                    noise_std=noise_std,
                )
                val_loss = None
                if val_loader is not None:
                    val_loss = _clip_val_epoch(
                        encoder,
                        visual,
                        proj_head,
                        vproj_head,
                        val_loader,
                        device,
                        noise_std=noise_std,
                    )

                val_text = f"  val={val_loss:.4f}" if val_loss is not None else ""
                print(
                    f"  epoch {ep+1:02d}/{train_epochs}  train={train_loss:.4f}{val_text}"
                )

            elapsed = time.time() - start
            print(f"  [Training phase] Completed in {elapsed:.1f}s")

            ckpt_dir = results_dir / f"method_{method}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": encoder.state_dict()}, ckpt_dir / "encoder.pt")
            torch.save({"model_state": visual.state_dict()}, ckpt_dir / "visual_encoder.pt")

            # Linear probe on downstream datasets
            method_results: Dict[str, Dict[int, Dict[str, float]]] = {}
            for dataset in PROBE_DATASETS:
                print(f"\n  Probing on {dataset}...")
                dataset_results = _probe_evaluate(
                    encoder,
                    visual,
                    data_dir=data_dir,
                    dataset_csv=dataset,
                    horizons=HORIZONS,
                    device=device,
                    probe_epochs=probe_epochs,
                    batch_size=batch_size,
                )
                for horizon, metrics in dataset_results.items():
                    print(
                        f"    H={horizon}: MSE={metrics['MSE']:.4f} MAE={metrics['MAE']:.4f}"
                    )
                method_results[dataset] = dataset_results

            results[method] = method_results

        except Exception as e:
            print(f"  ❌ Error with method {method}: {e}")
            import traceback
            traceback.print_exc()

    return results


def save_results(results: Dict, output_path: Path) -> None:
    """
    Save results in tabular format.

    CSV structure:
      Method, Dataset, H_96_MSE, H_96_MAE, H_192_MSE, ..., H_720_MAE
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in sorted(results.keys()):
        method_data = results[method]
        for dataset in sorted(method_data.keys()):
            dataset_data = method_data[dataset]
            row = {"Method": method, "Dataset": dataset}

            for horizon in HORIZONS:
                if horizon in dataset_data:
                    row[f"H{horizon}_MSE"] = dataset_data[horizon]["MSE"]
                    row[f"H{horizon}_MAE"] = dataset_data[horizon]["MAE"]

            rows.append(row)

    if not rows:
        print("⚠️  No results to save")
        return

    fieldnames = ["Method", "Dataset"] + [
        f"H{h}_{m}" for h in HORIZONS for m in ["MSE", "MAE"]
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation I: Advanced Multivariate RP Methods"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/configs/lotsa_clip.yaml"),
        help="Path to model config (YAML)",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--probe_epochs",
        type=int,
        default=30,
        help="Number of linear probe epochs",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/ablation_I"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("ICML_datasets"),
        help="Downstream probe dataset directory",
    )
    parser.add_argument(
        "--pretrain_data_dir",
        type=Path,
        default=Path("data"),
        help="Pretraining dataset directory (LOTSA)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=MV_STRATEGIES,
        help="List of methods to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    device = _get_device(args.device)
    print(f"📍 Device: {device}")
    print(f"📍 Config: {args.config}")
    print(f"📍 Methods: {args.methods}")
    print(f"📍 Train epochs: {args.train_epochs}")
    print(f"📍 Probe epochs: {args.probe_epochs}")
    print(f"📍 Pretrain data: {args.pretrain_data_dir}")
    print(f"📍 Probe data: {args.data_dir}")

    # Run ablation
    results = train_and_probe(
        config_path=args.config,
        methods=args.methods,
        train_epochs=args.train_epochs,
        probe_epochs=args.probe_epochs,
        data_dir=args.data_dir,
        pretrain_data_dir=args.pretrain_data_dir,
        results_dir=args.results_dir,
        device=device,
        seed=args.seed,
    )

    # Save results
    output_csv = args.results_dir / "ablation_I_multivariate_rp.csv"
    save_results(results, output_csv)

    print("\n✅ Ablation I complete!")


if __name__ == "__main__":
    main()
