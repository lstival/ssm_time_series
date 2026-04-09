"""
Ablation A — Multivariate RP Definition
========================================
Compares four strategies for building a Recurrence Plot from multivariate
time series patches:

  - per_channel : RP computed independently per channel → averaged (current default)
  - mean        : channels averaged first → single-channel RP
  - pca         : 1-component PCA projection → single-channel RP
  - joint       : Multivariate RP using L2 distance in F-dim state space

Protocol
--------
  1. Train CM-Mamba on LOTSA (or ICML datasets) for `--train_epochs` epochs
     using each strategy.
  2. Linear-probe the frozen temporal + visual encoders on ETTm1, Weather,
     Traffic for horizons [96, 192, 336, 720].
  3. Print and save MSE / MAE table (results/ablation_A_mv_rp.csv).

Usage
-----
  python src/experiments/ablation_A_mv_rp.py \
      --config src/configs/lotsa_clip.yaml \
      --train_epochs 20 \
      --probe_epochs 30 \
      --results_dir results/ablation_A
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    default_device,
    extract_sequence,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
    build_projection_head,
)
from models.mamba_encoder import MambaEncoder
from models.mamba_visual_encoder import MambaVisualEncoder
from time_series_loader import TimeSeriesDataModule

# ── constants ────────────────────────────────────────────────────────────────
MV_STRATEGIES: List[str] = ["per_channel", "mean", "pca", "joint"]
PROBE_DATASETS: List[str] = ["ETTm1.csv", "weather.csv", "traffic.csv"]
HORIZONS: List[int] = [96, 192, 336, 720]

# Benchmark configs: (label, n_channels) — covers low / medium / high channel counts
BENCHMARK_CONFIGS: List[Tuple[str, int]] = [
    ("low_ch_7",    7),    # ETTm1 / ETTh1
    ("mid_ch_21",  21),    # Weather
    ("high_ch_321", 321),  # Traffic
]
_BENCHMARK_SEQ_LEN: int = 96   # input context length
_BENCHMARK_BATCH:   int = 32   # batch size for speed measurement
_BENCHMARK_REPS:    int = 50   # repetitions (after warmup)


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def _build_encoders(
    model_cfg: Dict,
    rp_mv_strategy: str,
) -> Tuple[MambaEncoder, MambaVisualEncoder]:
    encoder = tu.build_encoder_from_config(model_cfg)
    visual = tu.build_visual_encoder_from_config(
        model_cfg,
        rp_mode="correct",
        rp_mv_strategy=rp_mv_strategy,
        repr_type="rp",
    )
    return encoder, visual


def _benchmark_speed(
    encoder: nn.Module,
    visual: nn.Module,
    device: torch.device,
    n_channels: int,
    seq_len: int = _BENCHMARK_SEQ_LEN,
    batch_size: int = _BENCHMARK_BATCH,
    n_reps: int = _BENCHMARK_REPS,
) -> Dict[str, float]:
    """Measure RP construction and full inference time per sample.

    Returns:
        rp_ms_per_sample   — time spent only in RP image construction
        infer_ms_per_sample — full visual encoder forward (RP + Mamba + proj)
    """
    encoder.eval(); visual.eval()
    x = torch.randn(batch_size, n_channels, seq_len, device=device)

    use_gpu = visual.use_gpu_rp
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)

    # ── warmup ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(5):
            visual(x)
    sync()

    # ── RP-only timing ────────────────────────────────────────────────────────
    # Replicate the tokenisation step that happens before RP construction
    x_sw = x.swapaxes(1, 2)          # (B, T, F)
    tokens = visual.tokenizer(x_sw)   # (B, windows, window_len, F)
    B, windows, window_len, F = tokens.shape
    tokens_rp = tokens.permute(0, 1, 3, 2).reshape(B * windows, F, window_len)

    rp_times: List[float] = []
    for _ in range(n_reps):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_gpu:
                _ = visual._time_series_2_image_gpu(tokens_rp)
            else:
                _ = visual._time_series_2_image(tokens_rp)
        sync()
        rp_times.append(time.perf_counter() - t0)

    # ── full inference timing ─────────────────────────────────────────────────
    infer_times: List[float] = []
    for _ in range(n_reps):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            visual(x)
        sync()
        infer_times.append(time.perf_counter() - t0)

    rp_ms     = 1000.0 * np.median(rp_times)    / batch_size
    infer_ms  = 1000.0 * np.median(infer_times) / batch_size

    return {"rp_ms_per_sample": rp_ms, "infer_ms_per_sample": infer_ms}


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


# ── linear-probe evaluation ───────────────────────────────────────────────────

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
    # freeze
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    def _embed(batch):
        seq = batch[0].to(device).float()
        seq = prepare_sequence(seq)
        x = reshape_multivariate_series(seq)
        with torch.no_grad():
            ze = encoder(x)   # (B, D)
            zv = visual(x)    # (B, D)
        return torch.cat([ze, zv], dim=1)  # (B, 2D)

    # collect embeddings + targets
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
                idx = perm[i : i + 64]
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
                if Y_test.ndim == 3: Y_test = Y_test[:, :H, 0]
                elif Y_test.ndim == 2: Y_test = Y_test[:, :H]
                if Y_test.shape[1] < H:
                    results[H] = {"mse": float("nan"), "mae": float("nan")}
                    continue
                mse = torchF.mse_loss(pred, Y_test).item()
                mae = (pred - Y_test).abs().mean().item()
            results[H] = {"mse": mse, "mae": mae}
        else:
            results[H] = {"mse": float("nan"), "mae": float("nan")}

    # unfreeze
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(True)

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation A: Multivariate RP strategies")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--train_epochs", type=int, default=20,
                   help="CLIP pre-training epochs per variant")
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_A")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--pretrain_data_dir", type=Path,
                   default=src_dir.parent / "data",
                   help="Data dir for CLIP pre-training (override for smoke tests)")
    p.add_argument("--strategies", nargs="+", default=MV_STRATEGIES,
                   choices=MV_STRATEGIES,
                   help="Subset of MV strategies to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_comet", action="store_true",
                   help="Disable Comet ML logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = _get_device("auto")
    print(f"Device: {device}")

    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Comet ML ──────────────────────────────────────────────────────────────
    experiment: Optional[Any] = None
    if not args.no_comet:
        try:
            import sys as _sys
            _sys.path.insert(0, str(src_dir))
            from comet_utils import create_comet_experiment
            experiment = create_comet_experiment("ablation_A_mv_rp")
            experiment.log_parameters({
                "train_epochs": args.train_epochs,
                "probe_epochs": args.probe_epochs,
                "strategies": args.strategies,
                "pretrain_data_dir": str(args.pretrain_data_dir),
                "seed": args.seed,
                "model_dim": config.model.get("model_dim"),
                "depth": config.model.get("depth"),
                "use_gpu_rp": config.model.get("use_gpu_rp", False),
            })
            print("Comet ML experiment created.")
        except Exception as exc:
            print(f"[WARN] Comet ML init failed ({exc}). Running without logging.")

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_time_series_dataloaders(
        data_dir=str(args.pretrain_data_dir),
        dataset_name=config.data.get("dataset_name", ""),
        dataset_type=config.data.get("dataset_type", "icml"),
        batch_size=int(config.data.get("batch_size", 128)),
        val_batch_size=int(config.data.get("val_batch_size", 64)),
        num_workers=int(config.data.get("num_workers", 4)),
        pin_memory=bool(config.data.get("pin_memory", True)),
        val_ratio=float(config.data.get("val_ratio", 0.1)),
        cronos_kwargs=dict(config.data.get("cronos_kwargs", {})),
        seed=args.seed,
    )

    rows: List[Dict] = []

    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        tu.set_seed(args.seed)
        encoder, visual = _build_encoders(config.model, strategy)
        encoder.to(device); visual.to(device)
        proj = build_projection_head(encoder).to(device)
        vproj = build_projection_head(visual).to(device)

        params = (list(encoder.parameters()) + list(visual.parameters())
                  + list(proj.parameters()) + list(vproj.parameters()))
        optimizer = torch.optim.AdamW(
            params,
            lr=float(config.training.get("learning_rate", 1e-3)),
            weight_decay=float(config.training.get("weight_decay", 1e-4)),
        )
        noise_std = float(config.training.get("noise_std", 0.01))

        t0 = time.time()
        for ep in range(args.train_epochs):
            train_loss = _clip_train_epoch(
                encoder, visual, proj, vproj, train_loader, optimizer, device,
                noise_std=noise_std,
            )
            val_loss: Optional[float] = None
            if val_loader is not None:
                val_loss = _clip_val_epoch(
                    encoder, visual, proj, vproj, val_loader, device, noise_std=noise_std,
                )

            val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
            print(f"  epoch {ep+1:02d}/{args.train_epochs}  "
                  f"train={train_loss:.4f}{val_str}")

            if experiment is not None:
                step = ep + 1
                experiment.log_metric(f"{strategy}/train_loss", train_loss, step=step)
                if val_loss is not None:
                    experiment.log_metric(f"{strategy}/val_loss", val_loss, step=step)

        elapsed = time.time() - t0
        ms_per_batch = 1000 * elapsed / (args.train_epochs * max(1, len(train_loader)))
        print(f"  Pre-training done — {elapsed:.0f}s  ({ms_per_batch:.1f} ms/batch)")

        if experiment is not None:
            experiment.log_metric(f"{strategy}/ms_per_batch", ms_per_batch)

        # ── speed benchmark (RP construction + full inference) ────────────────
        speed_results: Dict[str, Dict[str, float]] = {}
        for cfg_label, n_ch in BENCHMARK_CONFIGS:
            spd = _benchmark_speed(encoder, visual, device, n_channels=n_ch)
            speed_results[cfg_label] = spd
            print(f"  [{cfg_label}] RP={spd['rp_ms_per_sample']:.3f} ms/sample  "
                  f"Infer={spd['infer_ms_per_sample']:.3f} ms/sample")
            if experiment is not None:
                experiment.log_metric(f"{strategy}/{cfg_label}/rp_ms_per_sample",
                                      spd["rp_ms_per_sample"])
                experiment.log_metric(f"{strategy}/{cfg_label}/infer_ms_per_sample",
                                      spd["infer_ms_per_sample"])

        # Save checkpoint
        ckpt = args.results_dir / f"strategy_{strategy}"
        ckpt.mkdir(exist_ok=True)
        torch.save({"model_state": encoder.state_dict()}, ckpt / "encoder.pt")
        torch.save({"model_state": visual.state_dict()}, ckpt / "visual_encoder.pt")

        # Linear-probe evaluation
        for ds in PROBE_DATASETS:
            metrics = _probe_evaluate(
                encoder, visual,
                data_dir=args.data_dir,
                dataset_csv=ds,
                horizons=HORIZONS,
                device=device,
                probe_epochs=args.probe_epochs,
            )
            # pick the benchmark config closest to this dataset's channel count
            ds_key = ds.replace(".csv", "")
            spd = speed_results.get("low_ch_7", list(speed_results.values())[0])
            if ds_key == "weather":
                spd = speed_results.get("mid_ch_21", spd)
            elif ds_key == "traffic":
                spd = speed_results.get("high_ch_321", spd)

            for H, m in metrics.items():
                rows.append({
                    "strategy": strategy,
                    "dataset": ds_key,
                    "horizon": H,
                    "mse": f"{m['mse']:.4f}",
                    "mae": f"{m['mae']:.4f}",
                    "train_ms_per_batch": f"{ms_per_batch:.1f}",
                    "rp_ms_per_sample": f"{spd['rp_ms_per_sample']:.3f}",
                    "infer_ms_per_sample": f"{spd['infer_ms_per_sample']:.3f}",
                })
                print(f"  {ds}  H={H:3d}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")
                if experiment is not None:
                    experiment.log_metric(
                        f"{strategy}/probe_{ds_key}_H{H}_mse", m["mse"])
                    experiment.log_metric(
                        f"{strategy}/probe_{ds_key}_H{H}_mae", m["mae"])

    # Save CSV
    out_csv = args.results_dir / "ablation_A_results.csv"
    fieldnames = ["strategy", "dataset", "horizon", "mse", "mae",
                  "train_ms_per_batch", "rp_ms_per_sample", "infer_ms_per_sample"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")

    if experiment is not None:
        experiment.end()
        print("Comet ML experiment ended.")


if __name__ == "__main__":
    main()
