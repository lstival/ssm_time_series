"""
Benchmark: RP image computation overhead vs model forward/backward
==================================================================
Profiles the per-batch time breakdown of CLIP training:

  t_rp    — RP image computation (tokenise → image transform)
  t_model — rest of visual-encoder forward (conv + Mamba blocks + pooling)
  t_bwd   — backward pass
  t_total — full gradient step

Two backends are compared:
  CPU-pyts : current default — pyts.RecurrencePlot on CPU with GPU↔CPU transfer
  GPU-torch: new path        — recurrence_plot_gpu, stays on GPU the whole time

Data source can be ICML (ETT-small) or the Chronos corpus subset.

Usage
-----
  python benchmark_rp_overhead.py \\
      --config   src/experiments/smoke_test/smoke_config.yaml \\
      --data_dir ICML_datasets/ETT-small \\
      --n_batches 50

  # Chronos datasets instead of ICML:
  python benchmark_rp_overhead.py \\
      --config src/configs/lotsa_clip.yaml \\
      --dataset_type icml \\
      --n_batches 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ───────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import (
    build_projection_head,
    build_time_series_dataloaders,
    clip_contrastive_loss,
    extract_sequence,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
)
from models.mamba_visual_encoder import MambaVisualEncoder


# ── CUDA-accurate timer ───────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    _sync()
    return time.perf_counter()


# ── instrumented encoder ──────────────────────────────────────────────────────

class _TimedVisualEncoder(MambaVisualEncoder):
    """MambaVisualEncoder that records t_rp and t_model per forward call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_rp: List[float] = []
        self.t_model: List[float] = []

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        B, windows, window_len, F_dim = tokens.shape
        tokens_for_rp = tokens.permute(0, 1, 3, 2).reshape(B * windows, F_dim, window_len)

        # ── RP computation ────────────────────────────────────────────────────
        t0 = _now()
        if self.use_gpu_rp:
            img = self._time_series_2_image_gpu(tokens_for_rp)
        else:
            img = self._time_series_2_image(tokens_for_rp)
        t1 = _now()
        self.t_rp.append(t1 - t0)

        # ── model layers ─────────────────────────────────────────────────────
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float().to(x.device)
        if img.ndim == 4:
            img = img.mean(dim=1)
        img = img.view(B, windows, window_len, window_len)

        _sync()
        t2 = _now()
        x_out = self.input_proj(img)
        for block in self.blocks:
            x_out = block(x_out)
        result = self.final_norm(x_out)
        t3 = _now()
        self.t_model.append(t3 - t2)

        return result


def _build_timed_encoder(model_cfg: Dict, use_gpu_rp: bool) -> _TimedVisualEncoder:
    cfg = model_cfg
    return _TimedVisualEncoder(
        input_dim=int(cfg.get("input_dim", 32)),
        model_dim=int(cfg.get("model_dim", 128)),
        depth=int(cfg.get("depth", 6)),
        state_dim=int(cfg.get("state_dim", cfg.get("d_state", 16))),
        conv_kernel=max(1, int(cfg.get("conv_kernel", cfg.get("d_conv", 3)))),
        expand_factor=max(1.0, float(cfg.get("expand_factor", cfg.get("mlp_ratio", 1.5)))),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        pooling=str(cfg.get("pooling", "cls")).lower(),
        dropout=float(cfg.get("dropout", 0.05)),
        rp_mode=str(cfg.get("rp_mode", "correct")),
        rp_mv_strategy=str(cfg.get("rp_mv_strategy", "per_channel")),
        repr_type=str(cfg.get("repr_type", "rp")),
        use_gpu_rp=use_gpu_rp,
    )


# ── one benchmark run ─────────────────────────────────────────────────────────

def _run_benchmark(
    visual: _TimedVisualEncoder,
    encoder: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    device: torch.device,
    n_batches: int,
    noise_std: float,
    label: str,
) -> Dict[str, float]:
    """Run n_batches gradient steps and return timing statistics."""
    visual.train(); encoder.train(); proj.train(); vproj.train()
    visual.t_rp.clear(); visual.t_model.clear()

    all_params = (list(encoder.parameters()) + list(visual.parameters()) +
                  list(proj.parameters()) + list(vproj.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=1e-3)

    t_bwd_list: List[float] = []
    t_total_list: List[float] = []

    loader_iter = iter(loader)
    print(f"\n  [{label}] warming up …", end="", flush=True)

    for step in range(n_batches + 5):          # +5 warmup steps
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        seq = extract_sequence(batch)
        if isinstance(batch, dict) and "lengths" in batch:
            seq = seq[:, : int(batch["lengths"].max().item())]
        seq = prepare_sequence(seq.to(device).float())
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))

        t_start = _now()

        q = F.normalize(proj(encoder(x_q)), dim=1)
        k = F.normalize(vproj(visual(x_k)), dim=1)
        loss = clip_contrastive_loss(q, k)

        t_fwd_done = _now()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        t_end = _now()

        if step < 5:
            # discard warmup
            visual.t_rp.clear(); visual.t_model.clear()
            if step == 4:
                print(" done")
            continue

        t_bwd_list.append(t_end - t_fwd_done)
        t_total_list.append(t_end - t_start)

    def _ms(lst): return 1000.0 * float(np.mean(lst))
    def _std(lst): return 1000.0 * float(np.std(lst))

    t_rp_ms    = _ms(visual.t_rp)
    t_model_ms = _ms(visual.t_model)
    t_bwd_ms   = _ms(t_bwd_list)
    t_total_ms = _ms(t_total_list)

    # t_other = overhead not attributed to rp/model/bwd (data loading, loss, etc.)
    t_other_ms = max(0.0, t_total_ms - t_rp_ms - t_model_ms - t_bwd_ms)

    return {
        "label":      label,
        "t_rp_ms":    t_rp_ms,
        "t_rp_std":   _std(visual.t_rp),
        "t_model_ms": t_model_ms,
        "t_bwd_ms":   t_bwd_ms,
        "t_other_ms": t_other_ms,
        "t_total_ms": t_total_ms,
        "rp_pct":     100.0 * t_rp_ms / max(t_total_ms, 1e-9),
        "smp_s":      1000.0 * loader.batch_size / max(t_total_ms, 1e-9),
    }


# ── pretty printing ───────────────────────────────────────────────────────────

def _print_results(results: List[Dict]) -> None:
    header = (
        f"{'Backend':<14}  {'t_rp':>9}  {'t_model':>9}  "
        f"{'t_bwd':>9}  {'t_other':>9}  {'t_total':>9}  "
        f"{'RP%':>6}  {'smp/s':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['label']:<14}  "
            f"{r['t_rp_ms']:>7.1f}ms  "
            f"{r['t_model_ms']:>7.1f}ms  "
            f"{r['t_bwd_ms']:>7.1f}ms  "
            f"{r['t_other_ms']:>7.1f}ms  "
            f"{r['t_total_ms']:>7.1f}ms  "
            f"{r['rp_pct']:>5.1f}%  "
            f"{r['smp_s']:>6.0f}"
        )
    print(sep)

    if len(results) == 2:
        cpu_r, gpu_r = results
        speedup = cpu_r["t_total_ms"] / max(gpu_r["t_total_ms"], 1e-9)
        rp_speedup = cpu_r["t_rp_ms"] / max(gpu_r["t_rp_ms"], 1e-9)
        print(f"\n  GPU RP speedup  — total step:     {speedup:.2f}×")
        print(f"  GPU RP speedup  — RP stage only:  {rp_speedup:.2f}×")


def _print_recommendation(results: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("RECOMMENDATION — best method before LOTSA")
    print("=" * 60)

    gpu_result = next((r for r in results if "GPU" in r["label"]), None)
    cpu_result = next((r for r in results if "CPU" in r["label"]), None)

    if gpu_result and cpu_result:
        rp_pct_cpu = cpu_result["rp_pct"]
        rp_pct_gpu = gpu_result["rp_pct"]
        speedup = cpu_result["t_total_ms"] / max(gpu_result["t_total_ms"], 1e-9)

        if rp_pct_cpu > 50:
            print(f"  RP is {rp_pct_cpu:.0f}% of total step time with CPU-pyts.")
            print(f"  → Switching to GPU RP gives {speedup:.1f}× speedup.")
        elif rp_pct_cpu > 20:
            print(f"  RP is {rp_pct_cpu:.0f}% of step time — noticeable overhead.")
            print(f"  → GPU RP recommended ({speedup:.1f}× faster total).")
        else:
            print(f"  RP overhead is low ({rp_pct_cpu:.0f}%). Both backends are fine.")

    print("""
Strategy for efficient training on ICML before scaling to LOTSA:

  1. use_gpu_rp: true  (add to model config)
     — RP stays on GPU, no CPU↔GPU transfers per batch
     — Supports per_channel and mean MV strategies natively
     — Falls back to CPU pyts for GASF / MTF / STFT / shuffled modes

  2. dataset_type: icml  (8 datasets, ~10-50K samples total)
     — Fixed-length windows, no padding overhead
     — Fast to iterate: an epoch is seconds on A100

  3. two_views: true  (already enabled in lotsa_clip.yaml)
     — Two independent crops per series → stronger CLIP positive pairs
     — No extra compute cost

  4. use_amp: true  (already set in lotsa_clip.yaml)
     — Half-precision forward/backward, large gains on A100

  5. num_workers: 4 + pin_memory: true
     — Overlap data loading with GPU compute

  6. Train ~50-100 epochs on ICML first to validate the full pipeline,
     then switch dataset_type: lotsa and continue from the checkpoint.
""")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark RP overhead vs model compute")
    p.add_argument("--config", type=Path,
                   default=script_dir / "smoke_config.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets" / "ETT-small")
    p.add_argument("--dataset_type", default="icml",
                   choices=["icml", "cronos", "lotsa"])
    p.add_argument("--n_batches", type=int, default=50,
                   help="Number of timed batches per backend (5 warmup added automatically)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = tu.prepare_device("auto")

    print(f"Device      : {device}")
    print(f"Data        : {args.data_dir}  [{args.dataset_type}]")
    print(f"Timed steps : {args.n_batches} (+ 5 warmup each)")

    config = tu.load_config(args.config)
    model_cfg = config.model
    train_cfg = config.training

    # ── data ─────────────────────────────────────────────────────────────────
    print("\nBuilding dataloaders …")
    train_loader, _ = build_time_series_dataloaders(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        batch_size=int(config.data.get("batch_size", 64)),
        val_batch_size=int(config.data.get("val_batch_size", 32)),
        num_workers=int(config.data.get("num_workers", 0)),
        pin_memory=bool(config.data.get("pin_memory", False)),
        val_ratio=float(config.data.get("val_ratio", 0.1)),
        seed=args.seed,
    )
    bs = train_loader.batch_size
    print(f"  batch_size = {bs},  batches/epoch = {len(train_loader)}")

    # ── shared temporal encoder + projections (same weights for both runs) ───
    tu.set_seed(args.seed)
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    proj    = build_projection_head(encoder).to(device)
    vproj_ref = None   # rebuilt per run to keep weight init identical

    noise_std = float(train_cfg.get("noise_std", 0.01))

    all_results = []

    for use_gpu, label in [(False, "CPU-pyts"), (True, "GPU-torch")]:
        tu.set_seed(args.seed)
        visual = _build_timed_encoder(model_cfg, use_gpu_rp=use_gpu).to(device)
        vproj  = build_projection_head(visual).to(device)

        print(f"\n{'─'*50}")
        print(f"  Backend: {label}")
        result = _run_benchmark(
            visual, encoder, proj, vproj,
            train_loader, device,
            n_batches=args.n_batches,
            noise_std=noise_std,
            label=label,
        )
        all_results.append(result)

        print(f"  t_rp={result['t_rp_ms']:.1f}ms  "
              f"t_model={result['t_model_ms']:.1f}ms  "
              f"t_bwd={result['t_bwd_ms']:.1f}ms  "
              f"total={result['t_total_ms']:.1f}ms  "
              f"RP%={result['rp_pct']:.1f}%  "
              f"smp/s={result['smp_s']:.0f}")

    _print_results(all_results)
    _print_recommendation(all_results)


if __name__ == "__main__":
    main()
