#!/usr/bin/env python
"""Smoke test: loads the smallest dataset, checks data throughput and model correctness.

Run from the src/ directory:
    python smoke_test.py

Checks:
  1. Dataset loading (monash_m1_yearly — 156 KB, the smallest real dataset)
  2. DataLoader throughput (samples/sec)
  3. Model instantiation (MambaEncoder + MambaVisualEncoder)
  4. Forward pass — correct output shape and finite values
  5. Training steps — finite loss, gradient flow
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import torch
import yaml

# Allow running directly from src/
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import training_utils as tu
import util as u
from contrastive_training import ContrastiveModel, build_dataloaders, split_dataset
from dataloaders.cronos_loader_ts import load_cronos_time_series_dataset
from models.mamba_block import _FAST_MAMBA

# ── Configuration ───────────────────────────────────────────────────────────
DATASET       = "exchange_rate"          # 1.7 MB — smallest dataset with series long enough for patch_length=96
DATA_DIR      = SRC_DIR.parent / "data"
PATCH_LENGTH  = 96
BATCH_SIZE    = 32
N_BATCHES     = 5                        # batches used for throughput/training checks
TEMPERATURE   = 0.2

# Minimal model — small enough to run fast on CPU too
_SMALL_MODEL_CFG = dict(
    input_dim=32, model_dim=64, depth=2, state_dim=16,
    conv_kernel=3, expand_factor=2.0, embedding_dim=64,
    dropout=0.0, pooling="mean",
)

# ── Helpers ─────────────────────────────────────────────────────────────────

def _write_cronos_cfg(directory: Path) -> Path:
    cfg = {
        "datasets_to_load": [DATASET],
        "split": "train",
        "patch_length": PATCH_LENGTH,
        "target_dtype": "float64",
        "normalize": True,
        "normalize_mode": "global_standard",
    }
    path = directory / "smoke_cronos.yaml"
    path.write_text(yaml.dump(cfg))
    return path


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


# ── Test sections ────────────────────────────────────────────────────────────

def test_data_loading() -> tuple[object, list]:
    """Load dataset and iterate N_BATCHES; return (dataset, first_batches)."""
    print("── 1. Data loading")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        cronos_cfg = _write_cronos_cfg(Path(tmp))
        dataset = load_cronos_time_series_dataset(
            str(cronos_cfg),
            split="train",
            patch_length=PATCH_LENGTH,
            load_kwargs={
                "offline_cache_dir": str(DATA_DIR),
                "force_offline": True,
            },
        )
    t_load = time.perf_counter() - t0
    assert len(dataset) > 0, "Dataset is empty"
    _check("dataset loaded", True, f"{len(dataset):,} patches  in  {t_load:.2f}s")

    # DataLoader throughput
    print("\n── 2. DataLoader throughput")
    train_ds, val_ds = split_dataset(dataset, val_ratio=0.1, seed=42)
    train_loader, _ = build_dataloaders(
        train_ds, val_ds,
        batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    batches = []
    t0 = time.perf_counter()
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i + 1 >= N_BATCHES:
            break
    t_dl = time.perf_counter() - t0

    sps = len(batches) * BATCH_SIZE / t_dl
    _check("dataloader iterable", len(batches) == N_BATCHES, f"{sps:.0f} samples/s")

    return dataset, batches


def test_model(batches: list, device: torch.device) -> ContrastiveModel:
    """Build encoder, check parameter count, return model."""
    print("\n── 3. Model instantiation")
    feature_dim = u.prepare_sequence(u.extract_sequence(batches[0])).shape[-1]

    # Standard encoder
    encoder = tu.build_encoder_from_config(_SMALL_MODEL_CFG)
    model = ContrastiveModel(encoder, input_dim=feature_dim).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _check("MambaEncoder created", True, f"{n:,} params")

    # Visual encoder (smoke-check instantiation only — RP computation is slow)
    vis_encoder = tu.build_visual_encoder_from_config(
        {**_SMALL_MODEL_CFG, "pooling": "cls"}, rp_mode="correct"
    )
    n_vis = sum(p.numel() for p in vis_encoder.parameters() if p.requires_grad)
    _check("MambaVisualEncoder created", True, f"{n_vis:,} params")

    return model


def test_forward(model: ContrastiveModel, batches: list, device: torch.device) -> bool:
    """Single forward pass — check shape and finite values."""
    print("\n── 4. Forward pass")
    seq = u.prepare_sequence(u.extract_sequence(batches[0])).to(device).float()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(seq)
    t_fwd = (time.perf_counter() - t0) * 1000

    shape_ok  = out.ndim == 2 and out.shape[0] == seq.shape[0]
    finite_ok = torch.isfinite(out).all().item()
    ok = True
    ok &= _check("output shape", shape_ok,
                 f"got {tuple(out.shape)}, expected ({seq.shape[0]}, {out.shape[-1]})")
    ok &= _check("output finite", bool(finite_ok))
    ok &= _check("forward speed", True, f"{t_fwd:.1f} ms  (batch={seq.shape[0]})")
    return ok


def test_training(model: ContrastiveModel, batches: list, device: torch.device) -> bool:
    """Run N_BATCHES gradient steps, check losses are finite."""
    print("\n── 5. Training steps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    t0 = time.perf_counter()
    for batch in batches:
        loss = u.forward_contrastive_batch(
            model, batch,
            device=device,
            temperature=TEMPERATURE,
            use_amp=False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    t_steps = (time.perf_counter() - t0) * 1000

    finite_ok = all(torch.isfinite(torch.tensor(l)).item() for l in losses)
    step_ms   = t_steps / len(losses)
    loss_str  = "  ".join(f"{l:.4f}" for l in losses)

    ok = True
    ok &= _check("losses finite", finite_ok, f"[{loss_str}]")
    ok &= _check("training speed", True,
                 f"{step_ms:.1f} ms/step avg  over {len(losses)} steps")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*62}")
    print(f"  SMOKE TEST  —  {DATASET}")
    print(f"  device : {device}")
    print(f"  fast SSM kernel : {'YES  (mamba_ssm)' if _FAST_MAMBA else 'NO   (pure-PyTorch fallback)'}")
    print(f"{'='*62}\n")

    all_ok = True
    try:
        dataset, batches = test_data_loading()
    except Exception as exc:
        _check("data loading", False, str(exc))
        print("\n  Cannot continue without data.\n")
        return 1

    try:
        model = test_model(batches, device)
        all_ok &= test_forward(model, batches, device)
        all_ok &= test_training(model, batches, device)
    except Exception as exc:
        _check("unexpected error", False, str(exc))
        all_ok = False

    print(f"\n{'='*62}")
    result = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
    print(f"  {result}")
    print(f"{'='*62}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
