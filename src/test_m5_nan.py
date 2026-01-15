"""Quick NaN/Inf smoke test for Chronos supervised training on M5.

What it does
- Loads ONLY one Chronos dataset (default: m5)
- Forces global min-max normalization via a temporary override YAML
- Samples a random fraction of the training dataset (default: 10%)
- Runs a short training loop using the same batch preparation and model head
  as `chronos_supervised_training.py`
- Fails fast (exit code 1) if it encounters NaN/Inf in:
    - loss
    - gradients
    - model parameters

Usage (PowerShell)
- `python src/test_m5_nan.py`
- `python src/test_m5_nan.py --max-steps 200 --epochs 2`
- `python src/test_m5_nan.py --dataset m5 --fraction 0.1 --disable-amp`

Notes
- Run this from the repo root so `python src/...` adds `src/` to `sys.path`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

import training_utils as tu
import util as u
from moco_training import build_dataloaders, prepare_dataset, resolve_path

# Reuse the exact model + batch splitting logic from the training script.
from chronos_supervised_training import ChronosForecastModel, _prepare_forecast_batch


def _write_cronos_override(
    cronos_config_path: Path,
    *,
    out_path: Path,
    dataset_name: str,
    normalize_mode: str,
    split: str = "train",
) -> Path:
    """Create a Chronos loader YAML override for a single dataset."""
    import yaml

    with open(cronos_config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    raw_cfg["datasets_to_load"] = [str(dataset_name)]
    raw_cfg["split"] = str(split)
    raw_cfg["normalize"] = True
    raw_cfg["normalize_mode"] = str(normalize_mode)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_cfg, handle, sort_keys=False)

    return out_path


def _is_finite_tensor(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _nonfinite_counts(t: torch.Tensor) -> str:
    t = t.detach()
    nan = torch.isnan(t)
    posinf = torch.isposinf(t)
    neginf = torch.isneginf(t)
    total = t.numel()
    return (
        f"total={total} nan={int(nan.sum().item())} "
        f"+inf={int(posinf.sum().item())} -inf={int(neginf.sum().item())}"
    )


def _minmax_stats(t: torch.Tensor) -> str:
    t = t.detach()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return "(no finite values)"
    return f"min={finite.min().item():.6g} max={finite.max().item():.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "chronos_supervised.yaml"),
        help="Path to chronos_supervised.yaml",
    )
    parser.add_argument("--dataset", type=str, default="m5", help="Chronos dataset name")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of training set to sample (0 < fraction <= 1)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="How many epochs to run")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max batches per epoch (for faster smoke test)",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="global_minmax",
        help="Normalization mode to force in Chronos loader YAML",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Autocast dtype when AMP is enabled (cuda only).",
    )
    parser.add_argument("--disable-amp", action="store_true", help="Disable AMP even on CUDA")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--pred-len", type=int, default=None, help="Override pred_len")

    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = tu.load_config(config_path)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    tu.set_seed(cfg.seed)

    device = tu.prepare_device(cfg.device)

    data_cfg: Dict[str, Any] = dict(cfg.data)
    training_cfg: Dict[str, Any] = dict(cfg.training)

    if args.disable_amp:
        training_cfg["use_amp"] = False

    pred_len = int(args.pred_len) if args.pred_len is not None else int(training_cfg.get("pred_len", 96))

    # Resolve base Chronos loader config.
    cronos_config = data_cfg.get("cronos_config")
    if cronos_config is None:
        cronos_config = config_path.parent / "cronos_loader_example.yaml"
    cronos_config = resolve_path(config_path.parent, cronos_config)
    if cronos_config is None or not Path(cronos_config).exists():
        raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

    # Write override config that loads only the requested dataset + minmax normalization.
    out_dir = Path("checkpoints") / "nan_smoke_tests" / str(args.dataset)
    override_path = _write_cronos_override(
        Path(cronos_config),
        out_path=out_dir / "cronos_loader_override.yaml",
        dataset_name=str(args.dataset),
        normalize_mode=str(args.normalize_mode),
        split="train",
    )

    # IMPORTANT: `moco_training.prepare_dataset()` resolves relative `cronos_config`
    # paths relative to `src/configs/`. Use an absolute path to avoid accidental
    # prefixing like `src/configs/checkpoints/...`.
    override_path = override_path.resolve()

    data_cfg["cronos_config"] = str(override_path)
    data_cfg["dataset_name"] = str(args.dataset)
    data_cfg["patch_length"] = None  # do not override YAML patch_length

    dataset = prepare_dataset(config_path, data_cfg)
    total = len(dataset)
    if not (0.0 < float(args.fraction) <= 1.0):
        raise ValueError("--fraction must be in (0, 1]")

    subset_size = max(1, int(total * float(args.fraction)))
    gen = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(total, generator=gen)[:subset_size].tolist()
    subset = Subset(dataset, indices)

    # Build a loader from the subset.
    batch_size = int(args.batch_size) if args.batch_size is not None else int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader, _ = build_dataloaders(
        subset,
        None,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Infer feature dim from the first batch.
    first_batch = next(iter(train_loader))
    sample_seq = u.prepare_sequence(u.extract_sequence(first_batch))
    feature_dim = int(sample_seq.shape[-1])

    encoder = tu.build_encoder_from_config(cfg.model)
    model = ChronosForecastModel(encoder, feature_dim, target_dim=feature_dim, pred_len=pred_len).to(device)

    optimizer = tu.build_optimizer(model, training_cfg)
    criterion = nn.MSELoss()

    amp_enabled = bool(training_cfg.get("use_amp", False)) and (device.type == "cuda")
    amp_dtype = torch.float16 if str(args.amp_dtype).lower() == "float16" else torch.bfloat16
    # BF16 generally does not need GradScaler (and can be less stable with it).
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0)) or None

    print(
        "NaN smoke test config: "
        f"dataset={args.dataset} total={total} subset={subset_size} "
        f"batch_size={batch_size} pred_len={pred_len} device={device} amp={amp_enabled} amp_dtype={args.amp_dtype}"
    )
    print(f"Chronos config override: {override_path}")

    model.train()

    for epoch in range(int(args.epochs)):
        for step, batch in enumerate(train_loader):
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                break

            context, target, effective_pred_len = _prepare_forecast_batch(batch, pred_len, device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                preds_full = model(context)
                preds = preds_full[:, :effective_pred_len, :]

            with autocast(enabled=False):
                loss = criterion(preds.float(), target.float())

            if not torch.isfinite(loss).all():
                print(
                    f"FAIL: non-finite loss at epoch={epoch} step={step} "
                    f"loss={loss.detach().cpu().item()}\n"
                    f"  context: {_minmax_stats(context)}\n"
                    f"  target:  {_minmax_stats(target)}\n"
                    f"  preds:   {_minmax_stats(preds)}"
                )
                raise SystemExit(1)

            scaler.scale(loss).backward()

            # Check grads (after unscale, before clip/step).
            scaler.unscale_(optimizer)

            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if not _is_finite_tensor(p.grad):
                    scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
                    print(
                        f"FAIL: non-finite grad at epoch={epoch} step={step} param={name}\n"
                        f"  grad: {_minmax_stats(p.grad)}\n"
                        f"  counts: {_nonfinite_counts(p.grad)}\n"
                        f"  amp_scale: {scale:.6g}"
                    )
                    raise SystemExit(1)

            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            # Check params after update.
            for name, p in model.named_parameters():
                if not _is_finite_tensor(p.data):
                    scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
                    print(
                        f"FAIL: non-finite param after step at epoch={epoch} step={step} param={name}\n"
                        f"  value: {_minmax_stats(p.data)}\n"
                        f"  counts: {_nonfinite_counts(p.data)}\n"
                        f"  amp_scale: {scale:.6g}"
                    )
                    raise SystemExit(1)

            if step % 50 == 0:
                lr = float(optimizer.param_groups[0].get("lr", 0.0))
                print(f"ok epoch={epoch} step={step} loss={loss.item():.6g} lr={lr:.6g}")

    print("PASS: no NaN/Inf detected in loss/grad/params.")


if __name__ == "__main__":
    main()
