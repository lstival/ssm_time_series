"""
Ablation B — Visual Encoder Architecture Justification
========================================================
Compares four encoder architectures to justify the use of a *separate* visual
encoder with a Mamba backbone:

  Variant               | Training objective           | Description
  ----------------------|------------------------------|--------------------------------------------
  no_visual             | SimCLR (temporal↔temporal)   | Two augmented views of the SAME temporal
                        |                              | sequence through the temporal encoder.
                        |                              | A principled unimodal SSL baseline
                        |                              | (no visual branch used at all).
  shared_1d             | SimCLR (temporal↔temporal)   | Single MambaEncoder serves as BOTH
                        |                              | temporal and visual branch (shared wts).
  sep_cnn_only          | CLIP (temporal↔visual)       | Separate visual branch with CNN
                        |                              | projection only; no Mamba SSM blocks.
  sep_mamba_1d (base)   | CLIP (temporal↔visual)       | Current full CM-Mamba architecture
                        |                              | (separate Mamba visual encoder).

Protocol
--------
  1. Each variant is pre-trained on LOTSA for `--train_epochs` epochs.
  2. Linear-probe evaluation on ETTm1, ETTh1, Weather for H ∈ {96, 192, 336, 720}.
  3. Results written to results/ablation_B_encoder_arch.csv.

  Note on `no_visual`: this variant trains with a SimCLR-style objective where
  both query (q) and key (k) come from augmented views of the SAME temporal
  sequence through the same MambaEncoder. This is a principled SSL baseline
  (equivalent to SimCLR without an RP branch). It is NOT "no training" — the
  encoder still learns self-supervised temporal representations via contrastive
  learning between augmented temporal views.

Usage
-----
  python src/experiments/ablation_B_encoder_arch.py \
      --config src/configs/lotsa_clip.yaml \
      --train_epochs 20 --probe_epochs 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torchF

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import (
    build_time_series_dataloaders,
    clip_contrastive_loss,
    extract_sequence,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
    build_projection_head,
)
from models.mamba_encoder import MambaEncoder
from models.mamba_visual_encoder import MambaVisualEncoder, _InputConv
from time_series_loader import TimeSeriesDataModule

VARIANTS: List[str] = ["no_visual", "shared_1d", "sep_cnn_only", "sep_mamba_1d"]
PROBE_DATASETS: List[str] = ["ETTm1.csv", "ETTh1.csv", "weather.csv"]
HORIZONS: List[int] = [96, 192, 336, 720]


# ── CNN-only visual encoder ───────────────────────────────────────────────────

class CNNOnlyVisualEncoder(nn.Module):
    """RP → single Conv2D projection → mean pool → output embedding.

    No Mamba SSM blocks. Used to test whether the recurrent structure in the
    visual branch is necessary.
    """

    def __init__(
        self,
        input_dim: int = 32,
        model_dim: int = 256,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        from models.mamba_visual_encoder import tokenize_sequence
        self._tokenize = lambda x: tokenize_sequence(x, token_size=input_dim)
        self.input_dim = input_dim
        self.input_proj = _InputConv(token_len=input_dim, out_dim=model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from models.utils import time_series_2_recurrence_plot
        import numpy as np

        tokens = self._tokenize(x)          # (B, windows, L, F)
        B, windows, L, F = tokens.shape
        tokens_np = tokens.permute(0, 1, 3, 2).reshape(B * windows, F, L)
        tokens_np = tokens_np.detach().cpu().numpy()
        imgs = time_series_2_recurrence_plot(tokens_np)  # (N, L, L) or (N, F, L, L)
        if imgs.ndim == 4:
            imgs = imgs.mean(axis=1)
        imgs_t = torch.from_numpy(imgs).float().to(x.device).view(B, windows, L, L)
        feat = self.input_proj(imgs_t)      # (B, windows, model_dim)
        feat = feat.mean(dim=1)             # (B, model_dim)
        return self.output_proj(self.norm(feat))


# ── training helpers ──────────────────────────────────────────────────────────

def _clip_train_epoch_variant(
    encoder: nn.Module,
    visual: Optional[nn.Module],
    proj: nn.Module,
    vproj: Optional[nn.Module],
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.01,
    variant: str = "sep_mamba_1d",
) -> float:
    encoder.train(); proj.train()
    if visual is not None: visual.train()
    if vproj is not None:  vproj.train()

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

        if variant == "no_visual":
            # SimCLR-style: two augmented temporal views → temporal encoder (no visual branch)
            k = torchF.normalize(proj(encoder(x_k)), dim=1)
        elif variant == "shared_1d":
            # Visual branch IS the temporal encoder
            k = torchF.normalize(proj(encoder(x_k)), dim=1)
        else:
            k = torchF.normalize(vproj(visual(x_k)), dim=1)

        loss = clip_contrastive_loss(q, k)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()); n += 1

    return total / max(1, n)


def _build_variant(
    model_cfg: Dict,
    variant: str,
    device: torch.device,
) -> Tuple[nn.Module, Optional[nn.Module], nn.Module, Optional[nn.Module], torch.optim.Optimizer]:
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    proj = build_projection_head(encoder).to(device)
    params = list(encoder.parameters()) + list(proj.parameters())

    if variant == "no_visual":
        visual = vproj = None

    elif variant == "shared_1d":
        visual = encoder   # shared weights
        vproj = proj
        # params already included above

    elif variant == "sep_cnn_only":
        visual = CNNOnlyVisualEncoder(
            input_dim=int(model_cfg.get("input_dim", 32)),
            model_dim=int(model_cfg.get("model_dim", 256)),
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        ).to(device)
        vproj = build_projection_head(visual).to(device)
        params += list(visual.parameters()) + list(vproj.parameters())

    elif variant == "sep_mamba_1d":
        visual = tu.build_visual_encoder_from_config(model_cfg, rp_mode="correct").to(device)
        vproj = build_projection_head(visual).to(device)
        params += list(visual.parameters()) + list(vproj.parameters())

    else:
        raise ValueError(f"Unknown variant: {variant}")

    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    return encoder, visual, proj, vproj, optimizer


# ── linear probe ─────────────────────────────────────────────────────────────

def _probe_evaluate(
    encoder: nn.Module,
    visual: Optional[nn.Module],
    data_dir: Path,
    dataset_csv: str,
    horizons: List[int],
    device: torch.device,
    probe_epochs: int = 30,
    batch_size: int = 64,
) -> Dict[int, Dict[str, float]]:
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

    encoder.eval()
    if visual is not None: visual.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    if visual is not None:
        for p in visual.parameters(): p.requires_grad_(False)

    def _embed(batch):
        seq = batch[0].to(device).float()
        seq = prepare_sequence(seq)
        x = reshape_multivariate_series(seq)
        with torch.no_grad():
            ze = encoder(x)
            zv = visual(x) if visual is not None else ze
        return torch.cat([ze, zv], dim=1)

    def _collect(loader):
        zs, ys = [], []
        for batch in loader:
            zs.append(_embed(batch))
            y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
            ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    Z_tr, Y_tr = _collect(train_loader)
    feat_dim = Z_tr.shape[1]
    results: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        probe = nn.Linear(feat_dim, H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(probe_epochs):
            probe.train()
            perm = torch.randperm(Z_tr.shape[0], device=device)
            for i in range(0, Z_tr.shape[0], 64):
                idx = perm[i:i+64]
                z_b = Z_tr[idx]
                y_b = Y_tr[idx]
                if y_b.ndim == 3: y_b = y_b[:, :H, 0]
                elif y_b.ndim == 2: y_b = y_b[:, :H]
                if y_b.shape[1] < H: continue
                loss = torchF.mse_loss(probe(z_b), y_b)
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

    for p in encoder.parameters(): p.requires_grad_(True)
    if visual is not None:
        for p in visual.parameters(): p.requires_grad_(True)
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_B")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--pretrain_data_dir", type=Path,
                   default=src_dir.parent / "data",
                   help="Data dir for CLIP pre-training (override for smoke tests)")
    p.add_argument("--variants", nargs="+", default=VARIANTS,
                   choices=VARIANTS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _ = build_time_series_dataloaders(
        data_dir=str(args.pretrain_data_dir),
        dataset_name=config.data.get("dataset_name", ""),
        dataset_type=config.data.get("dataset_type", "lotsa"),
        batch_size=int(config.data.get("batch_size", 256)),
        val_batch_size=int(config.data.get("val_batch_size", 128)),
        num_workers=int(config.data.get("num_workers", 4)),
        pin_memory=bool(config.data.get("pin_memory", True)),
        val_ratio=float(config.data.get("val_ratio", 0.1)),
        cronos_kwargs=dict(config.data.get("cronos_kwargs", {})),
        seed=args.seed,
    )

    rows: List[Dict] = []

    for variant in args.variants:
        print(f"\n{'='*60}\nVariant: {variant}\n{'='*60}")
        tu.set_seed(args.seed)

        encoder, visual, proj, vproj, optimizer = _build_variant(
            config.model, variant, device
        )

        t0 = time.time()
        for ep in range(args.train_epochs):
            loss = _clip_train_epoch_variant(
                encoder, visual, proj, vproj, train_loader, optimizer, device,
                noise_std=float(config.training.get("noise_std", 0.01)),
                variant=variant,
            )
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  epoch {ep+1}/{args.train_epochs}  loss={loss:.4f}")
        elapsed = time.time() - t0
        ms_per_batch = 1000 * elapsed / (args.train_epochs * max(1, len(train_loader)))
        print(f"  Done — {elapsed:.0f}s ({ms_per_batch:.1f} ms/batch)")

        for ds in PROBE_DATASETS:
            metrics = _probe_evaluate(
                encoder, visual,
                data_dir=args.data_dir, dataset_csv=ds,
                horizons=HORIZONS, device=device,
                probe_epochs=args.probe_epochs,
            )
            for H, m in metrics.items():
                rows.append({
                    "variant": variant, "dataset": ds.replace(".csv", ""),
                    "horizon": H,
                    "mse": f"{m['mse']:.4f}", "mae": f"{m['mae']:.4f}",
                })
                print(f"  {ds}  H={H:3d}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    out_csv = args.results_dir / "ablation_B_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "dataset", "horizon", "mse", "mae"])
        writer.writeheader(); writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
