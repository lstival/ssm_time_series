"""
Stage 1 — MoMS Zero-Shot: train on LOTSA+Chronos, eval on 7 ICML benchmarks.

The encoder is fully frozen. Only MoMS prompts + heads are trained.
norm_mode='revin' so the model is agnostic to LOTSA min-max vs ICML StandardScaler.

Saves:
    results_dir/mop_zeroshot_<encoder>_results.csv
    results_dir/mop_zeroshot_<encoder>_checkpoint.pt   ← used as Stage 2 warm-start
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent
root_dir   = src_dir.parent
for p in (src_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from models.mop_crossattn import (
    MoPCrossAttnModel,
    MoPCrossAttnModelA,
    MoPCrossAttnModelB,
    MoPCrossAttnModelC,
)
from time_series_loader import TimeSeriesDataModule
from dataloaders.local_dataset_loader import build_combined_dataloaders
from dataloaders.gift_eval_loader import ALL_GIFT_SSL_SUBSETS


class ZeroEncoder(nn.Module):
    """Drop-in visual encoder returning dim-0 tensor — cat([ze, zv]) collapses to ze only."""
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
]
HORIZONS = [96, 192, 336, 720]


def build_icml_linear_probes(encoder, visual, icml_dir, ctx, batch_size,
                              probe_epochs, fusion_mode, device):
    """Train one linear probe per horizon on ICML train splits.

    Probes are Linear(embed_dim → H) where embed_dim is the frozen encoder
    output (enc_dim*2 for concat, enc_dim for film). They are returned as
    teacher models to be used as a distillation signal during MoMS training —
    they are NOT copied into MoMS heads (different input dimensionality).

    Returns:
        probes: dict {H: nn.Linear} frozen, on CPU
        Z_all:  (N, embed_dim) tensor of all ICML train embeddings, on CPU
        Y_all:  (N, max_H) tensor of all ICML train targets, on CPU
    """
    from util import prepare_sequence, reshape_multivariate_series
    max_h = max(HORIZONS)

    # Collect embeddings from all ICML train splits
    Z_all, Y_all = [], []
    for ds in ICML_DATASETS:
        ds_tag = ds.replace(".csv", "")
        resolved = str(icml_dir)
        for cand in icml_dir.rglob(ds):
            resolved = str(cand.parent); break
        try:
            mod = TimeSeriesDataModule(
                dataset_name=ds, data_dir=resolved,
                batch_size=batch_size, val_batch_size=batch_size,
                num_workers=0, pin_memory=False, normalize=True,
                train=True, val=False, test=False,
                sample_size=(ctx, 0, max_h), scaler_type="standard",
            )
            mod.setup()
            if not mod.train_loaders:
                continue
            loader = mod.train_loaders[0]
        except Exception as e:
            print(f"  [probe] skip {ds_tag}: {e}"); continue

        zs, ys = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                x = prepare_sequence(x)
                x_ci = reshape_multivariate_series(x)   # (B*C, 1, L)
                ze = encoder(x_ci)
                z = ze if fusion_mode == "film" else torch.cat([ze, visual(x_ci)], dim=-1)
                zs.append(z.cpu())
                if y.ndim == 3:
                    B, L, C = y.shape
                    y = y.permute(0, 2, 1).reshape(B * C, L)
                ys.append(y.cpu())
        if zs:
            Z_all.append(torch.cat(zs))
            Y_all.append(torch.cat(ys))
            print(f"  [probe] collected {ds_tag}: {Z_all[-1].shape[0]} samples")

    if not Z_all:
        print("  [probe] No ICML data — skipping probes"); return {}, None, None

    Z_all = torch.cat(Z_all)   # (N, embed_dim)
    Y_all = torch.cat(Y_all)   # (N, max_h)
    embed_dim = Z_all.shape[1]
    N = Z_all.shape[0]

    probes = {}
    for H in HORIZONS:
        print(f"  [probe] Training Linear({embed_dim}→{H}) for {probe_epochs} epochs ...", flush=True)
        probe = nn.Linear(embed_dim, H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for ep in range(probe_epochs):
            perm = torch.randperm(N)
            ep_loss, n_b = 0.0, 0
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                zb = Z_all[idx].to(device)
                yb = Y_all[idx, :H].to(device)
                loss = F.mse_loss(probe(zb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += loss.item(); n_b += 1
            if (ep + 1) % 5 == 0 or ep == probe_epochs - 1:
                print(f"    epoch {ep+1:2d}/{probe_epochs}  loss={ep_loss/max(n_b,1):.4f}")
        probe.eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        probes[H] = probe.cpu()
        print(f"  [probe] H={H} done.")

    return probes, Z_all, Y_all


def parse_args():
    p = argparse.ArgumentParser("MoMS Stage 1 — Zero-Shot (LOTSA+Chronos → ICML)")
    p.add_argument("--encoder_name",    required=True,
                   help="Short label used in output filenames, e.g. simclr, byol, clip, gram")
    p.add_argument("--checkpoint_dir",  type=Path, required=True)
    p.add_argument("--config",          type=Path, required=True)
    p.add_argument("--icml_data_dir",   type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir",     type=Path, default=root_dir / "results" / "moms_full_pipeline")
    p.add_argument("--epochs",          type=int,  default=50)
    p.add_argument("--batch_size",      type=int,  default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--hidden_dim",      type=int,  default=512)
    p.add_argument("--num_prompts",     type=int,  default=16)
    p.add_argument("--context_length",  type=int,  default=336)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--num_workers",     type=int,  default=4)
    p.add_argument("--unimodal",        action="store_true",
                   help="Use temporal encoder only (visual branch = zeros)")
    p.add_argument("--fusion_mode",     type=str, default="concat",
                   choices=["concat", "film"],
                   help="How z_e and z_v are fused: concat (default) or film")
    p.add_argument("--output_suffix",   type=str, default="",
                   help="Suffix appended to result/checkpoint filenames")
    p.add_argument("--include_gift",    action="store_true",
                   help="Add GIFT-Eval train splits to MoMS Stage 1 training corpus")
    p.add_argument("--include_icml",    action="store_true",
                   help="Add ICML benchmark train splits to MoMS Stage 1 training corpus")
    p.add_argument("--probe_distill",   action="store_true",
                   help="Train linear probes on ICML train splits and use them as distillation "
                        "teachers during MoMS LOTSA training (probe_loss added to forecast loss)")
    p.add_argument("--probe_epochs",    type=int, default=20,
                   help="Epochs to train ICML linear probes (default 20)")
    p.add_argument("--distill_alpha",   type=float, default=0.5,
                   help="Weight of distillation loss relative to forecast loss (default 0.5)")
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--save_predictions", action="store_true",
                   help="Save per-dataset predictions/targets to .npz")
    p.add_argument("--predictions_dir", type=Path, default=None,
                   help="Directory for saved prediction .npz files")
    p.add_argument("--psn",             action="store_true",
                   help="Apply Per-Sample Normalization (instance norm on context window) "
                        "during both training and evaluation instead of RevIN")
    # Head architecture options
    p.add_argument("--head_type",       type=str, default="linear", choices=["linear", "mlp"],
                   help="Head architecture: linear (default) or mlp (2-layer GELU)")
    p.add_argument("--use_ln_head",     action="store_true",
                   help="Add LayerNorm before each forecast head")
    p.add_argument("--head_dropout",    type=float, default=0.0,
                   help="Dropout inside mlp head (default 0.0)")
    p.add_argument("--skip_linear",     action="store_true",
                   help="Add Linear(context_length→H) skip bypass (DLinear residual)")
    p.add_argument("--skip_only",       action="store_true",
                   help="DLinear-only baseline: skip bypass without encoder/MoP")
    p.add_argument("--mop_crossattn",   action="store_true",
                   help="Use MoPCrossAttnModel (variant D, original)")
    p.add_argument("--crossattn_heads", type=int, default=4,
                   help="Number of attention heads in cross-attn variants (default 4)")
    p.add_argument("--crossattn_variant", type=str, default="D",
                   choices=["D", "A", "B", "C"],
                   help="Cross-attn variant: D=original, A=visual tokens, B=horizon-cond routing, C=shared proj+align")
    p.add_argument("--align_alpha",     type=float, default=0.1,
                   help="Weight of alignment loss in variant C (default 0.1)")
    return p.parse_args()


def resolve_dir(data_dir: Path, ds_name: str) -> str:
    for c in data_dir.rglob(ds_name):
        return str(c.parent)
    return str(data_dir)


def load_encoders(ckpt_dir: Path, config, device, unimodal: bool = False):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    for name in ["time_series_best.pt", "time_series_encoder.pt"]:
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            encoder.load_state_dict(state)
            break
    encoder.eval()

    if unimodal:
        visual = ZeroEncoder().to(device)
    else:
        visual = tu.build_visual_encoder_from_config(config.model).to(device)
        for name in ["visual_encoder_best.pt", "visual_encoder.pt"]:
            p = ckpt_dir / name
            if p.exists():
                state = torch.load(p, map_location=device)
                state = state.get("model_state_dict", state.get("model_state", state))
                visual.load_state_dict(state)
                break
        visual.eval()
    return encoder, visual


def build_icml_test_loaders(icml_dir: Path, ctx: int, max_h: int, bs: int) -> dict:
    loaders = {}
    for ds in ICML_DATASETS:
        tag = ds.replace(".csv", "")
        try:
            mod = TimeSeriesDataModule(
                dataset_name=ds, data_dir=resolve_dir(icml_dir, ds),
                batch_size=bs, val_batch_size=bs, num_workers=0,
                pin_memory=False, normalize=True,
                train=False, val=False, test=True,
                sample_size=(ctx, 0, max_h), scaler_type="standard",
                features='M',
            )
            mod.setup()
            if mod.test_loaders:
                loaders[tag] = mod.test_loaders[0]
        except Exception as e:
            print(f"  [WARN] {tag}: {e}")
    return loaders


def psn_normalize(x: torch.Tensor):
    """Per-Sample Normalization on context window. x: (B, C, L) or (B*C, 1, L)."""
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (x - mean) / std, mean, std


def train(mop: MoPForecastModel, train_loader, args, device,
          probes=None, Z_icml=None, Y_icml=None):
    """Train MoMS on LOTSA corpus.

    If probes/Z_icml/Y_icml are provided (probe_distill mode), a distillation
    loss is added at each step: MoMS is trained to match the probe predictions
    on a random mini-batch of ICML embeddings. This anchors the MoP+head stack
    to the forecast knowledge the probes learned, without any dimension mismatch
    (probes operate on embed_dim, MoMS on hidden_dim — they share the same
    encoder embeddings as input to the distillation mini-batch).
    """
    max_h = max(HORIZONS)
    full_window = args.context_length + max_h
    params = []
    if mop.mop is not None:
        params += list(mop.mop.parameters())
    if mop.heads:
        params += list(mop.heads.parameters())
    if getattr(mop, "skip_heads", None) is not None:
        params += list(mop.skip_heads.parameters())
    # MoPCrossAttnModel-specific trainable modules
    if getattr(mop, "token_proj", None) is not None:
        params += list(mop.token_proj.parameters())
    if getattr(mop, "mop_to_query", None) is not None:
        params += list(mop.mop_to_query.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    it = iter(train_loader)

    alpha = getattr(args, "distill_alpha", 0.5)
    distill = probes is not None and Z_icml is not None and Y_icml is not None
    N_icml = Z_icml.shape[0] if distill else 0

    for epoch in range(1, args.epochs + 1):
        mop.train(); total, n = 0.0, 0
        gn_sum, gn_max, gn_hits = 0.0, 0.0, 0; GN_MAX = 1.0
        for _ in range(args.batches_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader); batch = next(it)

            x_full = batch["target"].to(device).float()   # (B, full_window, 1)
            B, W, C = x_full.shape
            if W < args.context_length + 1:
                continue

            x_in = x_full[:, :args.context_length, :].permute(0, 2, 1)   # (B,C,L)
            y_full = x_full[:, args.context_length:, :]                    # (B,max_h,1)
            h = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
            if y_full.shape[1] < h:
                continue

            y_ch = y_full[:, :h, :].permute(0, 2, 1).reshape(B * C, h, 1)

            if getattr(args, "psn", False):
                x_in_bc = x_in.reshape(B * C, 1, args.context_length)
                x_in_bc, psn_mean, psn_std = psn_normalize(x_in_bc)
                x_in = x_in_bc.reshape(B, C, args.context_length)
                y_ch = (y_ch - psn_mean.reshape(B * C, 1, 1)) / psn_std.reshape(B * C, 1, 1)

            if hasattr(mop, "forward_with_loss"):
                pred, align_loss = mop.forward_with_loss(x_in, h)
            else:
                pred = mop(x_in, h)
                align_loss = None
            forecast_loss = F.mse_loss(pred, y_ch)

            if distill:
                # Sample a random mini-batch from ICML embeddings
                idx = torch.randint(0, N_icml, (B * C,))
                z_b = Z_icml[idx].to(device)           # (B*C, embed_dim)
                y_b = Y_icml[idx, :h].to(device)       # (B*C, h)
                with torch.no_grad():
                    probe_preds = probes[h](z_b)        # (B*C, h) — teacher, frozen

                # Pass the same embeddings through MoP+head
                # z_b are frozen encoder outputs → run through mop.mop + head only
                z_prompted = mop.mop(z_b)              # (B*C, hidden_dim)
                moms_preds = mop.heads[str(h)](z_prompted).squeeze(-1)  # (B*C, h)

                distill_loss = F.mse_loss(moms_preds, probe_preds)
                loss = (1 - alpha) * forecast_loss + alpha * distill_loss
            elif align_loss is not None:
                loss = forecast_loss + getattr(args, "align_alpha", 0.1) * align_loss
            else:
                loss = forecast_loss

            if torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(params, GN_MAX)  # returns pre-clip norm
                opt.step()
                total += loss.item(); n += 1
                gn = float(total_norm)
                if gn == gn:  # not NaN
                    gn_sum += gn; gn_max = max(gn_max, gn)
                    if gn > GN_MAX: gn_hits += 1

        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            gn_mean = gn_sum / max(1, n); gn_rate = gn_hits / max(1, n)
            print(f"  Epoch {epoch:3d}/{args.epochs} loss={total/max(1,n):.4f}  "
                  f"grad[mean={gn_mean:.3f} max={gn_max:.3f} clip_hit={gn_rate:.1%}]")


_MV_CHUNK = 128  # max B*C per forward pass to avoid OOM on high-channel datasets


def evaluate(mop: MoPForecastModel, test_loaders: dict, args, device) -> list:
    results = []
    mop.eval()
    with torch.no_grad():
        for tag, loader in test_loaders.items():
            pred_store = {}
            for H in HORIZONS:
                preds, trues = [], []
                for batch in loader:
                    x = batch[0].to(device).float()
                    y = batch[1].to(device).float()
                    if y.shape[1] < H: continue
                    B, L, C = x.shape
                    x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
                    y_t  = y[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)

                    if getattr(args, "psn", False):
                        x_in, psn_mean, psn_std = psn_normalize(x_in)
                        y_t = (y_t - psn_mean.reshape(B * C, 1, 1)) / psn_std.reshape(B * C, 1, 1)

                    # chunk B*C to avoid OOM on high-channel datasets
                    pred_chunks = []
                    for i in range(0, B * C, _MV_CHUNK):
                        pred_chunks.append(
                            mop.greedy_predict(x_in[i:i+_MV_CHUNK], H, args.context_length)
                        )
                    pred = torch.cat(pred_chunks, dim=0)
                    preds.append(pred); trues.append(y_t)
                if not preds: continue
                pred_cat = torch.cat(preds, dim=0).cpu()
                true_cat = torch.cat(trues, dim=0).cpu()
                # Chunked metric aggregation to avoid GPU OOM on large datasets
                mse_vals = []
                mae_vals = []
                for pred, true in zip(preds, trues):
                    mse_vals.append(torch.mean((pred - true) ** 2).item())
                    mae_vals.append(torch.mean(torch.abs(pred - true)).item())
                mse = sum(mse_vals) / len(mse_vals)
                mae = sum(mae_vals) / len(mae_vals)
                nrmse = mse ** 0.5
                print(f"  {tag:12s} H={H:3d}: MSE={mse:.4f} MAE={mae:.4f} NRMSE={nrmse:.4f}")
                results.append({"dataset": tag, "horizon": H, "mse": mse, "mae": mae, "nrmse": nrmse,
                                 "encoder": args.encoder_name, "stage": "zeroshot"})
                if args.save_predictions:
                    pred_store[f"preds_H{H}"] = pred_cat.numpy()
                    pred_store[f"targets_H{H}"] = true_cat.numpy()
            if args.save_predictions and pred_store:
                pred_dir = args.predictions_dir or (args.results_dir / "predictions" / "zeroshot")
                pred_dir.mkdir(parents=True, exist_ok=True)
                suffix = getattr(args, "output_suffix", "")
                out_path = pred_dir / f"{tag}_zeroshot{suffix}.npz"
                np.savez(
                    out_path,
                    dataset=tag,
                    seed=args.seed,
                    context_length=args.context_length,
                    horizons=np.array(HORIZONS, dtype=np.int32),
                    **pred_store,
                )
                print(f"  Saved predictions → {out_path}")
    return results


def main():
    args = parse_args()
    tu.set_seed(args.seed)
    torch.manual_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)

    mode = "unimodal" if args.unimodal else "bimodal"
    print(f"[Stage 1 Zero-Shot] encoder={args.encoder_name}  mode={mode}")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device, args.unimodal)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    fusion_mode = getattr(args, "fusion_mode", "concat")
    if args.unimodal:
        input_dim = enc_dim
    elif fusion_mode == "film":
        input_dim = enc_dim
    else:
        input_dim = enc_dim * 2

    if getattr(args, "mop_crossattn", False):
        variant   = getattr(args, "crossattn_variant", "D")
        n_heads   = getattr(args, "crossattn_heads", 4)
        norm_mode = "identity" if getattr(args, "psn", False) else "revin"
        _xa_kwargs = dict(
            encoder=encoder, visual_encoder=visual,
            emb_dim=enc_dim,
            num_prompts=args.num_prompts, horizons=HORIZONS,
            target_features=1, freeze_encoders=True,
            n_heads=n_heads, norm_mode=norm_mode,
            mop_hidden_dim=args.hidden_dim,
        )
        if variant == "A":
            mop = MoPCrossAttnModelA(**_xa_kwargs).to(device)
        elif variant == "B":
            mop = MoPCrossAttnModelB(**_xa_kwargs).to(device)
        elif variant == "C":
            mop = MoPCrossAttnModelC(**_xa_kwargs,
                                     align_alpha=getattr(args, "align_alpha", 0.1)).to(device)
        else:
            mop = MoPCrossAttnModel(**_xa_kwargs).to(device)
        print(f"[MoP] cross-attention variant={variant}: enc_dim={enc_dim}, heads={n_heads}")
    else:
        mop = MoPForecastModel(
            encoder=encoder, visual_encoder=visual,
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            num_prompts=args.num_prompts, horizons=HORIZONS,
            target_features=1, freeze_encoders=True,
            norm_mode="identity" if getattr(args, "psn", False) else "revin", scale_cond=False,
            fusion_mode=fusion_mode,
            head_type=getattr(args, "head_type", "linear"),
            use_ln_head=getattr(args, "use_ln_head", False),
            dropout=getattr(args, "head_dropout", 0.0),
            skip_linear=getattr(args, "skip_linear", False),
            skip_only=getattr(args, "skip_only", False),
            context_length=args.context_length,
        ).to(device)

    probes, Z_icml, Y_icml = None, None, None
    if getattr(args, "probe_distill", False):
        print("\nTraining ICML linear probes for distillation...")
        probes, Z_icml, Y_icml = build_icml_linear_probes(
            encoder=encoder, visual=visual,
            icml_dir=args.icml_data_dir,
            ctx=args.context_length,
            batch_size=args.batch_size,
            probe_epochs=getattr(args, "probe_epochs", 20),
            fusion_mode=fusion_mode,
            device=device,
        )
        # Move probes to device for distillation
        probes = {H: p.to(device) for H, p in probes.items()}
        print(f"  Probes ready. Distillation alpha={getattr(args, 'distill_alpha', 0.5)}\n")

    full_window = args.context_length + max(HORIZONS)
    gift_names = ALL_GIFT_SSL_SUBSETS if getattr(args, "include_gift", False) else None
    icml_dir = args.icml_data_dir if getattr(args, "include_icml", False) else None
    corpus_parts = ["LOTSA+local"]
    if gift_names: corpus_parts.append("GIFT")
    if icml_dir:   corpus_parts.append("ICML")
    corpus_label = "+".join(corpus_parts)
    print(f"Building {corpus_label} train loader...")
    train_loader, _ = build_combined_dataloaders(
        context_length=full_window, batch_size=args.batch_size,
        two_views=False, num_workers=args.num_workers,
        pin_memory=True, normalize_per_series=True,
        gift_names=gift_names,
        icml_data_dir=icml_dir,
        seed=args.seed,
    )
    print(f"  {len(train_loader.dataset):,} series in combined corpus")

    print("Training MoMS prompts + heads...")
    t0 = time.time()
    train(mop, train_loader, args, device,
          probes=probes, Z_icml=Z_icml, Y_icml=Y_icml)
    print(f"  Done in {time.time()-t0:.0f}s")

    suffix = getattr(args, "output_suffix", "")
    ckpt_path = args.results_dir / f"mop_zeroshot_{args.encoder_name}{suffix}_checkpoint.pt"
    torch.save({
        "mop_model": mop.state_dict(),
        "encoder_name": args.encoder_name,
        "enc_dim": enc_dim,
        "args": args,
    }, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    print("\nZero-shot evaluation on ICML test splits...")
    test_loaders = build_icml_test_loaders(
        args.icml_data_dir, args.context_length, max(HORIZONS), args.batch_size)
    results = evaluate(mop, test_loaders, args, device)

    csv_path = args.results_dir / f"mop_zeroshot_{args.encoder_name}{suffix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder","stage","dataset","horizon","mse","mae","nrmse"])
        writer.writeheader(); writer.writerows(results)
    print(f"Results → {csv_path}")


if __name__ == "__main__":
    main()
