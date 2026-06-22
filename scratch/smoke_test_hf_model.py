"""Smoke test: load CM-Mamba MoP model and run zero-shot on ICML datasets.

Two modes:
  --source local  Build from src + load hf_export/model.safetensors (no internet).
  --source hf     AutoModel.from_pretrained("lstival/cm-mamba-tiny") (after push).

Usage
-----
# Before push (local weights):
python scratch/smoke_test_hf_model.py --source local

# After push to HuggingFace:
python scratch/smoke_test_hf_model.py --source hf

Expected overall NRMSE ≈ 0.4743 on TSLib-7 (matches moms_simclr_enc_F_lasttoken run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

HF_DIR   = ROOT / "hf_export" / "cm_mamba_tiny"
MOP_CKPT = ROOT / "results" / "moms_simclr_enc_F_lasttoken" / "mop_zeroshot_simclr_enc_F_lasttoken_checkpoint.pt"

ICML_DIR  = ROOT / "ICML_datasets"
DATASETS  = ["ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
             "weather.csv", "traffic.csv", "electricity.csv"]
HORIZONS  = [96, 192, 336, 720]
CONTEXT   = 336


def resolve_ds_dir(ds_name: str) -> str:
    for cand in ICML_DIR.rglob(ds_name):
        return str(cand.parent)
    return str(ICML_DIR)


# ── local wrapper so evaluation code can call model(x, horizon=H) ──────────

class LocalMoPWrapper(torch.nn.Module):
    """Thin wrapper around MoPForecastModel exposing (B,T,C)→(B,H,C) interface."""

    def __init__(self, mop_model):
        super().__init__()
        self.model = mop_model

    def forward(self, x: torch.Tensor, horizon: int = 96) -> torch.Tensor:
        """x: (B, T, C)  →  (B, H, C)"""
        B, T, C = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * C, 1, T)   # (B*C, 1, T)
        pred = self.model.greedy_predict(x_ci, horizon, CONTEXT)  # (B*C, H, 1)
        return pred.squeeze(-1).reshape(B, C, horizon).permute(0, 2, 1)  # (B, H, C)


def build_local_model(device: torch.device):
    from models.mamba_encoder import MambaEncoder
    from models.mamba_visual_encoder import MambaVisualEncoder
    from models.mop_forecast import MoPForecastModel

    enc_kwargs = dict(
        input_dim=64, model_dim=192, embedding_dim=96, depth=5,
        state_dim=16, conv_kernel=4, expand_factor=1.5, dropout=0.05, pooling="last",
    )
    encoder = MambaEncoder(**enc_kwargs)
    visual_encoder = MambaVisualEncoder(**enc_kwargs)
    mop = MoPForecastModel(
        encoder=encoder, visual_encoder=visual_encoder,
        input_dim=96 * 2, hidden_dim=512, num_prompts=16,
        horizons=HORIZONS, target_features=1,
        freeze_encoders=False, norm_mode="revin",
        skip_linear=True, context_length=CONTEXT,
    )

    # Try safetensors first (exported), else raw checkpoint
    sf = HF_DIR / "model.safetensors"
    if sf.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sf))
        mop.load_state_dict(sd, strict=True)
        print(f"  Loaded weights from {sf}")
    else:
        payload = torch.load(str(MOP_CKPT), map_location="cpu", weights_only=False)
        mop.load_state_dict(payload["mop_model"], strict=False)
        print(f"  Loaded weights from {MOP_CKPT}")

    mop.to(device).eval()
    return LocalMoPWrapper(mop)


def build_hf_model():
    from transformers import AutoModel
    print("Downloading from HuggingFace: lstival/cm-mamba-tiny ...")
    return AutoModel.from_pretrained("lstival/cm-mamba-tiny", trust_remote_code=True)


def evaluate(model, device: torch.device) -> list[dict]:
    from time_series_loader import TimeSeriesDataModule

    model = model.to(device).eval()
    results = []

    for ds in DATASETS:
        tag = ds.replace(".csv", "")
        try:
            dm = TimeSeriesDataModule(
                dataset_name=ds, data_dir=resolve_ds_dir(ds),
                batch_size=64, val_batch_size=64, num_workers=0,
                pin_memory=False, normalize=True,
                train=False, val=False, test=True,
                sample_size=(CONTEXT, 0, max(HORIZONS)),
                scaler_type="standard", features="M",
            )
            dm.setup()
            loader = dm.test_loaders[0] if dm.test_loaders else None
            if loader is None:
                print(f"  [skip] {tag}: no test loader")
                continue
        except Exception as e:
            print(f"  [skip] {tag}: {e}")
            continue

        for H in HORIZONS:
            mse_list, mae_list = [], []
            with torch.no_grad():
                for batch in loader:
                    x, y = batch[0].to(device).float(), batch[1].to(device).float()
                    if y.shape[1] < H:
                        continue
                    pred = model(x, horizon=H)        # (B, H, C)
                    y_h = y[:, :H, :]
                    mse_list.append(F.mse_loss(pred, y_h).item())
                    mae_list.append(torch.mean(torch.abs(pred - y_h)).item())

            if not mse_list:
                continue
            mse  = sum(mse_list) / len(mse_list)
            mae  = sum(mae_list) / len(mae_list)
            nrmse = mse ** 0.5
            print(f"  {tag:12s} H={H:3d}: NRMSE={nrmse:.4f}  MAE={mae:.4f}")
            results.append({"dataset": tag, "horizon": H, "nrmse": nrmse, "mse": mse, "mae": mae})

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--device", default=None,
                        help="Device to run on (default: cuda if available, else cpu). "
                             "NOTE: --source local requires CUDA because mamba_ssm kernel "
                             "is GPU-only. Use --source hf for CPU inference via the "
                             "pure-PyTorch fallback in the HuggingFace package.")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    if args.source == "local" and device.type == "cpu":
        print("[WARNING] --source local uses mamba_ssm CUDA kernels.")
        print("          If you get a CUDA error, run on a GPU node or use --source hf.")
        print()

    if args.source == "local":
        model = build_local_model(device)
    else:
        model = build_hf_model()

    print(f"\nEvaluating on {len(DATASETS)} ICML datasets, H={HORIZONS} ...")
    results = evaluate(model, device)

    if results:
        overall = sum(r["nrmse"] for r in results) / len(results)
        print(f"\nOverall NRMSE = {overall:.4f}  (reference: 0.4743)")
        if abs(overall - 0.4743) < 0.005:
            print("  ✓ Matches reference — weights loaded correctly.")
        else:
            print(f"  ✗ Deviation from reference: {overall - 0.4743:+.4f}")
    else:
        print("No results — check ICML_datasets path.")


if __name__ == "__main__":
    main()
