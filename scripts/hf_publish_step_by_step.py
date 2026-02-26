"""Step-by-step helper to validate and upload CM-Mamba models to Hugging Face."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi

from cm_mamba.hf.forecasting import CM_MambaForecastConfig, CM_MambaForecastModel

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run_pytest(test_paths: list[Path]) -> None:
    args = [sys.executable, "-m", "pytest"] + [str(path) for path in test_paths]
    completed = subprocess.run(args, check=False)
    if completed.returncode != 0:
        raise RuntimeError("pytest reported failures")


def _smoke_test(model_dir: Path) -> None:
    config = CM_MambaForecastConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = CM_MambaForecastModel.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()

    seq_len = max(config.token_size * 2, 8)
    x = torch.randn(2, seq_len, config.input_dim)

    with torch.no_grad():
        preds = model(x)

    expected_horizon = max(config.horizons)
    expected_shape = (2, expected_horizon, config.target_features)
    if preds.shape != expected_shape:
        raise ValueError(f"Unexpected output shape {preds.shape}, expected {expected_shape}")
    if torch.isnan(preds).any():
        raise ValueError("NaNs detected in model outputs")


def _upload_to_hf(
    *,
    model_dir: Path,
    repo_id: str,
    token: Optional[str],
    private: bool,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(repo_id=repo_id, folder_path=str(model_dir), commit_message="Add CM-Mamba model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and upload CM-Mamba models to HF")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local HF model directory")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="HF token (or use HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest before upload")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading to HF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    required_files = ["config.json"]
    weights_files = ["pytorch_model.bin", "model.safetensors"]
    
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if not any((model_dir / w).exists() for w in weights_files):
        missing.append("weights file (pytorch_model.bin or model.safetensors)")
        
    if missing:
        raise FileNotFoundError(f"Missing required files in {model_dir}: {', '.join(missing)}")

    print("Step 1/3: Smoke test")
    _smoke_test(model_dir)
    print("✓ Smoke test passed")

    if args.run_tests:
        print("Step 2/3: Pytest suite")
        _run_pytest([
            ROOT_DIR / "tests" / "test_cm_mamba.py",
            ROOT_DIR / "tests" / "test_hf_forecast.py",
        ])
        print("✓ Pytest passed")
    else:
        print("Step 2/3: Pytest skipped")

    if args.skip_upload:
        print("Step 3/3: Upload skipped")
        return

    print("Step 3/3: Upload to Hugging Face")
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    _upload_to_hf(
        model_dir=model_dir,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
    )
    print(f"✓ Uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
