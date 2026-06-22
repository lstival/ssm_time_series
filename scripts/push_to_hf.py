"""Push hf_export/cm_mamba_tiny to HuggingFace.

Usage:
    /lustre/nobackup/WUR/AIN/stiva001/kanga/bin/python3 scripts/push_to_hf.py --token hf_xxx
    /lustre/nobackup/WUR/AIN/stiva001/kanga/bin/python3 scripts/push_to_hf.py  # reads HF_TOKEN env var
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / "hf_export" / "cm_mamba_tiny"
REPO_ID = "lstival/cm-mamba-tiny"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None,
                        help="HuggingFace write token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Provide --token hf_xxx or export HF_TOKEN=hf_xxx")

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"Uploading {HF_DIR} → {REPO_ID} ...")
    url = api.upload_folder(
        folder_path=str(HF_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update to MoP model (GIFT NRMSE=0.4075, TSLib NRMSE=0.4743)",
    )
    print(f"Done: {url}")


if __name__ == "__main__":
    main()
