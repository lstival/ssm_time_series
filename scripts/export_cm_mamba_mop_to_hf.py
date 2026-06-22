"""Export CM-Mamba MoP checkpoint to HuggingFace format.

Usage
-----
python scripts/export_cm_mamba_mop_to_hf.py

Reads:
  results/moms_simclr_enc_F_lasttoken/mop_zeroshot_simclr_enc_F_lasttoken_checkpoint.pt

Writes:
  hf_export/cm_mamba_tiny/model.safetensors

Then push to HuggingFace:
  huggingface-cli upload lstival/cm-mamba-tiny hf_export/cm_mamba_tiny .
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

HF_DIR = ROOT / "hf_export" / "cm_mamba_tiny"
MOP_CKPT = ROOT / "results" / "moms_simclr_enc_F_lasttoken" / "mop_zeroshot_simclr_enc_F_lasttoken_checkpoint.pt"


def main() -> None:
    print(f"Loading MoP checkpoint: {MOP_CKPT}")
    payload = torch.load(MOP_CKPT, map_location="cpu", weights_only=False)
    state_dict: dict = payload["mop_model"]

    # Build model from src (avoids HF relative-import issues in the export script)
    from models.mamba_encoder import MambaEncoder
    from models.mamba_visual_encoder import MambaVisualEncoder
    from models.mop_forecast import MoPForecastModel

    enc_kwargs = dict(
        input_dim=64,
        model_dim=192,
        embedding_dim=96,
        depth=5,
        state_dim=16,
        conv_kernel=4,
        expand_factor=1.5,
        dropout=0.05,
        pooling="last",
    )
    encoder = MambaEncoder(**enc_kwargs)
    visual_encoder = MambaVisualEncoder(**enc_kwargs)
    model = MoPForecastModel(
        encoder=encoder,
        visual_encoder=visual_encoder,
        input_dim=96 * 2,          # concat of temporal + visual embeddings
        hidden_dim=512,
        num_prompts=16,
        horizons=[96, 192, 336, 720],
        target_features=1,
        freeze_encoders=False,     # we'll load all weights
        norm_mode="revin",
        skip_linear=True,
        context_length=336,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("  State dict loaded.")

    # Sanity check: verify all expected keys are present (no GPU forward needed for export)
    expected_prefixes = {"encoder.", "visual_encoder.", "mop.", "heads.", "skip_heads.", "revin."}
    loaded_prefixes = {k.split(".")[0] + "." for k in state_dict}
    assert expected_prefixes == loaded_prefixes, f"Unexpected key prefixes: {loaded_prefixes}"
    n_params = sum(v.numel() for v in model.state_dict().values())
    print(f"  Sanity check passed: {n_params:,} parameters, prefixes {sorted(loaded_prefixes)}")

    # Save to HF dir
    final_state = model.state_dict()
    try:
        # Try safetensors without going through transformers
        import importlib.util
        spec = importlib.util.find_spec("safetensors")
        if spec is not None:
            from safetensors.torch import save_file  # type: ignore
            out_path = HF_DIR / "model.safetensors"
            save_file({k: v.contiguous().float() for k, v in final_state.items()}, str(out_path))
            print(f"  Saved → {out_path}")
        else:
            raise ImportError("safetensors not found")
    except (ImportError, Exception) as e:
        print(f"  safetensors unavailable ({e}), saving PyTorch bin")
        out_path = HF_DIR / "pytorch_model.bin"
        torch.save(final_state, str(out_path))
        print(f"  Saved → {out_path}")

    print("\nDone. To push to HuggingFace:")
    print(f"  huggingface-cli upload lstival/cm-mamba-tiny {HF_DIR} .")


if __name__ == "__main__":
    main()
