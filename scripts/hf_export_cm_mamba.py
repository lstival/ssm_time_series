"""Export CM-Mamba checkpoints to Hugging Face format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is in path so we can import without installing
sys.path.append(str(Path(__file__).parents[1] / "src"))

import argparse
from typing import Any, Dict

import yaml

from cm_mamba.hf.forecasting import (
    CM_MambaForecastExportSpec,
    CM_MambaForecastModel,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _apply_overrides(payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    payload = dict(payload)
    paths = dict(payload.get("paths") or {})
    if args.checkpoint is not None:
        paths["checkpoint"] = args.checkpoint
    if args.output_dir is not None:
        paths["output_dir"] = args.output_dir
    if args.model_id is not None:
        paths["model_id"] = args.model_id
    if args.model_card_template is not None:
        paths["model_card_template"] = args.model_card_template
    payload["paths"] = paths
    return payload


def _resolve_paths(payload: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    payload = dict(payload)
    paths = dict(payload.get("paths") or {})
    for key in ("checkpoint", "output_dir", "model_card_template"):
        value = paths.get(key)
        if value is None:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            paths[key] = str((base_dir / path).resolve())
    payload["paths"] = paths
    return payload


def _render_model_card(template_text: str, *, replacements: Dict[str, str]) -> str:
    output = template_text
    for key, value in replacements.items():
        output = output.replace(f"{{{{{key}}}}}", value)
    return output


def export_model(spec: CM_MambaForecastExportSpec) -> Path:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint_path}")
    model = CM_MambaForecastModel.from_checkpoint(
        config=spec.config,
        checkpoint_path=spec.checkpoint_path,
        device="cpu",
        load_encoder=True,
        load_head=True,
    )

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(spec.output_dir)
    spec.config.save_pretrained(spec.output_dir)

    # Copy the necessary source file for the custom model
    import shutil
    import cm_mamba.hf.forecasting as forecasting_module
    
    source_file = Path(forecasting_module.__file__)
    shutil.copy(source_file, spec.output_dir / "forecasting.py")

    if spec.model_card_template is not None and spec.model_card_template.exists():
        template_text = spec.model_card_template.read_text(encoding="utf-8")
        replacements = {
            "MODEL_ID": spec.model_id or spec.output_dir.name,
            "CHECKPOINT_SOURCE": spec.checkpoint_path.parent.name,
            "HORIZONS": ", ".join(str(h) for h in spec.config.horizons),
            "TARGET_FEATURES": str(spec.config.target_features),
        }
        rendered = _render_model_card(template_text, replacements=replacements)
        (spec.output_dir / "README.md").write_text(rendered, encoding="utf-8")

    return spec.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CM-Mamba checkpoints to HF format")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to HF export YAML config (see configs/hf/*.yaml)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--model-id", type=str, default=None, help="Override model_id for the card")
    parser.add_argument(
        "--model-card-template",
        type=Path,
        default=None,
        help="Override model card template path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    payload = _load_yaml(config_path)
    payload = _apply_overrides(payload, args)
    payload = _resolve_paths(payload, config_path.parent)
    spec = CM_MambaForecastExportSpec.from_dict(payload)
    export_path = export_model(spec)
    print(f"Exported model to: {export_path}")


if __name__ == "__main__":
    main()
