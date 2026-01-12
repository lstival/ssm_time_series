"""Utility script to measure parameter count, FLOPs, and latency for common vision backbones.

Benchmarks two torchvision baselines:
- ResNet-50
- ViT-B/32

Metrics reported per model:
- Trainable parameter count
- Forward FLOPs estimated via `torch.profiler`
- Average latency (ms) and throughput (samples/sec) over dummy image batches

Run with: `python tools/measure_vision_models.py`
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchvision import models

from ssm_time_series.tools.benchmark_utils import (
    count_parameters,
    measure_flops,
    measure_latency,
    select_device,
    load_yaml,
)


# Benchmark settings loaded from YAML
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_models.yaml"
_CONFIG = load_yaml(CONFIG_PATH)
_COMMON = _CONFIG.get("common", {})
_VISION_CFG = _CONFIG.get("vision", {})

DEVICE_NAME = str(_COMMON.get("device_name", "auto"))
BATCH_SIZE = int(_COMMON.get("batch_size", 8))
WARMUP_RUNS = int(_COMMON.get("warmup_runs", 5))
MEASURE_RUNS = int(_COMMON.get("measure_runs", 20))
IMAGE_SIZE = int(_VISION_CFG.get("image_size", 224))
CHANNELS = int(_VISION_CFG.get("channels", 3))
_VARIANTS_CFG = _VISION_CFG.get("variants", [])

VISION_BUILDERS: Dict[str, Callable[[], torch.nn.Module]] = {
    "resnet50": lambda: models.resnet50(weights=None),
    "vit_b_32": lambda: models.vit_b_32(weights=None),
}


def _load_variants(configured: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for cfg in configured:
        builder_key = str(cfg.get("builder", "")).lower()
        builder = VISION_BUILDERS.get(builder_key)
        if builder is None:
            print(f"Skipping unknown vision builder '{builder_key}'.")
            continue
        variants.append({"label": cfg.get("label", builder_key), "builder": builder})
    return variants


VARIANTS: List[Dict[str, Any]] = _load_variants(_VARIANTS_CFG)
if not VARIANTS:
    VARIANTS = [
        {"label": "ResNet-50", "builder": VISION_BUILDERS["resnet50"]},
        {"label": "ViT-B/32", "builder": VISION_BUILDERS["vit_b_32"]},
    ]


def generate_dummy(batch_size: int, channels: int, image_size: int, device: torch.device) -> torch.Tensor:
    """Create random image-like input for benchmarking."""
    return torch.randn(batch_size, channels, image_size, image_size, device=device)


def _format_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "N/A"


def print_summary_table(results: List[Dict[str, object]]) -> None:
    headers = ["Model", "Params", "FLOPs", "Latency (ms)", "Throughput (samples/s)"]
    rows: List[List[str]] = []

    for res in results:
        rows.append(
            [
                str(res["label"]),
                f"{res['params']:,}",
                _format_int(res["flops"] if isinstance(res.get("flops"), int) else None),
                f"{res['latency_ms']:.3f}",
                f"{res['throughput']:.2f}",
            ]
        )

    col_widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

    def _format_row(values: List[str]) -> str:
        formatted = []
        for idx, val in enumerate(values):
            if idx == 0:
                formatted.append(val.ljust(col_widths[idx]))
            else:
                formatted.append(val.rjust(col_widths[idx]))
        return " | ".join(formatted)

    print(_format_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(_format_row(row))


def benchmark_models(
    device: Optional[torch.device] = None, *, verbose: bool = True
) -> List[Dict[str, object]]:
    log = print if verbose else (lambda *args, **kwargs: None)
    device = device or select_device(DEVICE_NAME)
    log(f"Using device: {device}")
    log(
        f"Benchmark settings -> batch_size={BATCH_SIZE}, image_size={IMAGE_SIZE}, channels={CHANNELS}"
    )

    results: List[Dict[str, object]] = []

    for variant in VARIANTS:
        label = variant["label"]
        builder: Callable[[], torch.nn.Module] = variant["builder"]

        log(f"\nBenchmarking {label}")
        model = builder().to(device)
        model.eval()

        dummy = generate_dummy(BATCH_SIZE, CHANNELS, IMAGE_SIZE, device)

        param_count = count_parameters(model)
        log(f"Trainable parameters: {param_count:,}")

        flops: Optional[int] = None
        try:
            measured_flops = measure_flops(model, dummy)
            if measured_flops > 0:
                flops = measured_flops
                log(f"Estimated forward FLOPs: {flops:,}")
            else:
                log("Estimated forward FLOPs: unavailable (profiler returned zero)")
        except RuntimeError as exc:
            log(f"Could not measure FLOPs: {exc}")

        latency_ms, throughput = measure_latency(
            model,
            dummy,
            warmup=max(0, WARMUP_RUNS),
            runs=max(1, MEASURE_RUNS),
        )
        log(f"Average latency: {latency_ms:.3f} ms per batch")
        log(f"Throughput: {throughput:.2f} samples/sec")

        results.append(
            {
                "label": label,
                "params": param_count,
                "flops": flops,
                "latency_ms": latency_ms,
                "throughput": throughput,
            }
        )

    return results


def main() -> None:
    device = select_device(DEVICE_NAME)
    results = benchmark_models(device=device, verbose=True)
    print(
        "\nSummary comparison table "
        f"(batch_size={BATCH_SIZE}, image_size={IMAGE_SIZE}, channels={CHANNELS}, device={device}):"
    )
    print_summary_table(results)


if __name__ == "__main__":
    main()
