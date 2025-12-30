"""Utility script to measure basic efficiency metrics for the Mamba encoder.

The script reports:
- Trainable parameter count
- Forward FLOPs estimated via `torch.profiler`
- Average latency (ms) and throughput (samples/sec) over dummy data

Two presets are benchmarked automatically:
- CM-Mamba Tiny  (depth=8)
- CM-Mamba Small (depth=16)

Model hyperparameters, device, and batch size are read from `configs/benchmark_models.yaml`.
Depth is overridden per preset (8 for Tiny, 16 for Small).

Run with: `python tools/measure_mamba_encoder.py`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
try:
    from .benchmark_utils import (
        count_parameters,
        load_yaml,
        measure_flops,
        measure_latency,
        select_device,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from tools.benchmark_utils import (
        count_parameters,
        load_yaml,
        measure_flops,
        measure_latency,
        select_device,
    )

try:
    from ..models.mamba_encoder import MambaEncoder
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.models.mamba_encoder import MambaEncoder


# Configuration loading
BENCH_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_models.yaml"
_CONFIG = load_yaml(BENCH_CONFIG_PATH)
_COMMON = _CONFIG.get("common", {})
_MAMBA_CFG = _CONFIG.get("mamba", {})

# Benchmark presets derived from config (with sane fallbacks)
DEVICE_NAME = str(_COMMON.get("device_name", "auto"))
BATCH_SIZE = int(_COMMON.get("batch_size", 8))
SEQ_LEN = int(_MAMBA_CFG.get("seq_len", 128))
FEATURE_DIM = int(_MAMBA_CFG.get("feature_dim", 384))
MODEL_DIM = int(_MAMBA_CFG.get("model_dim", 768))
EMBEDDING_DIM = int(_MAMBA_CFG.get("embedding_dim", 128))
POOLING = _MAMBA_CFG.get("pooling", "mean")
WARMUP_RUNS = int(_COMMON.get("warmup_runs", 5))
MEASURE_RUNS = int(_COMMON.get("measure_runs", 20))
_VARIANTS_CFG = _MAMBA_CFG.get(
    "variants",
    [
        {"label": "CM-Mamba Tiny", "depth": 8},
        {"label": "CM-Mamba Small", "depth": 16},
    ],
)
VARIANTS: List[Dict[str, int]] = list(_VARIANTS_CFG)
def generate_dummy(batch_size: int, seq_len: int, feature_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, feature_dim, device=device)


def _format_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "N/A"


def print_summary_table(results: List[Dict[str, object]]) -> None:
    headers = ["Model", "Depth", "Params", "FLOPs", "Latency (ms)", "Throughput (samples/s)"]
    rows: List[List[str]] = []

    for res in results:
        rows.append(
            [
                str(res["label"]),
                str(res["depth"]),
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
    log(f"Loaded benchmark config from: {BENCH_CONFIG_PATH}")
    log(
        f"Benchmark settings -> batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, feature_dim={FEATURE_DIM}"
    )

    results: List[Dict[str, object]] = []

    for variant in VARIANTS:
        label = variant["label"]
        depth = int(variant["depth"])

        log(f"\nBenchmarking {label} (depth={depth})")
        model = MambaEncoder(
            input_dim=FEATURE_DIM,
            depth=depth,
            model_dim=MODEL_DIM,
            embedding_dim=EMBEDDING_DIM,
            pooling=POOLING,
        ).to(device)
        model.eval()

        dummy = generate_dummy(BATCH_SIZE, SEQ_LEN, FEATURE_DIM, device)

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
                "depth": depth,
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
        f"(batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, feature_dim={FEATURE_DIM}, device={device}):"
    )
    print_summary_table(results)


if __name__ == "__main__":
    main()
