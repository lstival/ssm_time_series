"""Benchmark Chronos and TimesFM models for parameter count, FLOPs, and latency.

This script mirrors the metrics from tools/measure_vision_models.py and reports:
- Trainable parameter count
- Forward FLOPs (via torch.profiler)
- Average latency (ms) and throughput (samples/sec) on dummy data

Models benchmarked:
- Chronos-T5-Small (ChronosPipeline)
- TimesFM-2.0-500M (TimesFmModelForPrediction)

Run with: `python tools/measure_time_series_models.py`
Settings are shared via `configs/benchmark_models.yaml`.
Note: First run will download models from Hugging Face. Ensure you have internet
access or cached weights.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from chronos import ChronosPipeline
from transformers import TimesFmModelForPrediction

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


# Benchmark settings loaded from YAML
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_models.yaml"
_CONFIG = load_yaml(CONFIG_PATH)
_COMMON = _CONFIG.get("common", {})
_TS_CFG = _CONFIG.get("time_series", {})

DEVICE_NAME = str(_COMMON.get("device_name", "auto"))
BATCH_SIZE = int(_COMMON.get("batch_size", 4))
CONTEXT_LENGTH = int(_TS_CFG.get("context_length", 256))
WARMUP_RUNS = int(_COMMON.get("warmup_runs", 5))
MEASURE_RUNS = int(_COMMON.get("measure_runs", 20))
_VARIANTS_CFG = _TS_CFG.get("variants", [])

DEFAULT_TS_VARIANTS: List[Dict[str, str]] = [
    {"label": "Chronos-T5-Small", "kind": "chronos", "model_id": "amazon/chronos-t5-small"},
    {"label": "TimesFM-2.0-500M", "kind": "timesfm", "model_id": "google/timesfm-2.0-500m-pytorch"},
]


class ChronosWrapper(nn.Module):
    """Thin wrapper to expose ChronosPipeline as an nn.Module."""

    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id, device_map=self.device, dtype=self.dtype
        )
        # Register underlying model parameters if available
        self.model = getattr(self.pipeline, "model", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chronos pipeline expects CPU float inputs for embed; align then return to device
        x_cpu = x.detach().cpu().float()
        emb, _ = self.pipeline.embed(x_cpu)
        return emb.to(device=self.device, dtype=self.dtype)


class TimesFmWrapper(nn.Module):
    """Wrapper around TimesFmModelForPrediction returning last hidden state."""

    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        # TimesFM defaults to float32 even on CUDA for stability
        self.model = TimesFmModelForPrediction.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(past_values=x, return_dict=True)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        # Fallback for dict-like outputs
        return out["last_hidden_state"]



def _build_variants(configured: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    source = configured if configured else DEFAULT_TS_VARIANTS
    for cfg in source:
        label = cfg.get("label") or cfg.get("model_id") or "time-series-model"
        kind = str(cfg.get("kind", "")).lower()
        model_id = cfg.get("model_id")
        if not model_id:
            print(f"Skipping {label}: model_id missing.")
            continue
        if kind == "chronos":
            builder = lambda device, mid=model_id: ChronosWrapper(mid, device)
        elif kind in {"timesfm", "times_fm"}:
            builder = lambda device, mid=model_id: TimesFmWrapper(mid, device)
        else:
            print(f"Skipping {label}: unknown model kind '{kind}'.")
            continue
        variants.append({"label": label, "builder": builder, "kind": kind})
    return variants


VARIANTS: List[Dict[str, Any]] = _build_variants(_VARIANTS_CFG)
def generate_dummy(batch_size: int, context_length: int, device: torch.device) -> torch.Tensor:
    """Create random univariate time series input."""
    return torch.randn(batch_size, context_length, device=device, dtype=torch.float32)




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
        f"Benchmark settings -> batch_size={BATCH_SIZE}, context_length={CONTEXT_LENGTH}"
    )

    results: List[Dict[str, object]] = []

    for variant in VARIANTS:
        label = variant["label"]
        builder: Callable[[torch.device], nn.Module] = variant["builder"]

        log(f"\nBenchmarking {label}")
        model = builder(device)
        model.to(device)
        model.eval()

        dummy = generate_dummy(BATCH_SIZE, CONTEXT_LENGTH, device)

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
        f"(batch_size={BATCH_SIZE}, context_length={CONTEXT_LENGTH}, device={device}):"
    )
    print_summary_table(results)


if __name__ == "__main__":
    main()
