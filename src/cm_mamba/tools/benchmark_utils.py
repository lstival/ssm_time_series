"""Shared utilities for benchmarking models across domains."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.profiler import ProfilerActivity, profile
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file with safe defaults."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}
    except FileNotFoundError:
        print(f"Config file not found at {path}; using defaults.")
    except yaml.YAMLError as exc:
        print(f"Could not parse YAML at {path}: {exc}; using defaults.")
    return {}


def select_device(name: str) -> torch.device:
    """Choose an available device, preferring CUDA when requested or available."""
    name = name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device(name)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: torch.nn.Module, x: torch.Tensor) -> int:
    """Estimate forward FLOPs using torch.profiler."""
    activities = [ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
        with torch.no_grad():
            _ = model(x)

    total_flops = 0
    for evt in prof.key_averages():
        if evt.flops is not None:
            total_flops += evt.flops
    return int(total_flops)


def measure_latency(
    model: torch.nn.Module, x: torch.Tensor, *, warmup: int, runs: int
) -> Tuple[float, float]:
    """Measure average latency (ms) and throughput (samples/sec)."""
    timings = []
    with torch.no_grad():
        for i in range(warmup + runs):
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if i >= warmup:
                timings.append(elapsed)
    avg_time = sum(timings) / len(timings) if timings else 0.0
    throughput = x.size(0) / avg_time if avg_time > 0 else float("inf")
    return avg_time * 1000.0, throughput


# Local import guard for scripts executed without the package on sys.path
if __name__ == "__main__":
    print("This module provides helpers and is not intended to be run directly.")
