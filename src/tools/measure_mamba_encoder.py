"""Utility script to measure basic efficiency metrics for the Mamba encoder.

The script reports:
- Trainable parameter count
- Forward FLOPs estimated via `torch.profiler`
- Average latency (ms) and throughput (samples/sec) over dummy data

Example
-------
python tools/measure_mamba_encoder.py --device auto --batch-size 8 --seq-len 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from torch.profiler import ProfilerActivity, profile
try:
    from ..models.mamba_encoder import MambaEncoder
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.models.mamba_encoder import MambaEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure MambaEncoder FLOPs and latency")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--batch-size", type=int, default=8, help="Dummy batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Dummy sequence length (timesteps)")
    parser.add_argument("--feature-dim", type=int, default=384, help="Feature dimension of dummy input")
    parser.add_argument("--depth", type=int, default=6, help="Number of encoder blocks")
    parser.add_argument("--model-dim", type=int, default=768, help="Hidden dimension inside the encoder")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Output embedding dimension")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Warm-up iterations before measuring latency")
    parser.add_argument("--measure-runs", type=int, default=20, help="Number of timed iterations")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device(name)


def generate_dummy(batch_size: int, seq_len: int, feature_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, feature_dim, device=device)


def measure_flops(model: MambaEncoder, x: torch.Tensor) -> int:
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


def measure_latency(model: MambaEncoder, x: torch.Tensor, *, warmup: int, runs: int) -> Tuple[float, float]:
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
    avg_time = sum(timings) / len(timings)
    throughput = x.size(0) / avg_time if avg_time > 0 else float("inf")
    return avg_time * 1000.0, throughput


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")

    model = MambaEncoder(
        input_dim=args.feature_dim,
        depth=args.depth,
        model_dim=args.model_dim,
        embedding_dim=args.embedding_dim,
        pooling="mean",
    ).to(device)
    model.eval()

    dummy = generate_dummy(args.batch_size, args.seq_len, args.feature_dim, device)

    param_count = model.count_parameters()
    print(f"Trainable parameters: {param_count:,}")

    try:
        flops = measure_flops(model, dummy)
        if flops > 0:
            print(f"Estimated forward FLOPs: {flops:,}")
        else:
            print("Estimated forward FLOPs: unavailable (profiler returned zero)")
    except RuntimeError as exc:
        print(f"Could not measure FLOPs: {exc}")

    latency_ms, throughput = measure_latency(
        model,
        dummy,
        warmup=max(0, args.warmup_runs),
        runs=max(1, args.measure_runs),
    )
    print(f"Average latency: {latency_ms:.3f} ms per batch")
    print(f"Throughput: {throughput:.2f} samples/sec")


if __name__ == "__main__":
    main()
