"""CUDA-event profiler with a portable wall-clock fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProfileResult:
    latency_ms: float
    bandwidth_gbps: float
    tflops: float
    iterations: int


def profile_callable(fn, *args, bytes_transferred: int = 0, flops: int = 0,
                     warmup: int = 20, iterations: int = 100, **kwargs) -> ProfileResult:
    """Measure a callable using CUDA events when available, otherwise perf_counter."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn(*args, **kwargs)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end) / iterations
    else:
        start_time = time.perf_counter()
        for _ in range(iterations):
            fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / iterations
    seconds = elapsed_ms / 1000.0
    return ProfileResult(
        latency_ms=elapsed_ms,
        bandwidth_gbps=bytes_transferred / max(seconds * 1e9, 1e-12),
        tflops=flops / max(seconds * 1e12, 1e-12),
        iterations=iterations,
    )
