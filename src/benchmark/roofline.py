"""Operational-intensity and roofline calculations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    memory_bandwidth_gbps: float
    peak_tflops: float


def operational_intensity(flops: float, bytes_transferred: float) -> float:
    return flops / max(bytes_transferred, 1.0)


def roofline_tflops(intensity: float, hardware: Hardware) -> float:
    bandwidth_ceiling = intensity * hardware.memory_bandwidth_gbps / 1000.0
    return min(bandwidth_ceiling, hardware.peak_tflops)


def roofline_curve(hardware: Hardware, points: int = 100):
    import numpy as np

    intensity = np.logspace(-2, 3, points)
    return intensity, np.minimum(intensity * hardware.memory_bandwidth_gbps / 1000.0, hardware.peak_tflops)
