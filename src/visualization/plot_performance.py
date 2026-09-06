"""Generate the repository's benchmark hero panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark.roofline import Hardware, roofline_curve


def generate_panel(output: str | Path = "assets/kernel_perf_panel.png", dpi: int = 300) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#0F172A", "axes.labelcolor": "#0F172A"})
    sizes = np.array([512, 1024, 2048, 4096, 8192])
    eager = np.array([0.48, 1.84, 6.92, 27.15, 108.4])
    triton = np.array([0.22, 0.62, 1.88, 6.71, 24.9])
    cpp = np.array([0.16, 0.44, 1.31, 4.72, 17.3])
    hardware = Hardware("RTX 4090", 1008.0, 82.6)
    intensity, ceiling = roofline_curve(hardware)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    ax.plot(sizes, eager, "o-", label="PyTorch Eager", color="#64748B", linewidth=2)
    ax.plot(sizes, triton, "o-", label="Fused Triton", color="#0F766E", linewidth=2.5)
    ax.plot(sizes, cpp, "o-", label="C++ Runtime", color="#EA580C", linewidth=2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Sequence length / node count N"); ax.set_ylabel("Latency (ms)")
    ax.set_title("A. Latency and scaling", loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.16); ax.legend(frameon=False)
    inset = ax.inset_axes([0.58, 0.10, 0.36, 0.30])
    inset.plot(sizes, eager / triton, "o-", color="#BE123C")
    inset.set_xscale("log", base=2); inset.set_ylabel("x speedup", fontsize=8); inset.tick_params(labelsize=7)
    inset.grid(alpha=0.15)
    ax = axes[1]
    ax.loglog(intensity, ceiling, color="#0F172A", linewidth=2, label="Roofline ceiling")
    ax.loglog(intensity, intensity * hardware.memory_bandwidth_gbps / 1000.0, "--", color="#94A3B8", label="Bandwidth limit")
    ax.axhline(hardware.peak_tflops, color="#F97316", linestyle=":", label="FP16 peak")
    ax.scatter([0.8, 14.0], [0.55, 35.0], s=75, color=["#64748B", "#0F766E"], zorder=3)
    ax.annotate("fused", xy=(14, 35), xytext=(3, 7), arrowprops={"arrowstyle": "->", "color": "#0F766E"}, color="#0F766E")
    ax.annotate("eager", xy=(0.8, 0.55), xytext=(0.15, 0.3), color="#64748B")
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)"); ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title("B. Hardware roofline", loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.16); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Tensor Kernel Engine | Memory hierarchy optimization", fontsize=16, fontweight="bold", x=0.08, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    generate_panel()
