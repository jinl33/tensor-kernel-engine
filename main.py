"""Unified command-line interface for demos, benchmarks, and visuals."""

from __future__ import annotations

import argparse

import torch

from src.benchmark.profiler import profile_callable
from src.kernels.fused_edge_attention import eager_edge_attention
from src.ops import edge_attention
from src.visualization.plot_performance import generate_panel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tensor Kernel Engine")
    parser.add_argument("--demo", action="store_true", help="run a numerical sanity check")
    parser.add_argument("--benchmark", action="store_true", help="run a latency sweep")
    parser.add_argument("--profile", action="store_true", help="profile one representative workload")
    parser.add_argument("--visualize", action="store_true", help="export the roofline performance panel")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--output", default="assets/kernel_perf_panel.png")
    return parser


def _workload(n: int, dim: int, dtype: torch.dtype):
    q = torch.randn(1, 1, n, dim, dtype=dtype)
    k = torch.randn(1, 1, n, dim, dtype=dtype)
    v = torch.randn(1, 1, n, dim, dtype=dtype)
    edge = torch.rand(1, 1, n, n, dtype=dtype)
    return q, k, v, edge


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if args.demo:
        q, k, v, edge = _workload(16, args.dim, torch.float32)
        result = edge_attention(q, k, v, edge)
        reference = eager_edge_attention(q, k, v, edge)
        print(f"demo: device={q.device} max_error={(result - reference).abs().max().item():.3e}")
    if args.benchmark or args.profile:
        for n in ([512, 1024, 2048, 4096, 8192] if args.benchmark else [1024]):
            q, k, v, edge = _workload(n, args.dim, dtype)
            result = profile_callable(edge_attention, q, k, v, edge, iterations=3, warmup=1)
            print(f"N={n:5d} latency={result.latency_ms:8.3f} ms")
    if args.visualize:
        print(f"wrote {generate_panel(args.output)}")
    if not any((args.demo, args.benchmark, args.profile, args.visualize)):
        _parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
