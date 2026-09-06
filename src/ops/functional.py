"""Autograd-friendly public wrappers for fused operations."""

from __future__ import annotations

import torch

from ..kernels.fused_edge_attention import edge_attention as _edge_attention
from ..kernels.fused_voxel_scatter import voxel_scatter as _voxel_scatter


class EdgeAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, edge, beta=1.0, eps=1e-6):
        return _edge_attention(q, k, v, edge, beta, eps)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("Use the eager path when gradients are required")


def edge_attention(q, k, v, edge, beta=1.0, eps=1e-6):
    if torch.is_grad_enabled() or not q.is_cuda:
        return _edge_attention(q, k, v, edge, beta, eps)
    return EdgeAttentionFunction.apply(q, k, v, edge, beta, eps)


def voxel_scatter(points, features, grid_size, voxel_size=1.0, origin=None, reduction="mean"):
    return _voxel_scatter(points, features, grid_size, voxel_size, origin, reduction)
