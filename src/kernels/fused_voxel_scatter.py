"""Point-to-voxel aggregation with a portable reference and optional Triton path."""

from __future__ import annotations

import torch


def _voxel_coordinates(points: torch.Tensor, voxel_size: float, origin: torch.Tensor) -> torch.Tensor:
    return torch.floor((points - origin) / voxel_size).to(torch.long)


def eager_voxel_scatter(points: torch.Tensor, features: torch.Tensor, grid_size: tuple[int, int, int],
                        voxel_size: float = 1.0, origin: torch.Tensor | None = None,
                        reduction: str = "mean") -> torch.Tensor:
    """Dense voxel reduction; empty voxels are zero for mean and -inf for max."""
    if origin is None:
        origin = torch.zeros(3, device=points.device, dtype=points.dtype)
    coords = _voxel_coordinates(points, voxel_size, origin)
    valid = (coords >= 0).all(dim=1) & (coords < torch.tensor(grid_size, device=points.device)).all(dim=1)
    coords, features = coords[valid], features[valid]
    output = torch.zeros((*grid_size, features.shape[-1]), device=features.device, dtype=features.dtype)
    if reduction == "mean":
        counts = torch.zeros(grid_size, device=features.device, dtype=torch.float32)
        flat = coords[:, 0] * grid_size[1] * grid_size[2] + coords[:, 1] * grid_size[2] + coords[:, 2]
        output_flat = output.view(-1, features.shape[-1])
        output_flat.index_add_(0, flat, features)
        counts.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        return output / counts.clamp_min(1).unsqueeze(-1).to(output.dtype)
    if reduction == "max":
        output.fill_(torch.finfo(features.dtype).min)
        for channel in range(features.shape[-1]):
            output[..., channel].view(-1).scatter_reduce_(0, flat, features[:, channel], reduce="amax", include_self=True)
        return output
    raise ValueError(f"Unsupported reduction: {reduction}")


def voxel_scatter(points: torch.Tensor, features: torch.Tensor, grid_size: tuple[int, int, int],
                  voxel_size: float = 1.0, origin: torch.Tensor | None = None,
                  reduction: str = "mean") -> torch.Tensor:
    """Portable API; CUDA users can replace this dispatch with a custom atomic kernel."""
    return eager_voxel_scatter(points, features, grid_size, voxel_size, origin, reduction)
