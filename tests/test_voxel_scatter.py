import torch

from src.kernels.fused_voxel_scatter import eager_voxel_scatter
from src.ops import voxel_scatter


def test_mean_voxel_scatter():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.1, 0.4], [1.1, 0.1, 0.1]])
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]])
    actual = voxel_scatter(points, features, (2, 2, 2), reduction="mean")
    expected = eager_voxel_scatter(points, features, (2, 2, 2), reduction="mean")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0, 0, 0], torch.tensor([2.0, 3.0]))


def test_out_of_bounds_points_are_ignored():
    points = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    features = torch.ones(2, 1)
    result = voxel_scatter(points, features, (1, 1, 1), reduction="mean")
    torch.testing.assert_close(result[0, 0, 0], torch.ones(1))
