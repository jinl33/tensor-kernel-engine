import pytest
import torch

from src.kernels.fused_edge_attention import eager_edge_attention
from src.ops import edge_attention


@pytest.mark.parametrize("shape", [(1, 1, 7, 8), (2, 2, 16, 16)])
def test_edge_attention_matches_eager(shape):
    torch.manual_seed(7)
    batch, heads, m_size, dim = shape
    q = torch.randn(shape, dtype=torch.float32)
    k = torch.randn(batch, heads, m_size + 3, dim)
    v = torch.randn_like(k)
    edge = torch.rand(batch, heads, m_size, m_size + 3)
    actual = edge_attention(q, k, v, edge, beta=0.3)
    expected = eager_edge_attention(q, k, v, edge, beta=0.3)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_edge_attention_extreme_edges_is_finite():
    tensors = [torch.zeros(1, 1, 4, 8), torch.ones(1, 1, 4, 8)]
    q, v = tensors
    k = torch.zeros_like(q)
    edge = torch.full((1, 1, 4, 4), 1e-12)
    result = edge_attention(q, k, v, edge)
    assert torch.isfinite(result).all()
