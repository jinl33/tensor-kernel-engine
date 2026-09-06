"""Fused edge-affinity attention with an optional Triton implementation."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the local GPU environment
    triton = None
    tl = None


def eager_edge_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         edge: torch.Tensor, beta: float = 1.0,
                         eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation for [B, H, M, D] tensors."""
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    logits = logits + beta * torch.log(edge.float().clamp_min(eps))
    return torch.softmax(logits, dim=-1).matmul(v.float()).to(v.dtype)


if triton is not None:

    @triton.jit
    def _edge_attention_kernel(
        q_ptr, k_ptr, v_ptr, e_ptr, o_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_eb, stride_eh, stride_em, stride_en,
        stride_ob, stride_oh, stride_om, stride_od,
        m_size, n_size, d_size, n_heads, beta, eps,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // n_heads
        head = pid_bh % n_heads
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < m_size
        q = tl.load(q_ptr + batch * stride_qb + head * stride_qh + rows[:, None] * stride_qm + dims[None, :] * stride_qd,
                    mask=row_mask[:, None] & (dims[None, :] < d_size), other=0.0)
        running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
        running_norm = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        scale = 1.0 / tl.sqrt(tl.full([], d_size, tl.float32))
        for start in range(0, n_size, BLOCK_N):
            keys = start + cols
            key_mask = keys < n_size
            k = tl.load(k_ptr + batch * stride_kb + head * stride_kh + keys[:, None] * stride_kn + dims[None, :] * stride_kd,
                        mask=key_mask[:, None] & (dims[None, :] < d_size), other=0.0)
            v = tl.load(v_ptr + batch * stride_vb + head * stride_vh + keys[:, None] * stride_vn + dims[None, :] * stride_vd,
                        mask=key_mask[:, None] & (dims[None, :] < d_size), other=0.0)
            edge = tl.load(e_ptr + batch * stride_eb + head * stride_eh + rows[:, None] * stride_em + keys[None, :] * stride_en,
                           mask=row_mask[:, None] & key_mask[None, :], other=eps)
            scores = tl.dot(q, tl.trans(k)) * scale + beta * tl.log(tl.maximum(edge, eps))
            scores = tl.where(row_mask[:, None] & key_mask[None, :], scores, -float("inf"))
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            alpha = tl.exp(running_max - new_max)
            probs = tl.exp(scores - new_max[:, None])
            running_norm = running_norm * alpha + tl.sum(probs, axis=1)
            acc = acc * alpha[:, None] + tl.dot(probs.to(q.dtype), v)
            running_max = new_max
        out = acc / running_norm[:, None]
        tl.store(o_ptr + batch * stride_ob + head * stride_oh + rows[:, None] * stride_om + dims[None, :] * stride_od,
                 out, mask=row_mask[:, None] & (dims[None, :] < d_size))


def triton_edge_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          edge: torch.Tensor, beta: float = 1.0,
                          eps: float = 1e-6, block_m: int = 64,
                          block_n: int = 64, num_warps: int = 4) -> torch.Tensor:
    """Run the fused online-softmax kernel. Inputs must be CUDA and contiguous."""
    if triton is None or not q.is_cuda:
        return eager_edge_attention(q, k, v, edge, beta, eps)
    batch, heads, m_size, d_size = q.shape
    n_size = k.shape[-2]
    output = torch.empty_like(v.new_empty((batch, heads, m_size, d_size)))
    grid = (triton.cdiv(m_size, block_m), batch * heads)
    _edge_attention_kernel[grid](
        q, k, v, edge, output, *q.stride(), *k.stride(), *v.stride(), *edge.stride(), *output.stride(),
        m_size, n_size, d_size, heads, beta, eps, BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(d_size), num_warps=num_warps,
    )
    return output


def edge_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   edge: torch.Tensor, beta: float = 1.0, eps: float = 1e-6,
                   block_m: int = 64, block_n: int = 64, num_warps: int = 4) -> torch.Tensor:
    """Dispatch to Triton for inference CUDA tensors, otherwise use the reference."""
    can_use_triton = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    if can_use_triton and not torch.is_grad_enabled():
        return triton_edge_attention(q.contiguous(), k.contiguous(), v.contiguous(), edge.contiguous(),
                                     beta, eps, block_m, block_n, num_warps)
    return eager_edge_attention(q, k, v, edge, beta, eps)
