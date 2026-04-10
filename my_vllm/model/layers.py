"""Common model layers: RMSNorm, RotaryEmbedding, linear wrappers."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by Qwen/LLaMA family)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
            residual = x

        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        output = self.weight * x.to(self.weight.dtype)

        if residual is not None:
            return output, residual
        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Pre-computes cos/sin cache and applies rotation to Q and K tensors.
    Uses NeoX-style interleaving (first half / second half split).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=dtype)
        freqs = torch.outer(t, inv_freq)
        cos_cache = freqs.cos()
        sin_cache = freqs.sin()
        cache = torch.cat([cos_cache, sin_cache], dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        key_shape = key.shape
        num_tokens = positions.shape[0]

        q = query.view(num_tokens, -1, self.head_dim)
        k = key.view(num_tokens, -1, self.head_dim)

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        return q.reshape(query_shape), k.reshape(key_shape)

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply NeoX-style rotary embedding: split into two halves."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        # cos, sin: (num_tokens, half) -> (num_tokens, 1, half) for broadcast
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([o1, o2], dim=-1)


class ColumnParallelLinear(nn.Module):
    """Linear layer that is column-parallel on a single GPU (regular Linear)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Linear layer that is row-parallel on a single GPU (regular Linear)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QKVParallelLinear(nn.Module):
    """Merged QKV projection supporting GQA.

    Fuses Q, K, V projections into a single linear layer.
    Output can be split into Q (num_heads * head_dim),
    K (num_kv_heads * head_dim), V (num_kv_heads * head_dim).
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.output_size = self.q_size + 2 * self.kv_size

        self.linear = nn.Linear(hidden_size, self.output_size, bias=bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.linear(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v


class MergedColumnParallelLinear(nn.Module):
    """Merged gate+up projection for SwiGLU MLP."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        self.linear = nn.Linear(input_size, total, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
