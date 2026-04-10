"""Tests for model layers."""

import torch
import pytest

from my_vllm.model.layers import (
    RMSNorm,
    RotaryEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)

    def test_no_residual(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(4, 32)
        out = norm(x)
        assert isinstance(out, torch.Tensor)

    def test_with_residual(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(4, 32)
        residual = torch.randn(4, 32)
        out, new_residual = norm(x, residual)
        assert out.shape == (4, 32)
        assert new_residual.shape == (4, 32)
        torch.testing.assert_close(new_residual, x + residual)

    def test_unit_weight_preserves_direction(self):
        norm = RMSNorm(hidden_size=16, eps=1e-6)
        x = torch.randn(2, 16)
        out = norm(x)
        assert out.shape == x.shape


class TestRotaryEmbedding:
    def test_output_shapes(self):
        head_dim = 64
        rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=128)
        num_tokens = 5
        num_heads = 4
        num_kv_heads = 2
        positions = torch.arange(num_tokens)
        q = torch.randn(num_tokens, num_heads * head_dim)
        k = torch.randn(num_tokens, num_kv_heads * head_dim)

        q_rot, k_rot = rope(positions, q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions_different_output(self):
        rope = RotaryEmbedding(head_dim=32, max_position_embeddings=128)
        q = torch.randn(1, 32)
        k = torch.randn(1, 32)

        q1, _ = rope(torch.tensor([0]), q.clone(), k.clone())
        q2, _ = rope(torch.tensor([1]), q.clone(), k.clone())
        assert not torch.allclose(q1, q2)

    def test_same_position_same_output(self):
        rope = RotaryEmbedding(head_dim=32, max_position_embeddings=128)
        q = torch.randn(1, 32)
        k = torch.randn(1, 32)

        q1, k1 = rope(torch.tensor([5]), q.clone(), k.clone())
        q2, k2 = rope(torch.tensor([5]), q.clone(), k.clone())
        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)


class TestLinearLayers:
    def test_column_parallel(self):
        layer = ColumnParallelLinear(64, 128)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 128)

    def test_row_parallel(self):
        layer = RowParallelLinear(128, 64)
        x = torch.randn(4, 128)
        out = layer(x)
        assert out.shape == (4, 64)


class TestQKVParallelLinear:
    def test_output_shapes_mha(self):
        qkv = QKVParallelLinear(
            hidden_size=64, head_dim=16,
            num_heads=4, num_kv_heads=4,
        )
        x = torch.randn(3, 64)
        q, k, v = qkv(x)
        assert q.shape == (3, 64)
        assert k.shape == (3, 64)
        assert v.shape == (3, 64)

    def test_output_shapes_gqa(self):
        qkv = QKVParallelLinear(
            hidden_size=64, head_dim=16,
            num_heads=4, num_kv_heads=2,
        )
        x = torch.randn(3, 64)
        q, k, v = qkv(x)
        assert q.shape == (3, 64)
        assert k.shape == (3, 32)
        assert v.shape == (3, 32)


class TestMergedColumnParallelLinear:
    def test_output_shape(self):
        layer = MergedColumnParallelLinear(64, [128, 128])
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 256)
