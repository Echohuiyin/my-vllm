"""Tests for PagedAttention."""

import torch
import pytest

from my_vllm.attention.paged_attention import PagedAttention


def _make_kv_cache(
    num_blocks: int = 8,
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.zeros(2, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)


class TestWriteToCache:
    def test_write_single_token(self):
        pa = PagedAttention(num_heads=4, head_dim=16, num_kv_heads=2)
        kv_cache = _make_kv_cache()
        key = torch.randn(1, 32)
        value = torch.randn(1, 32)
        slot_mapping = torch.tensor([0])

        pa._write_to_cache(key, value, kv_cache, slot_mapping)
        assert kv_cache[0, 0, 0].abs().sum() > 0
        assert kv_cache[1, 0, 0].abs().sum() > 0

    def test_write_multiple_tokens(self):
        pa = PagedAttention(num_heads=4, head_dim=16, num_kv_heads=2)
        kv_cache = _make_kv_cache()
        key = torch.randn(3, 32)
        value = torch.randn(3, 32)
        slot_mapping = torch.tensor([0, 1, 4])

        pa._write_to_cache(key, value, kv_cache, slot_mapping)
        assert kv_cache[0, 0, 0].abs().sum() > 0  # slot 0 -> block 0, offset 0
        assert kv_cache[0, 0, 1].abs().sum() > 0  # slot 1 -> block 0, offset 1
        assert kv_cache[0, 1, 0].abs().sum() > 0  # slot 4 -> block 1, offset 0


class TestPrefillAttention:
    def test_output_shape(self):
        num_heads, head_dim, num_kv_heads = 4, 16, 2
        pa = PagedAttention(num_heads, head_dim, num_kv_heads)
        seq_len = 5
        q = torch.randn(seq_len, num_heads * head_dim)
        k = torch.randn(seq_len, num_kv_heads * head_dim)
        v = torch.randn(seq_len, num_kv_heads * head_dim)
        seq_lens = torch.tensor([seq_len])

        out = pa._prefill_attention(q, k, v, seq_lens)
        assert out.shape == (seq_len, num_heads * head_dim)

    def test_multiple_sequences(self):
        num_heads, head_dim, num_kv_heads = 2, 8, 2
        pa = PagedAttention(num_heads, head_dim, num_kv_heads)
        s1, s2 = 3, 5
        total = s1 + s2
        q = torch.randn(total, num_heads * head_dim)
        k = torch.randn(total, num_kv_heads * head_dim)
        v = torch.randn(total, num_kv_heads * head_dim)
        seq_lens = torch.tensor([s1, s2])

        out = pa._prefill_attention(q, k, v, seq_lens)
        assert out.shape == (total, num_heads * head_dim)

    def test_causal_mask(self):
        num_heads, head_dim = 1, 8
        pa = PagedAttention(num_heads, head_dim, num_heads)
        seq_len = 4
        q = torch.randn(seq_len, head_dim)
        k = torch.randn(seq_len, head_dim)
        v = torch.ones(seq_len, head_dim)
        v[3] = 100.0  # future token has very different value
        seq_lens = torch.tensor([seq_len])

        out = pa._prefill_attention(q, k, v, seq_lens)
        # First position should not see the future value
        assert out[0].abs().max() < 10.0


class TestDecodeAttention:
    def test_output_shape(self):
        num_heads, head_dim, num_kv_heads = 4, 16, 2
        block_size = 4
        pa = PagedAttention(num_heads, head_dim, num_kv_heads)
        kv_cache = _make_kv_cache(num_blocks=8, block_size=block_size,
                                  num_kv_heads=num_kv_heads, head_dim=head_dim)

        # Populate cache with some data
        for s in range(5):
            kv_cache[0, s // block_size, s % block_size] = torch.randn(num_kv_heads, head_dim)
            kv_cache[1, s // block_size, s % block_size] = torch.randn(num_kv_heads, head_dim)

        batch_size = 1
        q = torch.randn(batch_size, num_heads * head_dim)
        block_tables = torch.tensor([[0, 1, 0, 0]])
        seq_lens = torch.tensor([5])

        out = pa._decode_attention(q, kv_cache, block_tables, seq_lens)
        assert out.shape == (1, num_heads * head_dim)

    def test_batch_decode(self):
        num_heads, head_dim, num_kv_heads = 2, 8, 2
        block_size = 4
        pa = PagedAttention(num_heads, head_dim, num_kv_heads)
        kv_cache = _make_kv_cache(num_blocks=8, block_size=block_size,
                                  num_kv_heads=num_kv_heads, head_dim=head_dim)

        for b in range(8):
            for s in range(block_size):
                kv_cache[0, b, s] = torch.randn(num_kv_heads, head_dim)
                kv_cache[1, b, s] = torch.randn(num_kv_heads, head_dim)

        batch_size = 2
        q = torch.randn(batch_size, num_heads * head_dim)
        block_tables = torch.tensor([[0, 1], [2, 3]])
        seq_lens = torch.tensor([5, 7])

        out = pa._decode_attention(q, kv_cache, block_tables, seq_lens)
        assert out.shape == (2, num_heads * head_dim)


class TestPagedAttentionForward:
    def test_prefill_forward(self):
        num_heads, head_dim, num_kv_heads = 2, 16, 2
        block_size = 4
        pa = PagedAttention(num_heads, head_dim, num_kv_heads)
        kv_cache = _make_kv_cache(num_blocks=8, block_size=block_size,
                                  num_kv_heads=num_kv_heads, head_dim=head_dim)

        seq_len = 5
        q = torch.randn(seq_len, num_heads * head_dim)
        k = torch.randn(seq_len, num_kv_heads * head_dim)
        v = torch.randn(seq_len, num_kv_heads * head_dim)
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.tensor([0, 1, 2, 3, 4])
        seq_lens = torch.tensor([seq_len])

        out = pa(q, k, v, kv_cache, block_tables, slot_mapping, seq_lens, is_prefill=True)
        assert out.shape == (seq_len, num_heads * head_dim)
        # Verify KV cache was written
        assert kv_cache[0, 0, 0].abs().sum() > 0

    def test_decode_forward(self):
        num_heads, head_dim, num_kv_heads = 2, 16, 2
        block_size = 4
        pa = PagedAttention(num_heads, head_dim, num_kv_heads)
        kv_cache = _make_kv_cache(num_blocks=8, block_size=block_size,
                                  num_kv_heads=num_kv_heads, head_dim=head_dim)

        # Pre-populate cache
        for s in range(4):
            kv_cache[0, 0, s] = torch.randn(num_kv_heads, head_dim)
            kv_cache[1, 0, s] = torch.randn(num_kv_heads, head_dim)

        q = torch.randn(1, num_heads * head_dim)
        k = torch.randn(1, num_kv_heads * head_dim)
        v = torch.randn(1, num_kv_heads * head_dim)
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.tensor([4])  # 5th token -> block 1, offset 0
        seq_lens = torch.tensor([5])

        out = pa(q, k, v, kv_cache, block_tables, slot_mapping, seq_lens, is_prefill=False)
        assert out.shape == (1, num_heads * head_dim)
