"""PagedAttention implementation using pure PyTorch.

Implements the paged KV cache mechanism: K/V tensors are stored in
fixed-size blocks, read via block tables during decode, and written
via slot mappings during both prefill and decode.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PagedAttention(nn.Module):
    """Attention with paged KV cache.

    KV cache layout: (2, num_blocks, block_size, num_kv_heads, head_dim)
      - index 0 = key cache, index 1 = value cache

    Args:
        num_heads: number of query attention heads
        head_dim: dimension per head
        num_kv_heads: number of key/value heads (for GQA)
        scale: attention scale factor (default 1/sqrt(head_dim))
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = scale or (1.0 / math.sqrt(head_dim))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_lens: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        """
        Args:
            query:  (num_tokens, num_heads * head_dim)
            key:    (num_tokens, num_kv_heads * head_dim)
            value:  (num_tokens, num_kv_heads * head_dim)
            kv_cache: (2, num_blocks, block_size, num_kv_heads, head_dim)
            block_tables: (batch_size, max_blocks_per_seq)
            slot_mapping: (num_tokens,)
            seq_lens: (batch_size,) — total length of each sequence
            is_prefill: whether this is a prefill step
        """
        self._write_to_cache(key, value, kv_cache, slot_mapping)

        if is_prefill:
            return self._prefill_attention(query, key, value, seq_lens)
        else:
            return self._decode_attention(
                query, kv_cache, block_tables, seq_lens
            )

    def _write_to_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write K/V into the paged cache using slot_mapping."""
        block_size = kv_cache.shape[2]
        block_ids = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        k = key.view(-1, self.num_kv_heads, self.head_dim)
        v = value.view(-1, self.num_kv_heads, self.head_dim)

        kv_cache[0, block_ids, block_offsets] = k
        kv_cache[1, block_ids, block_offsets] = v

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Standard causal attention for prefill, handling variable-length sequences."""
        outputs = []
        offset = 0
        for seq_len in seq_lens.tolist():
            seq_len = int(seq_len)
            q = query[offset: offset + seq_len].view(
                seq_len, self.num_heads, self.head_dim
            )
            k = key[offset: offset + seq_len].view(
                seq_len, self.num_kv_heads, self.head_dim
            )
            v = value[offset: offset + seq_len].view(
                seq_len, self.num_kv_heads, self.head_dim
            )

            if self.num_kv_groups > 1:
                k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1)
                k = k.reshape(seq_len, self.num_heads, self.head_dim)
                v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1)
                v = v.reshape(seq_len, self.num_heads, self.head_dim)

            # (num_heads, seq_len, head_dim)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

            out = F.scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                is_causal=True, scale=self.scale,
            )
            out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
            outputs.append(out)
            offset += seq_len

        return torch.cat(outputs, dim=0)

    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Paged attention for decode: gather KV from block tables."""
        batch_size = query.shape[0]
        block_size = kv_cache.shape[2]

        q = query.view(batch_size, self.num_heads, self.head_dim)
        outputs = []

        for i in range(batch_size):
            seq_len = int(seq_lens[i].item())
            blocks = block_tables[i]
            num_blocks_needed = (seq_len + block_size - 1) // block_size

            k_parts = []
            v_parts = []
            remaining = seq_len
            for b in range(num_blocks_needed):
                block_id = int(blocks[b].item())
                slots_in_block = min(block_size, remaining)
                k_parts.append(kv_cache[0, block_id, :slots_in_block])
                v_parts.append(kv_cache[1, block_id, :slots_in_block])
                remaining -= slots_in_block

            k = torch.cat(k_parts, dim=0)  # (seq_len, num_kv_heads, head_dim)
            v = torch.cat(v_parts, dim=0)

            if self.num_kv_groups > 1:
                k = k.unsqueeze(1).expand(-1, self.num_kv_groups, -1, -1)
                k = k.reshape(seq_len, self.num_heads, self.head_dim)
                v = v.unsqueeze(1).expand(-1, self.num_kv_groups, -1, -1)
                v = v.reshape(seq_len, self.num_heads, self.head_dim)

            qi = q[i].unsqueeze(1)  # (num_heads, 1, head_dim)
            k = k.transpose(0, 1)  # (num_heads, seq_len, head_dim)
            v = v.transpose(0, 1)

            attn_weights = torch.matmul(
                qi, k.transpose(-2, -1)
            ) * self.scale  # (num_heads, 1, seq_len)
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v)  # (num_heads, 1, head_dim)
            out = out.squeeze(1).reshape(-1)
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (batch_size, num_heads * head_dim)
