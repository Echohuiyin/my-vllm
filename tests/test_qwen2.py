"""Tests for Qwen2 model components."""

import torch
import pytest

from my_vllm.model.qwen2 import (
    Qwen2MLP,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
)


HIDDEN = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS  # 16
INTERMEDIATE = 128
VOCAB = 100
NUM_LAYERS = 2
BLOCK_SIZE = 4
NUM_BLOCKS = 16
RMS_EPS = 1e-6
MAX_POS = 128
ROPE_THETA = 10000.0


def _make_kv_caches(
    num_layers: int = NUM_LAYERS,
    num_blocks: int = NUM_BLOCKS,
) -> list[torch.Tensor]:
    return [
        torch.zeros(2, num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        for _ in range(num_layers)
    ]


class TestQwen2MLP:
    def test_output_shape(self):
        mlp = Qwen2MLP(HIDDEN, INTERMEDIATE)
        x = torch.randn(5, HIDDEN)
        out = mlp(x)
        assert out.shape == (5, HIDDEN)


class TestQwen2Attention:
    def test_prefill_shape(self):
        attn = Qwen2Attention(
            HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_POS, ROPE_THETA
        )
        kv_cache = torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        seq_len = 5
        hidden = torch.randn(seq_len, HIDDEN)
        positions = torch.arange(seq_len)
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.arange(seq_len)
        seq_lens = torch.tensor([seq_len])

        out = attn(positions, hidden, kv_cache, block_tables, slot_mapping, seq_lens, True)
        assert out.shape == (seq_len, HIDDEN)

    def test_decode_shape(self):
        attn = Qwen2Attention(
            HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_POS, ROPE_THETA
        )
        kv_cache = torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

        # Pre-fill cache
        for s in range(4):
            kv_cache[0, 0, s] = torch.randn(NUM_KV_HEADS, HEAD_DIM)
            kv_cache[1, 0, s] = torch.randn(NUM_KV_HEADS, HEAD_DIM)

        hidden = torch.randn(1, HIDDEN)
        positions = torch.tensor([4])
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.tensor([4])
        seq_lens = torch.tensor([5])

        out = attn(positions, hidden, kv_cache, block_tables, slot_mapping, seq_lens, False)
        assert out.shape == (1, HIDDEN)


class TestQwen2DecoderLayer:
    def test_output_shape(self):
        layer = Qwen2DecoderLayer(
            HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            INTERMEDIATE, RMS_EPS, MAX_POS, ROPE_THETA,
        )
        kv_cache = torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        seq_len = 5
        hidden = torch.randn(seq_len, HIDDEN)
        positions = torch.arange(seq_len)
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.arange(seq_len)
        seq_lens = torch.tensor([seq_len])

        out, residual = layer(
            positions, hidden, None,
            kv_cache, block_tables, slot_mapping, seq_lens, True,
        )
        assert out.shape == (seq_len, HIDDEN)
        assert residual.shape == (seq_len, HIDDEN)


class TestQwen2Model:
    def test_output_shape(self):
        model = Qwen2Model(
            VOCAB, HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            INTERMEDIATE, RMS_EPS, MAX_POS, ROPE_THETA,
        )
        kv_caches = _make_kv_caches()
        seq_len = 5
        input_ids = torch.randint(0, VOCAB, (seq_len,))
        positions = torch.arange(seq_len)
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.arange(seq_len)
        seq_lens = torch.tensor([seq_len])

        out = model(input_ids, positions, kv_caches, block_tables, slot_mapping, seq_lens, True)
        assert out.shape == (seq_len, HIDDEN)


class TestQwen2ForCausalLM:
    def test_prefill_logits_shape(self):
        model = Qwen2ForCausalLM(
            VOCAB, HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            INTERMEDIATE, RMS_EPS, MAX_POS, ROPE_THETA,
        )
        kv_caches = _make_kv_caches()
        seq_len = 5
        input_ids = torch.randint(0, VOCAB, (seq_len,))
        positions = torch.arange(seq_len)
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.arange(seq_len)
        seq_lens = torch.tensor([seq_len])

        logits = model(input_ids, positions, kv_caches, block_tables, slot_mapping, seq_lens, True)
        # Prefill: returns logits for last token of each sequence
        assert logits.shape == (1, VOCAB)

    def test_decode_logits_shape(self):
        model = Qwen2ForCausalLM(
            VOCAB, HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            INTERMEDIATE, RMS_EPS, MAX_POS, ROPE_THETA,
        )
        kv_caches = _make_kv_caches()

        for kv in kv_caches:
            for s in range(4):
                kv[0, 0, s] = torch.randn(NUM_KV_HEADS, HEAD_DIM)
                kv[1, 0, s] = torch.randn(NUM_KV_HEADS, HEAD_DIM)

        input_ids = torch.randint(0, VOCAB, (1,))
        positions = torch.tensor([4])
        block_tables = torch.tensor([[0, 1]])
        slot_mapping = torch.tensor([4])
        seq_lens = torch.tensor([5])

        logits = model(input_ids, positions, kv_caches, block_tables, slot_mapping, seq_lens, False)
        assert logits.shape == (1, VOCAB)

    def test_load_weights_name_mapping(self):
        model = Qwen2ForCausalLM(
            VOCAB, HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            INTERMEDIATE, RMS_EPS, MAX_POS, ROPE_THETA,
        )
        name = "model.layers.0.self_attn.o_proj.weight"
        mapped = model._map_hf_name(name)
        assert mapped == "model.layers.0.self_attn.o_proj.linear.weight"

    def test_map_hf_name_passthrough(self):
        model = Qwen2ForCausalLM(
            VOCAB, HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            INTERMEDIATE, RMS_EPS, MAX_POS, ROPE_THETA,
        )
        name = "model.embed_tokens.weight"
        assert model._map_hf_name(name) == name
