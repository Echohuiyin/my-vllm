"""Tests for configuration classes."""

import pytest
import torch

from my_vllm.config import ModelConfig, CacheConfig, SchedulerConfig, _resolve_dtype


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.block_size == 16
        assert cfg.gpu_memory_utilization == 0.9
        assert cfg.num_gpu_blocks is None

    def test_custom_values(self):
        cfg = CacheConfig(block_size=32, gpu_memory_utilization=0.8)
        assert cfg.block_size == 32
        assert cfg.gpu_memory_utilization == 0.8

    def test_invalid_gpu_memory_utilization_zero(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            CacheConfig(gpu_memory_utilization=0.0)

    def test_invalid_gpu_memory_utilization_over_one(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            CacheConfig(gpu_memory_utilization=1.5)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            CacheConfig(block_size=0)


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.max_num_seqs == 128
        assert cfg.max_num_batched_tokens == 2048

    def test_custom_values(self):
        cfg = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=4096)
        assert cfg.max_num_seqs == 64
        assert cfg.max_num_batched_tokens == 4096

    def test_invalid_max_num_seqs(self):
        with pytest.raises(ValueError, match="max_num_seqs"):
            SchedulerConfig(max_num_seqs=0)

    def test_invalid_max_num_batched_tokens(self):
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            SchedulerConfig(max_num_batched_tokens=0)


class TestResolveDtype:
    def test_explicit_float16(self):
        assert _resolve_dtype("float16", None) == torch.float16

    def test_explicit_half(self):
        assert _resolve_dtype("half", None) == torch.float16

    def test_explicit_bfloat16(self):
        assert _resolve_dtype("bfloat16", None) == torch.bfloat16

    def test_explicit_float32(self):
        assert _resolve_dtype("float32", None) == torch.float32

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _resolve_dtype("int8", None)

    def test_auto_with_hf_config(self):
        class FakeConfig:
            torch_dtype = torch.bfloat16

        assert _resolve_dtype("auto", FakeConfig()) == torch.bfloat16

    def test_auto_with_string_dtype(self):
        class FakeConfig:
            torch_dtype = "bfloat16"

        assert _resolve_dtype("auto", FakeConfig()) == torch.bfloat16

    def test_auto_fallback(self):
        class FakeConfig:
            pass

        assert _resolve_dtype("auto", FakeConfig()) == torch.float16
