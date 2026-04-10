"""Tests for model loader (unit tests with mock data, no actual model download)."""

import os
import tempfile

import torch
import pytest

from my_vllm.model.model_loader import _iterate_weights, create_model
from my_vllm.model.qwen2 import Qwen2ForCausalLM


class TestCreateModel:
    def test_creates_correct_type(self):
        class FakeConfig:
            model = "test"
            vocab_size = 100
            hidden_size = 64
            num_hidden_layers = 2
            num_attention_heads = 4
            num_key_value_heads = 2
            head_dim = 16
            intermediate_size = 128
            rms_norm_eps = 1e-6
            max_position_embeddings = 128
            rope_theta = 10000.0
            torch_dtype = torch.float32

        model = create_model(FakeConfig())
        assert isinstance(model, Qwen2ForCausalLM)


class TestIterateWeights:
    def test_safetensors(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tensors = {
                "layer.weight": torch.randn(4, 4),
                "layer.bias": torch.randn(4),
            }
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            loaded = dict(_iterate_weights(tmpdir, torch.float32))
            assert "layer.weight" in loaded
            assert "layer.bias" in loaded
            assert loaded["layer.weight"].shape == (4, 4)

    def test_no_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                list(_iterate_weights(tmpdir, torch.float32))

    def test_dtype_conversion(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tensors = {"w": torch.randn(4, 4, dtype=torch.float32)}
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            loaded = dict(_iterate_weights(tmpdir, torch.float16))
            assert loaded["w"].dtype == torch.float16
