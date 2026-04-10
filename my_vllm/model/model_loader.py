"""Model loader: loads HuggingFace model weights into Qwen2ForCausalLM."""

from __future__ import annotations

import os
import glob
from typing import Iterator, Tuple

import torch
from safetensors.torch import load_file as safetensors_load_file

from my_vllm.config import ModelConfig
from my_vllm.model.qwen2 import Qwen2ForCausalLM


def create_model(model_config: ModelConfig) -> Qwen2ForCausalLM:
    """Instantiate a Qwen2ForCausalLM from ModelConfig (empty weights)."""
    model = Qwen2ForCausalLM(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        intermediate_size=model_config.intermediate_size,
        rms_norm_eps=model_config.rms_norm_eps,
        max_position_embeddings=model_config.max_position_embeddings,
        rope_theta=model_config.rope_theta,
        dtype=model_config.torch_dtype,
    )
    return model


def load_model_weights(
    model: Qwen2ForCausalLM,
    model_config: ModelConfig,
) -> None:
    """Load weights from disk (safetensors or PyTorch format)."""
    model_path = _resolve_model_path(model_config.model)
    weights_iter = _iterate_weights(model_path, model_config.torch_dtype)
    model.load_weights(weights_iter)
    model.to(model_config.torch_dtype)


def _resolve_model_path(model_name_or_path: str) -> str:
    """Resolve to a local directory path.

    If it's already a local path, return as-is.
    Otherwise, download from HuggingFace Hub.
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    from huggingface_hub import snapshot_download
    return snapshot_download(model_name_or_path)


def _iterate_weights(
    model_path: str,
    dtype: torch.dtype,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Iterate over weight tensors from safetensors or .bin files."""
    safetensors_files = sorted(glob.glob(
        os.path.join(model_path, "*.safetensors")
    ))
    if safetensors_files:
        for filepath in safetensors_files:
            state_dict = safetensors_load_file(filepath)
            for name, tensor in state_dict.items():
                yield name, tensor.to(dtype)
        return

    bin_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
    if bin_files:
        for filepath in bin_files:
            state_dict = torch.load(filepath, map_location="cpu")
            for name, tensor in state_dict.items():
                yield name, tensor.to(dtype)
        return

    raise FileNotFoundError(
        f"No safetensors or .bin weight files found in {model_path}"
    )
