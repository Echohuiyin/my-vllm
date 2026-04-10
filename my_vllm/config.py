"""Configuration classes for my-vllm engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model: str
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    trust_remote_code: bool = False
    tokenizer: Optional[str] = None
    seed: int = 0

    hf_config: PretrainedConfig = field(init=False, repr=False)
    vocab_size: int = field(init=False)
    hidden_size: int = field(init=False)
    num_hidden_layers: int = field(init=False)
    num_attention_heads: int = field(init=False)
    num_key_value_heads: int = field(init=False)
    intermediate_size: int = field(init=False)
    head_dim: int = field(init=False)
    max_position_embeddings: int = field(init=False)
    rms_norm_eps: float = field(init=False)
    rope_theta: float = field(init=False)
    torch_dtype: torch.dtype = field(init=False)

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = self.model

        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )

        self.vocab_size = self.hf_config.vocab_size
        self.hidden_size = self.hf_config.hidden_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.num_attention_heads = self.hf_config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.hf_config, "num_key_value_heads", self.num_attention_heads
        )
        self.intermediate_size = self.hf_config.intermediate_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = getattr(
            self.hf_config, "max_position_embeddings", 8192
        )
        self.rms_norm_eps = getattr(self.hf_config, "rms_norm_eps", 1e-6)
        self.rope_theta = getattr(self.hf_config, "rope_theta", 10000.0)

        self.torch_dtype = _resolve_dtype(self.dtype, self.hf_config)

        if self.max_model_len is None:
            self.max_model_len = self.max_position_embeddings
        if self.max_model_len > self.max_position_embeddings:
            raise ValueError(
                f"max_model_len ({self.max_model_len}) exceeds "
                f"max_position_embeddings ({self.max_position_embeddings})"
            )

    def get_num_kv_heads(self) -> int:
        return self.num_key_value_heads

    def get_head_size(self) -> int:
        return self.head_dim


@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    num_gpu_blocks: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError(
                f"max_num_seqs must be >= 1, got {self.max_num_seqs}"
            )
        if self.max_num_batched_tokens < 1:
            raise ValueError(
                f"max_num_batched_tokens must be >= 1, "
                f"got {self.max_num_batched_tokens}"
            )


def _resolve_dtype(dtype_str: str, hf_config: PretrainedConfig) -> torch.dtype:
    """Resolve dtype string to torch.dtype."""
    if dtype_str == "auto":
        cfg_dtype = getattr(hf_config, "torch_dtype", None)
        if cfg_dtype is not None:
            if isinstance(cfg_dtype, str):
                return getattr(torch, cfg_dtype, torch.float16)
            return cfg_dtype
        return torch.float16

    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if dtype_str in mapping:
        return mapping[dtype_str]
    raise ValueError(f"Unsupported dtype: {dtype_str}")
