"""Qwen2 model implementation for my-vllm.

Implements Qwen2ForCausalLM with paged attention support, referencing
the HuggingFace Qwen2 architecture and vLLM's implementation.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_vllm.attention.paged_attention import PagedAttention
from my_vllm.model.layers import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)


class Qwen2MLP(nn.Module):
    """Qwen2 SwiGLU MLP: gate_up_proj -> SiLU(gate) * up -> down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    """Qwen2 multi-head attention with GQA and RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim, num_heads, num_kv_heads, bias=True
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False
        )
        self.rotary_emb = RotaryEmbedding(
            head_dim, max_position_embeddings, rope_theta, dtype
        )
        self.attn = PagedAttention(num_heads, head_dim, num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_lens: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(
            q, k, v, kv_cache, block_tables, slot_mapping, seq_lens, is_prefill
        )
        output = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    """Single Qwen2 decoder layer: self-attention + MLP with pre-norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            max_position_embeddings, rope_theta, dtype,
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_lens: torch.Tensor,
        is_prefill: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

        hidden_states = self.self_attn(
            positions, hidden_states, kv_cache,
            block_tables, slot_mapping, seq_lens, is_prefill,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    """Qwen2 transformer model (without LM head)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                hidden_size, num_attention_heads, num_kv_heads, head_dim,
                intermediate_size, rms_norm_eps, max_position_embeddings,
                rope_theta, dtype,
            )
            for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_lens: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                kv_caches[i], block_tables, slot_mapping, seq_lens, is_prefill,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 for causal language modeling."""

    # Maps vLLM fused names -> HuggingFace separate weight names
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model = Qwen2Model(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            num_kv_heads, head_dim, intermediate_size, rms_norm_eps,
            max_position_embeddings, rope_theta, dtype,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_lens: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        """Returns logits for the last token of each sequence."""
        hidden_states = self.model(
            input_ids, positions, kv_caches,
            block_tables, slot_mapping, seq_lens, is_prefill,
        )
        if is_prefill:
            last_positions = seq_lens.cumsum(0) - 1
            hidden_states = hidden_states[last_positions]
        logits = self.lm_head(hidden_states)
        return logits.float()

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> None:
        """Load weights from HuggingFace checkpoint format.

        Handles merging separate Q/K/V projections into fused qkv_proj,
        and gate/up projections into fused gate_up_proj.
        """
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            matched = False
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Map to our internal naming: "model.layers.X.self_attn.qkv_proj.linear.weight"
                param_key = self._map_hf_name(name)
                if param_key not in params_dict:
                    continue

                param = params_dict[param_key]
                self._load_shard(param, loaded_weight, shard_id, param_name)
                matched = True
                break

            if not matched:
                param_key = self._map_hf_name(name)
                if param_key in params_dict:
                    param = params_dict[param_key]
                    param.data.copy_(loaded_weight)

    def _map_hf_name(self, name: str) -> str:
        """Map HF weight name to my_vllm parameter name.

        HF uses e.g. 'model.layers.0.self_attn.o_proj.weight'
        We use 'model.layers.0.self_attn.o_proj.linear.weight'
        """
        replacements = {
            "self_attn.qkv_proj.weight": "self_attn.qkv_proj.linear.weight",
            "self_attn.qkv_proj.bias": "self_attn.qkv_proj.linear.bias",
            "self_attn.o_proj.weight": "self_attn.o_proj.linear.weight",
            "self_attn.o_proj.bias": "self_attn.o_proj.linear.bias",
            "mlp.gate_up_proj.weight": "mlp.gate_up_proj.linear.weight",
            "mlp.gate_up_proj.bias": "mlp.gate_up_proj.linear.bias",
            "mlp.down_proj.weight": "mlp.down_proj.linear.weight",
            "mlp.down_proj.bias": "mlp.down_proj.linear.bias",
        }
        for old, new in replacements.items():
            if name.endswith(old):
                return name[: -len(old)] + new
        return name

    def _load_shard(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | int,
        param_name: str,
    ) -> None:
        """Load a shard of a fused weight (q/k/v or gate/up)."""
        if param_name == "qkv_proj":
            head_dim = loaded_weight.shape[0] // (
                loaded_weight.shape[0] // self._get_head_dim()
            )
            num_heads_loaded = loaded_weight.shape[0] // head_dim

            if shard_id == "q":
                offset = 0
            elif shard_id == "k":
                q_size = self._get_num_heads() * head_dim
                offset = q_size
            else:  # v
                q_size = self._get_num_heads() * head_dim
                kv_size = self._get_num_kv_heads() * head_dim
                offset = q_size + kv_size

            size = loaded_weight.shape[0]
            if loaded_weight.dim() == 2:
                param.data[offset: offset + size, :] = loaded_weight
            else:
                param.data[offset: offset + size] = loaded_weight

        elif param_name == "gate_up_proj":
            half = param.data.shape[0] // 2
            if shard_id == 0:
                param.data[:half] = loaded_weight
            else:
                param.data[half:] = loaded_weight

    def _get_head_dim(self) -> int:
        layer = self.model.layers[0]
        return layer.self_attn.head_dim

    def _get_num_heads(self) -> int:
        layer = self.model.layers[0]
        return layer.self_attn.num_heads

    def _get_num_kv_heads(self) -> int:
        layer = self.model.layers[0]
        return layer.self_attn.num_kv_heads
