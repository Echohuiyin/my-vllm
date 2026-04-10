"""GPU Worker and Model Runner for executing model forward passes.

The ModelRunner handles input preparation, model execution, and sampling.
The GPUWorker manages device initialization and KV cache allocation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from my_vllm.block_manager import BlockManager
from my_vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from my_vllm.model.model_loader import create_model, load_model_weights
from my_vllm.model.qwen2 import Qwen2ForCausalLM
from my_vllm.sampler import Sampler
from my_vllm.sampling_params import SamplingParams
from my_vllm.scheduler import SchedulerOutput
from my_vllm.sequence import Request
from my_vllm.utils import cdiv


class ModelRunner:
    """Handles model execution: input preparation, forward pass, sampling."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.model: Optional[Qwen2ForCausalLM] = None
        self.sampler = Sampler()

    def load_model(self) -> None:
        self.model = create_model(self.model_config)
        load_model_weights(self.model, self.model_config)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        kv_caches: List[torch.Tensor],
    ) -> List[int]:
        """Execute one forward pass and sample next tokens.

        Returns list of sampled token IDs, one per scheduled request.
        """
        if scheduler_output.is_empty:
            return []

        (
            input_ids,
            positions,
            block_tables,
            slot_mapping,
            seq_lens,
            is_prefill,
            sampling_params_list,
        ) = self._prepare_inputs(scheduler_output)

        logits = self.model(
            input_ids, positions, kv_caches,
            block_tables, slot_mapping, seq_lens, is_prefill,
        )

        sampled_ids = self.sampler(logits, sampling_params_list)
        return sampled_ids

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
        List[SamplingParams],
    ]:
        """Prepare model input tensors from scheduled requests.

        For prefill: concatenate all prompt tokens.
        For decode: one token per request (the last generated token).
        For mixed: treat as prefill with decode requests having 1 token.
        """
        requests = scheduler_output.scheduled_requests
        has_prefill = scheduler_output.num_prefill_tokens > 0
        has_decode = scheduler_output.num_decode_tokens > 0

        is_prefill = has_prefill and not has_decode

        all_input_ids = []
        all_positions = []
        all_slot_mappings = []
        seq_lens_list = []
        block_tables_list = []
        sampling_params_list = []
        max_blocks = 0

        for req in requests:
            bt = req.block_table
            max_blocks = max(max_blocks, len(bt))

        for req in requests:
            block_size = self.cache_config.block_size

            if req.is_prefill():
                tokens = req.prompt_token_ids
                num_tokens = len(tokens)
                pos_start = 0
                positions = list(range(pos_start, pos_start + num_tokens))
                slots = []
                for p in range(num_tokens):
                    block_idx = p // block_size
                    block_offset = p % block_size
                    if block_idx < len(req.block_table):
                        slots.append(
                            req.block_table[block_idx] * block_size + block_offset
                        )
                all_input_ids.extend(tokens)
                all_positions.extend(positions)
                all_slot_mappings.extend(slots)
                seq_lens_list.append(num_tokens)
            else:
                last_token = req.get_last_token_id()
                pos = req.get_len() - 1
                slot_pos = pos
                block_idx = slot_pos // block_size
                block_offset = slot_pos % block_size
                if block_idx < len(req.block_table):
                    slot = req.block_table[block_idx] * block_size + block_offset
                else:
                    slot = 0
                all_input_ids.append(last_token)
                all_positions.append(pos)
                all_slot_mappings.append(slot)
                seq_lens_list.append(req.get_len())

            bt = req.block_table
            padded_bt = bt + [0] * (max_blocks - len(bt))
            block_tables_list.append(padded_bt)
            sampling_params_list.append(req.sampling_params)

        input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)
        slot_mapping = torch.tensor(all_slot_mappings, dtype=torch.long, device=self.device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=self.device)
        block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=self.device)

        return (
            input_ids, positions, block_tables, slot_mapping,
            seq_lens, is_prefill, sampling_params_list,
        )

    def profile_num_available_blocks(
        self,
        gpu_memory_utilization: float,
    ) -> int:
        """Estimate how many KV cache blocks fit in GPU memory."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        total_memory = torch.cuda.get_device_properties(self.device).total_mem
        peak_memory = torch.cuda.max_memory_allocated(self.device)

        available = total_memory * gpu_memory_utilization - peak_memory
        if available <= 0:
            return 0

        block_size = self.cache_config.block_size
        num_layers = self.model_config.num_hidden_layers
        num_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim

        kv_cache_element_size = torch.tensor(
            [], dtype=self.model_config.torch_dtype
        ).element_size()

        # Each block: 2 (k+v) * block_size * num_kv_heads * head_dim * element_size * num_layers
        bytes_per_block = (
            2 * block_size * num_kv_heads * head_dim
            * kv_cache_element_size * num_layers
        )

        num_blocks = int(available // bytes_per_block)
        return max(num_blocks, 0)


class GPUWorker:
    """Manages GPU device, model, and KV cache."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device or torch.device("cuda:0")
        self.model_runner = ModelRunner(model_config, cache_config, self.device)
        self.kv_caches: List[torch.Tensor] = []

    def init_device(self) -> None:
        torch.cuda.set_device(self.device)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> int:
        return self.model_runner.profile_num_available_blocks(
            self.cache_config.gpu_memory_utilization
        )

    def init_kv_cache(self, num_blocks: int) -> None:
        """Allocate KV cache tensors on GPU."""
        block_size = self.cache_config.block_size
        num_layers = self.model_config.num_hidden_layers
        num_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim
        dtype = self.model_config.torch_dtype

        self.kv_caches = []
        for _ in range(num_layers):
            kv_cache = torch.zeros(
                2, num_blocks, block_size, num_kv_heads, head_dim,
                dtype=dtype, device=self.device,
            )
            self.kv_caches.append(kv_cache)

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> List[int]:
        return self.model_runner.execute_model(scheduler_output, self.kv_caches)
