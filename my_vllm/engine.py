"""LLMEngine: the core inference engine.

Coordinates tokenizer, scheduler, and worker to process generation requests.
Each ``step()`` call runs one scheduling + model execution cycle.
"""

from __future__ import annotations

from typing import List, Optional

from my_vllm.block_manager import BlockManager
from my_vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from my_vllm.outputs import CompletionOutput, RequestOutput
from my_vllm.sampling_params import SamplingParams
from my_vllm.scheduler import Scheduler
from my_vllm.sequence import Request, SequenceStatus
from my_vllm.tokenizer import Tokenizer
from my_vllm.utils import Counter
from my_vllm.worker import GPUWorker


class LLMEngine:
    """Synchronous inference engine.

    Manages the full lifecycle of requests: tokenization, scheduling,
    model execution, detokenization, and output assembly.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config

        self.tokenizer = Tokenizer(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.worker = GPUWorker(model_config, cache_config)
        self.worker.init_device()
        self.worker.load_model()

        if cache_config.num_gpu_blocks is None:
            num_blocks = self.worker.determine_num_available_blocks()
            cache_config.num_gpu_blocks = max(num_blocks, 16)

        self.worker.init_kv_cache(cache_config.num_gpu_blocks)

        self.block_manager = BlockManager(
            block_size=cache_config.block_size,
            num_gpu_blocks=cache_config.num_gpu_blocks,
        )
        self.scheduler = Scheduler(
            scheduler_config, cache_config, self.block_manager
        )

        self.request_counter = Counter()
        self._requests: dict[str, Request] = {}

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> None:
        """Add a new generation request."""
        prompt_token_ids = self.tokenizer.encode(prompt)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            sampling_params.update_eos_token_id(eos_id)

        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        self._requests[request_id] = request
        self.scheduler.add_request(request)

    def step(self) -> List[RequestOutput]:
        """Run one scheduling + execution step. Returns outputs for finished requests."""
        scheduler_output = self.scheduler.schedule()

        if scheduler_output.is_empty:
            return []

        sampled_ids = self.worker.execute_model(scheduler_output)

        finished = self.scheduler.update_from_output(
            scheduler_output.scheduled_requests, sampled_ids
        )

        outputs = []
        for req in finished:
            output = self._make_output(req)
            outputs.append(output)
            del self._requests[req.request_id]

        return outputs

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def abort_request(self, request_id: str) -> None:
        self.scheduler.abort_request(request_id)
        self._requests.pop(request_id, None)

    def _make_output(self, request: Request) -> RequestOutput:
        """Build a RequestOutput from a finished request."""
        output_text = self.tokenizer.decode(request.output_token_ids)
        finish_reason = request.status.get_finished_reason()

        completion = CompletionOutput(
            index=0,
            text=output_text,
            token_ids=list(request.output_token_ids),
            finish_reason=finish_reason,
        )

        return RequestOutput(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=list(request.prompt_token_ids),
            outputs=[completion],
            finished=True,
        )
