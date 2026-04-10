"""High-level LLM class for offline inference.

This is the main user-facing API. Usage:

    from my_vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2-0.5B")
    outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=50))
    for output in outputs:
        print(output.outputs[0].text)
"""

from __future__ import annotations

from typing import List, Optional, Union

from my_vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from my_vllm.engine import LLMEngine
from my_vllm.outputs import RequestOutput
from my_vllm.sampling_params import SamplingParams
from my_vllm.utils import Counter


class LLM:
    """Offline LLM inference engine.

    Wraps LLMEngine to provide a simple ``generate()`` interface that
    accepts prompts and returns completed results.
    """

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 128,
        max_num_batched_tokens: int = 2048,
        block_size: int = 16,
        trust_remote_code: bool = False,
        seed: int = 0,
    ) -> None:
        model_config = ModelConfig(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )

        cache_config = CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        self.engine = LLMEngine(model_config, cache_config, scheduler_config)
        self.request_counter = Counter()

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: a single prompt string or list of prompt strings
            sampling_params: sampling parameters (shared across all prompts)

        Returns:
            list of RequestOutput, one per prompt, in the same order
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()

        request_ids = []
        for prompt in prompts:
            request_id = str(next(self.request_counter))
            self.engine.add_request(request_id, prompt, sampling_params)
            request_ids.append(request_id)

        return self._run_engine(request_ids)

    def _run_engine(self, request_ids: List[str]) -> List[RequestOutput]:
        """Run the engine until all requests are complete."""
        outputs_map: dict[str, RequestOutput] = {}

        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                outputs_map[output.request_id] = output

        return [outputs_map[rid] for rid in request_ids]
