"""Tests for LLMEngine.

These tests mock the worker to avoid GPU/model dependencies,
focusing on the engine's request management and step logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from my_vllm.config import CacheConfig, SchedulerConfig
from my_vllm.engine import LLMEngine
from my_vllm.outputs import RequestOutput
from my_vllm.sampling_params import SamplingParams
from my_vllm.sequence import Request, SequenceStatus


class FakeModelConfig:
    model = "test"
    tokenizer = "test"
    trust_remote_code = False
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
    torch_dtype = None
    max_model_len = 128
    seed = 0


def _create_engine_with_mocks():
    """Create an LLMEngine with mocked worker and tokenizer."""
    model_config = FakeModelConfig()
    cache_config = CacheConfig(block_size=4, num_gpu_blocks=20)
    scheduler_config = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=32)

    with patch.object(LLMEngine, '__init__', lambda self, *a, **kw: None):
        engine = LLMEngine.__new__(LLMEngine)

    engine.model_config = model_config
    engine.cache_config = cache_config
    engine.scheduler_config = scheduler_config

    from my_vllm.block_manager import BlockManager
    from my_vllm.scheduler import Scheduler
    from my_vllm.utils import Counter

    engine.block_manager = BlockManager(
        block_size=cache_config.block_size,
        num_gpu_blocks=cache_config.num_gpu_blocks,
    )
    engine.scheduler = Scheduler(
        scheduler_config, cache_config, engine.block_manager,
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    mock_tokenizer.decode.return_value = "generated text"
    mock_tokenizer.eos_token_id = 151643
    engine.tokenizer = mock_tokenizer

    mock_worker = MagicMock()
    mock_worker.execute_model.return_value = [42]
    engine.worker = mock_worker

    engine.request_counter = Counter()
    engine._requests = {}

    return engine


class TestEngineAddRequest:
    def test_add_request(self):
        engine = _create_engine_with_mocks()
        engine.add_request("r0", "hello", SamplingParams(max_tokens=5))
        assert engine.has_unfinished_requests()
        assert "r0" in engine._requests

    def test_add_multiple_requests(self):
        engine = _create_engine_with_mocks()
        engine.add_request("r0", "hello", SamplingParams())
        engine.add_request("r1", "world", SamplingParams())
        assert len(engine._requests) == 2


class TestEngineStep:
    def test_step_returns_finished(self):
        engine = _create_engine_with_mocks()
        engine.add_request("r0", "hi", SamplingParams(max_tokens=1))

        engine.worker.execute_model.return_value = [42]
        outputs = engine.step()

        assert len(outputs) == 1
        assert outputs[0].request_id == "r0"
        assert outputs[0].finished
        assert not engine.has_unfinished_requests()

    def test_step_unfinished(self):
        engine = _create_engine_with_mocks()
        engine.add_request("r0", "hi", SamplingParams(max_tokens=5))

        engine.worker.execute_model.return_value = [42]
        outputs = engine.step()

        assert len(outputs) == 0
        assert engine.has_unfinished_requests()

    def test_multiple_steps_to_finish(self):
        engine = _create_engine_with_mocks()
        engine.add_request("r0", "hi", SamplingParams(max_tokens=3))

        all_outputs = []
        engine.worker.execute_model.return_value = [42]
        for _ in range(10):
            outputs = engine.step()
            all_outputs.extend(outputs)
            if not engine.has_unfinished_requests():
                break

        assert len(all_outputs) == 1
        assert all_outputs[0].finished
        assert all_outputs[0].outputs[0].finish_reason == "length"

    def test_step_empty_when_no_requests(self):
        engine = _create_engine_with_mocks()
        outputs = engine.step()
        assert outputs == []


class TestEngineAbort:
    def test_abort_removes_request(self):
        engine = _create_engine_with_mocks()
        engine.add_request("r0", "hi", SamplingParams())
        engine.abort_request("r0")
        assert not engine.has_unfinished_requests()
        assert "r0" not in engine._requests


class TestEngineMakeOutput:
    def test_output_structure(self):
        engine = _create_engine_with_mocks()
        req = Request(
            request_id="r0",
            prompt="hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            output_token_ids=[10, 20],
            status=SequenceStatus.FINISHED_LENGTH,
        )
        output = engine._make_output(req)
        assert isinstance(output, RequestOutput)
        assert output.request_id == "r0"
        assert output.prompt == "hello"
        assert output.finished
        assert output.outputs[0].finish_reason == "length"
        assert output.outputs[0].token_ids == [10, 20]
