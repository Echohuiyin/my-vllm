"""Tests for GPU worker and model runner (using CPU and small mock models)."""

import torch
import pytest

from my_vllm.config import CacheConfig, ModelConfig
from my_vllm.model.qwen2 import Qwen2ForCausalLM
from my_vllm.sampler import Sampler
from my_vllm.sampling_params import SamplingParams
from my_vllm.scheduler import SchedulerOutput
from my_vllm.sequence import Request, SequenceStatus
from my_vllm.worker import ModelRunner


HIDDEN = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
INTERMEDIATE = 128
VOCAB = 100
NUM_LAYERS = 2
BLOCK_SIZE = 4
NUM_BLOCKS = 16


class FakeModelConfig:
    """Minimal ModelConfig-like object for testing without downloading a model."""
    model = "test"
    tokenizer = "test"
    trust_remote_code = False
    vocab_size = VOCAB
    hidden_size = HIDDEN
    num_hidden_layers = NUM_LAYERS
    num_attention_heads = NUM_HEADS
    num_key_value_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    intermediate_size = INTERMEDIATE
    rms_norm_eps = 1e-6
    max_position_embeddings = 128
    rope_theta = 10000.0
    torch_dtype = torch.float32
    max_model_len = 128
    seed = 0


def _make_model_runner() -> ModelRunner:
    model_config = FakeModelConfig()
    cache_config = CacheConfig(block_size=BLOCK_SIZE)
    runner = ModelRunner(model_config, cache_config, torch.device("cpu"))
    runner.model = Qwen2ForCausalLM(
        VOCAB, HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        INTERMEDIATE, 1e-6, 128, 10000.0,
    )
    runner.model.eval()
    return runner


def _make_kv_caches() -> list[torch.Tensor]:
    return [
        torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        for _ in range(NUM_LAYERS)
    ]


def _make_request(
    request_id: str = "r0",
    prompt_len: int = 4,
    block_table: list = None,
) -> Request:
    req = Request(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=5),
    )
    if block_table:
        req.block_table = block_table
    return req


class TestModelRunnerPrepareInputs:
    def test_prefill_input_shapes(self):
        runner = _make_model_runner()
        req = _make_request(prompt_len=5, block_table=[0, 1])

        sched_out = SchedulerOutput(
            scheduled_requests=[req],
            num_prefill_tokens=5,
            num_decode_tokens=0,
        )

        input_ids, positions, bt, sm, sl, is_pf, sp_list = runner._prepare_inputs(sched_out)
        assert input_ids.shape == (5,)
        assert positions.shape == (5,)
        assert sm.shape == (5,)
        assert sl.shape == (1,)
        assert sl[0] == 5
        assert is_pf is True
        assert len(sp_list) == 1

    def test_decode_input_shapes(self):
        runner = _make_model_runner()
        req = _make_request(prompt_len=4, block_table=[0, 1])
        req.num_computed_tokens = 4
        req.append_token(42)

        sched_out = SchedulerOutput(
            scheduled_requests=[req],
            num_prefill_tokens=0,
            num_decode_tokens=1,
        )

        input_ids, positions, bt, sm, sl, is_pf, sp_list = runner._prepare_inputs(sched_out)
        assert input_ids.shape == (1,)
        assert input_ids[0] == 42
        assert positions.shape == (1,)
        assert positions[0] == 4  # position is seq_len - 1 = 5 - 1 = 4
        assert is_pf is False

    def test_batch_decode(self):
        runner = _make_model_runner()
        r1 = _make_request("r1", prompt_len=4, block_table=[0])
        r1.num_computed_tokens = 4
        r1.append_token(10)

        r2 = _make_request("r2", prompt_len=3, block_table=[1])
        r2.num_computed_tokens = 3
        r2.append_token(20)

        sched_out = SchedulerOutput(
            scheduled_requests=[r1, r2],
            num_prefill_tokens=0,
            num_decode_tokens=2,
        )

        input_ids, positions, bt, sm, sl, is_pf, sp_list = runner._prepare_inputs(sched_out)
        assert input_ids.shape == (2,)
        assert len(sp_list) == 2


class TestModelRunnerExecute:
    def test_execute_prefill(self):
        runner = _make_model_runner()
        kv_caches = _make_kv_caches()
        req = _make_request(prompt_len=4, block_table=[0])

        sched_out = SchedulerOutput(
            scheduled_requests=[req],
            num_prefill_tokens=4,
            num_decode_tokens=0,
        )

        results = runner.execute_model(sched_out, kv_caches)
        assert len(results) == 1
        assert 0 <= results[0] < VOCAB

    def test_execute_decode(self):
        runner = _make_model_runner()
        kv_caches = _make_kv_caches()

        req = _make_request(prompt_len=4, block_table=[0, 1])
        req.num_computed_tokens = 4
        req.append_token(50)

        for s in range(4):
            for kv in kv_caches:
                kv[0, 0, s] = torch.randn(NUM_KV_HEADS, HEAD_DIM)
                kv[1, 0, s] = torch.randn(NUM_KV_HEADS, HEAD_DIM)

        sched_out = SchedulerOutput(
            scheduled_requests=[req],
            num_prefill_tokens=0,
            num_decode_tokens=1,
        )

        results = runner.execute_model(sched_out, kv_caches)
        assert len(results) == 1
        assert 0 <= results[0] < VOCAB

    def test_execute_empty(self):
        runner = _make_model_runner()
        kv_caches = _make_kv_caches()
        sched_out = SchedulerOutput(scheduled_requests=[])
        results = runner.execute_model(sched_out, kv_caches)
        assert results == []
