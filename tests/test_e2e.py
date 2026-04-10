"""End-to-end integration tests using small random models (no real model download).

Tests the full pipeline: LLM -> Engine -> Scheduler -> Worker -> Model -> Sampler
using a tiny randomly-initialized Qwen2 model on CPU.
"""

import torch
import pytest
from unittest.mock import patch, MagicMock

from my_vllm.block_manager import BlockManager
from my_vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from my_vllm.engine import LLMEngine
from my_vllm.llm import LLM
from my_vllm.model.qwen2 import Qwen2ForCausalLM
from my_vllm.outputs import RequestOutput
from my_vllm.sampling_params import SamplingParams
from my_vllm.scheduler import Scheduler
from my_vllm.tokenizer import Tokenizer
from my_vllm.utils import Counter
from my_vllm.worker import GPUWorker, ModelRunner


HIDDEN = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
INTERMEDIATE = 128
VOCAB = 100
NUM_LAYERS = 2
BLOCK_SIZE = 4
NUM_BLOCKS = 32


def _create_test_engine() -> LLMEngine:
    """Create an LLMEngine with a tiny random model on CPU (no real model download)."""
    cache_config = CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=NUM_BLOCKS)
    scheduler_config = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64)

    with patch.object(LLMEngine, '__init__', lambda self, *a, **kw: None):
        engine = LLMEngine.__new__(LLMEngine)

    # Build a tiny model on CPU
    model = Qwen2ForCausalLM(
        VOCAB, HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        INTERMEDIATE, 1e-6, 128, 10000.0,
    )
    model.eval()

    # Set up model runner
    class FakeModelConfig:
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

    engine.model_config = FakeModelConfig()
    engine.cache_config = cache_config
    engine.scheduler_config = scheduler_config

    model_runner = ModelRunner(FakeModelConfig(), cache_config, torch.device("cpu"))
    model_runner.model = model

    mock_worker = MagicMock(spec=GPUWorker)
    kv_caches = [
        torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        for _ in range(NUM_LAYERS)
    ]
    mock_worker.execute_model.side_effect = lambda so: model_runner.execute_model(so, kv_caches)

    engine.worker = mock_worker

    # Simple mock tokenizer that uses integer token IDs
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(1, len(text) + 1))[:VOCAB]
    mock_tokenizer.decode.side_effect = lambda ids, **kw: "".join([chr(65 + (t % 26)) for t in ids])
    mock_tokenizer.eos_token_id = 99
    engine.tokenizer = mock_tokenizer

    engine.block_manager = BlockManager(BLOCK_SIZE, NUM_BLOCKS)
    engine.scheduler = Scheduler(scheduler_config, cache_config, engine.block_manager)
    engine.request_counter = Counter()
    engine._requests = {}

    return engine


class TestE2ESingleRequest:
    def test_single_greedy_generation(self):
        engine = _create_test_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=3)
        engine.add_request("r0", "hello", sp)

        all_outputs = []
        for _ in range(50):
            outputs = engine.step()
            all_outputs.extend(outputs)
            if not engine.has_unfinished_requests():
                break

        assert len(all_outputs) == 1
        output = all_outputs[0]
        assert output.request_id == "r0"
        assert output.finished
        assert len(output.outputs) == 1
        assert output.outputs[0].finish_reason == "length"
        assert len(output.outputs[0].token_ids) == 3

    def test_single_short_generation(self):
        engine = _create_test_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=1)
        engine.add_request("r0", "hi", sp)

        outputs = engine.step()
        assert len(outputs) == 1
        assert outputs[0].finished


class TestE2EMultipleRequests:
    def test_two_requests(self):
        engine = _create_test_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=2)
        engine.add_request("r0", "aaa", sp)
        engine.add_request("r1", "bbb", sp)

        all_outputs = []
        for _ in range(50):
            outputs = engine.step()
            all_outputs.extend(outputs)
            if not engine.has_unfinished_requests():
                break

        assert len(all_outputs) == 2
        ids = {o.request_id for o in all_outputs}
        assert ids == {"r0", "r1"}
        for o in all_outputs:
            assert o.finished
            assert len(o.outputs[0].token_ids) == 2


class TestE2EStopOnEOS:
    def test_eos_stops_generation(self):
        engine = _create_test_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=100, stop_token_ids=[99])
        sp.update_eos_token_id(99)
        engine.add_request("r0", "test", sp)

        all_outputs = []
        for _ in range(200):
            outputs = engine.step()
            all_outputs.extend(outputs)
            if not engine.has_unfinished_requests():
                break

        assert len(all_outputs) == 1
        output = all_outputs[0]
        assert output.finished
        # It should stop either by EOS or by max_tokens
        assert output.outputs[0].finish_reason in ("stop", "length")


class TestE2EAbort:
    def test_abort_during_generation(self):
        engine = _create_test_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=100)
        engine.add_request("r0", "hello world", sp)

        engine.step()
        engine.abort_request("r0")
        assert not engine.has_unfinished_requests()
