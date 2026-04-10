"""Tests for the LLM class (mocked engine, no GPU required)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from my_vllm.llm import LLM
from my_vllm.outputs import CompletionOutput, RequestOutput
from my_vllm.sampling_params import SamplingParams


def _make_mock_llm():
    """Create an LLM with mocked engine to avoid model loading."""
    with patch.object(LLM, '__init__', lambda self, *a, **kw: None):
        llm = LLM.__new__(LLM)

    from my_vllm.utils import Counter
    llm.request_counter = Counter()

    mock_engine = MagicMock()
    mock_engine.has_unfinished_requests.side_effect = [True, False]
    mock_engine.step.return_value = [
        RequestOutput(
            request_id="0",
            prompt="hello",
            prompt_token_ids=[1, 2, 3],
            outputs=[CompletionOutput(
                index=0, text="world", token_ids=[4, 5],
                finish_reason="length",
            )],
            finished=True,
        )
    ]
    llm.engine = mock_engine
    return llm


class TestLLMGenerate:
    def test_single_prompt(self):
        llm = _make_mock_llm()
        outputs = llm.generate("hello", SamplingParams(max_tokens=5))
        assert len(outputs) == 1
        assert outputs[0].finished
        llm.engine.add_request.assert_called_once()

    def test_string_prompt_wrapped(self):
        llm = _make_mock_llm()
        llm.generate("single prompt")
        llm.engine.add_request.assert_called_once()

    def test_multiple_prompts(self):
        llm = _make_mock_llm()
        llm.engine.has_unfinished_requests.side_effect = [True, True, False]
        out1 = RequestOutput(
            request_id="0", prompt="a", prompt_token_ids=[1],
            outputs=[CompletionOutput(0, "x", [2], "length")],
            finished=True,
        )
        out2 = RequestOutput(
            request_id="1", prompt="b", prompt_token_ids=[3],
            outputs=[CompletionOutput(0, "y", [4], "length")],
            finished=True,
        )
        llm.engine.step.side_effect = [[out1], [out2]]

        outputs = llm.generate(["a", "b"], SamplingParams(max_tokens=1))
        assert len(outputs) == 2
        assert outputs[0].request_id == "0"
        assert outputs[1].request_id == "1"

    def test_default_sampling_params(self):
        llm = _make_mock_llm()
        outputs = llm.generate("hello")
        assert len(outputs) == 1

    def test_request_ids_are_unique(self):
        llm = _make_mock_llm()
        llm.engine.has_unfinished_requests.side_effect = [True, True, False]
        out1 = RequestOutput(
            request_id="0", prompt="a", prompt_token_ids=[1],
            outputs=[CompletionOutput(0, "x", [2], "stop")],
            finished=True,
        )
        out2 = RequestOutput(
            request_id="1", prompt="b", prompt_token_ids=[1],
            outputs=[CompletionOutput(0, "y", [3], "stop")],
            finished=True,
        )
        llm.engine.step.side_effect = [[out1], [out2]]

        llm.generate(["a", "b"])
        calls = llm.engine.add_request.call_args_list
        ids = [c[0][0] for c in calls]
        assert len(set(ids)) == 2
