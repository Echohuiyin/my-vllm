"""Tests for sequence / request state management."""

import pytest

from my_vllm.sampling_params import SamplingParams
from my_vllm.sequence import Request, SequenceStatus


class TestSequenceStatus:
    def test_waiting_not_finished(self):
        assert not SequenceStatus.is_finished(SequenceStatus.WAITING)

    def test_running_not_finished(self):
        assert not SequenceStatus.is_finished(SequenceStatus.RUNNING)

    def test_finished_stopped(self):
        assert SequenceStatus.is_finished(SequenceStatus.FINISHED_STOPPED)

    def test_finished_length(self):
        assert SequenceStatus.is_finished(SequenceStatus.FINISHED_LENGTH)

    def test_finished_eos(self):
        assert SequenceStatus.is_finished(SequenceStatus.FINISHED_EOS)

    def test_finish_reason_stopped(self):
        assert SequenceStatus.FINISHED_STOPPED.get_finished_reason() == "stop"

    def test_finish_reason_length(self):
        assert SequenceStatus.FINISHED_LENGTH.get_finished_reason() == "length"

    def test_finish_reason_eos(self):
        assert SequenceStatus.FINISHED_EOS.get_finished_reason() == "stop"

    def test_finish_reason_waiting(self):
        assert SequenceStatus.WAITING.get_finished_reason() is None

    def test_finish_reason_running(self):
        assert SequenceStatus.RUNNING.get_finished_reason() is None


class TestRequest:
    def _make_request(self, prompt_len=5, **kwargs):
        return Request(
            request_id="req-0",
            prompt="hello",
            prompt_token_ids=list(range(prompt_len)),
            sampling_params=SamplingParams(),
            **kwargs,
        )

    def test_initial_state(self):
        req = self._make_request()
        assert req.status == SequenceStatus.WAITING
        assert req.get_len() == 5
        assert req.get_prompt_len() == 5
        assert req.get_output_len() == 0
        assert req.output_token_ids == []
        assert req.block_table == []
        assert req.num_computed_tokens == 0

    def test_get_token_ids(self):
        req = self._make_request(prompt_len=3)
        assert req.get_token_ids() == [0, 1, 2]
        req.append_token(10)
        assert req.get_token_ids() == [0, 1, 2, 10]

    def test_get_num_new_tokens_initial(self):
        req = self._make_request(prompt_len=4)
        assert req.get_num_new_tokens() == 4

    def test_get_num_new_tokens_after_compute(self):
        req = self._make_request(prompt_len=4)
        req.num_computed_tokens = 4
        assert req.get_num_new_tokens() == 0
        req.append_token(99)
        assert req.get_num_new_tokens() == 1

    def test_get_last_token_id_prompt_only(self):
        req = self._make_request(prompt_len=3)
        assert req.get_last_token_id() == 2

    def test_get_last_token_id_with_output(self):
        req = self._make_request()
        req.append_token(42)
        assert req.get_last_token_id() == 42

    def test_append_token(self):
        req = self._make_request()
        req.append_token(10)
        req.append_token(20)
        assert req.output_token_ids == [10, 20]
        assert req.get_output_len() == 2
        assert req.get_len() == 7

    def test_is_finished_false(self):
        req = self._make_request()
        assert not req.is_finished()

    def test_is_finished_true(self):
        req = self._make_request()
        req.status = SequenceStatus.FINISHED_EOS
        assert req.is_finished()

    def test_is_prefill_initial(self):
        req = self._make_request(prompt_len=5)
        assert req.is_prefill()

    def test_is_prefill_after_partial_compute(self):
        req = self._make_request(prompt_len=5)
        req.num_computed_tokens = 3
        assert req.is_prefill()

    def test_is_prefill_false_after_full_compute(self):
        req = self._make_request(prompt_len=5)
        req.num_computed_tokens = 5
        assert not req.is_prefill()

    def test_get_num_uncomputed_prompt_tokens(self):
        req = self._make_request(prompt_len=10)
        assert req.get_num_uncomputed_prompt_tokens() == 10
        req.num_computed_tokens = 4
        assert req.get_num_uncomputed_prompt_tokens() == 6
        req.num_computed_tokens = 10
        assert req.get_num_uncomputed_prompt_tokens() == 0

    def test_get_num_uncomputed_prompt_tokens_past_prompt(self):
        req = self._make_request(prompt_len=5)
        req.num_computed_tokens = 7
        assert req.get_num_uncomputed_prompt_tokens() == 0
