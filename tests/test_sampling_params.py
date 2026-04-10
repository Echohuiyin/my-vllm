"""Tests for SamplingParams."""

import pytest

from my_vllm.sampling_params import SamplingParams


class TestSamplingParams:
    def test_defaults(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.top_p == 1.0
        assert sp.top_k == 0
        assert sp.max_tokens == 16
        assert sp.stop_token_ids == []
        assert sp.seed is None
        assert sp.n == 1
        assert not sp.ignore_eos

    def test_greedy(self):
        sp = SamplingParams(temperature=0.0)
        assert sp.is_greedy
        assert sp.top_p == 1.0
        assert sp.top_k == 0

    def test_not_greedy(self):
        sp = SamplingParams(temperature=0.8)
        assert not sp.is_greedy

    def test_stop_token_ids(self):
        sp = SamplingParams(stop_token_ids=[1, 2, 3])
        assert sp.stop_token_ids == [1, 2, 3]
        assert sp._all_stop_token_ids == {1, 2, 3}

    def test_update_eos_token_id(self):
        sp = SamplingParams()
        sp.update_eos_token_id(151643)
        assert 151643 in sp._all_stop_token_ids

    def test_update_eos_ignored_when_ignore_eos(self):
        sp = SamplingParams(ignore_eos=True)
        sp.update_eos_token_id(151643)
        assert 151643 not in sp._all_stop_token_ids

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-1.0)

    def test_invalid_top_p_zero(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)

    def test_invalid_top_p_over_one(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.5)

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-1)

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=0)

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n"):
            SamplingParams(n=0)

    def test_none_max_tokens(self):
        sp = SamplingParams(max_tokens=None)
        assert sp.max_tokens is None
