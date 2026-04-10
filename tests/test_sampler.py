"""Tests for the sampler."""

import torch
import pytest

from my_vllm.sampler import Sampler
from my_vllm.sampling_params import SamplingParams


class TestSamplerGreedy:
    def test_greedy_picks_argmax(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        sp = SamplingParams(temperature=0.0)
        results = sampler(logits, [sp])
        assert results == [1]

    def test_greedy_batch(self):
        sampler = Sampler()
        logits = torch.tensor([
            [1.0, 5.0, 2.0],
            [3.0, 1.0, 0.0],
        ])
        sp = SamplingParams(temperature=0.0)
        results = sampler(logits, [sp, sp])
        assert results == [1, 0]


class TestSamplerRandom:
    def test_random_returns_valid_token(self):
        sampler = Sampler()
        vocab_size = 100
        logits = torch.randn(1, vocab_size)
        sp = SamplingParams(temperature=1.0)
        results = sampler(logits, [sp])
        assert 0 <= results[0] < vocab_size

    def test_seed_reproducibility(self):
        sampler = Sampler()
        logits = torch.randn(1, 1000)
        sp = SamplingParams(temperature=1.0, seed=42)
        r1 = sampler(logits.clone(), [sp])
        r2 = sampler(logits.clone(), [sp])
        assert r1 == r2


class TestSamplerTopK:
    def test_top_k(self):
        sampler = Sampler()
        logits = torch.tensor([[10.0, 9.0, 1.0, 1.0, 1.0]])
        sp = SamplingParams(temperature=0.01, top_k=2)
        results = sampler(logits, [sp])
        assert results[0] in [0, 1]


class TestSamplerTopP:
    def test_top_p(self):
        sampler = Sampler()
        logits = torch.zeros(1, 5)
        logits[0, 0] = 100.0
        sp = SamplingParams(temperature=1.0, top_p=0.1)
        results = sampler(logits, [sp])
        assert results[0] == 0


class TestSamplerApplyTopK:
    def test_masks_low_values(self):
        logits = torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1])
        filtered = Sampler._apply_top_k(logits, top_k=2)
        assert filtered[0] == 5.0
        assert filtered[1] == 3.0
        assert filtered[2] == float("-inf")


class TestSamplerApplyTopP:
    def test_keeps_top_probability_mass(self):
        logits = torch.tensor([10.0, 5.0, 0.0, -5.0, -10.0])
        filtered = Sampler._apply_top_p(logits, top_p=0.9)
        probs = torch.softmax(filtered, dim=-1)
        assert probs[0] > 0
        assert probs[-1] < 1e-6
