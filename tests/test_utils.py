"""Tests for utility functions."""

import torch

from my_vllm.utils import set_random_seed, get_dtype_size, cdiv, Counter


class TestSetRandomSeed:
    def test_reproducibility(self):
        set_random_seed(42)
        a = torch.randn(5)
        set_random_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_random_seed(1)
        a = torch.randn(100)
        set_random_seed(2)
        b = torch.randn(100)
        assert not torch.equal(a, b)


class TestGetDtypeSize:
    def test_float16(self):
        assert get_dtype_size(torch.float16) == 2

    def test_bfloat16(self):
        assert get_dtype_size(torch.bfloat16) == 2

    def test_float32(self):
        assert get_dtype_size(torch.float32) == 4


class TestCdiv:
    def test_exact_division(self):
        assert cdiv(10, 5) == 2

    def test_ceil(self):
        assert cdiv(11, 5) == 3

    def test_one(self):
        assert cdiv(1, 5) == 1

    def test_zero(self):
        assert cdiv(0, 5) == 0


class TestCounter:
    def test_counting(self):
        c = Counter()
        assert next(c) == 0
        assert next(c) == 1
        assert next(c) == 2

    def test_start_value(self):
        c = Counter(start=10)
        assert next(c) == 10
        assert next(c) == 11

    def test_reset(self):
        c = Counter()
        next(c)
        next(c)
        c.reset()
        assert next(c) == 0

    def test_iterator(self):
        c = Counter()
        vals = []
        for i, v in enumerate(c):
            vals.append(v)
            if i >= 2:
                break
        assert vals == [0, 1, 2]
