"""Utility functions for my-vllm."""

from __future__ import annotations

import random
from typing import Iterator

import torch


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size in bytes for a given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


class Counter:
    """Thread-safe counter for generating unique request IDs."""

    def __init__(self, start: int = 0) -> None:
        self._count = start

    def __next__(self) -> int:
        val = self._count
        self._count += 1
        return val

    def __iter__(self) -> Iterator[int]:
        return self

    def reset(self) -> None:
        self._count = 0
