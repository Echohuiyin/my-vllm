"""Sampling parameters for text generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

_SAMPLING_EPS = 1e-5


@dataclass
class SamplingParams:
    """Parameters for sampling from the model output."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: Optional[int] = 16
    stop_token_ids: Optional[List[int]] = None
    seed: Optional[int] = None
    ignore_eos: bool = False
    n: int = 1

    _all_stop_token_ids: Set[int] = field(
        default_factory=set, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.stop_token_ids is None:
            self.stop_token_ids = []
        self._all_stop_token_ids = set(self.stop_token_ids)
        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            self.top_p = 1.0
            self.top_k = 0

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}"
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1, got {self.max_tokens}"
            )

    @property
    def is_greedy(self) -> bool:
        return self.temperature < _SAMPLING_EPS

    def update_eos_token_id(self, eos_token_id: int) -> None:
        """Add the model's EOS token to stop token ids."""
        if not self.ignore_eos:
            self._all_stop_token_ids.add(eos_token_id)
