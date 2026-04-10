"""Request and sequence state management."""

from __future__ import annotations

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Optional

from my_vllm.sampling_params import SamplingParams


class SequenceStatus(IntEnum):
    """Status of a sequence/request in the system."""
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH = auto()
    FINISHED_EOS = auto()

    @staticmethod
    def is_finished(status: SequenceStatus) -> bool:
        return status in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_EOS,
        )

    def get_finished_reason(self) -> Optional[str]:
        finish_reasons = {
            SequenceStatus.FINISHED_STOPPED: "stop",
            SequenceStatus.FINISHED_LENGTH: "length",
            SequenceStatus.FINISHED_EOS: "stop",
        }
        return finish_reasons.get(self)


@dataclass
class Request:
    """Represents a single generation request in the engine.

    Tracks the full lifecycle of a request: its prompt tokens, generated
    output tokens, block table for KV cache, and current status.
    """

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    output_token_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    def get_len(self) -> int:
        """Total length: prompt + generated tokens."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        """All token IDs in order (prompt + output)."""
        return self.prompt_token_ids + self.output_token_ids

    def get_num_new_tokens(self) -> int:
        """Number of tokens not yet computed."""
        return self.get_len() - self.num_computed_tokens

    def get_last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int) -> None:
        """Append a generated token."""
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def is_prefill(self) -> bool:
        """True if the request has not computed any token yet (needs prefill)."""
        return self.num_computed_tokens < self.get_prompt_len()

    def get_num_uncomputed_prompt_tokens(self) -> int:
        """Number of prompt tokens not yet computed."""
        if self.num_computed_tokens >= self.get_prompt_len():
            return 0
        return self.get_prompt_len() - self.num_computed_tokens
