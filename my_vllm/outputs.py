"""Output data structures for generation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class CompletionOutput:
    """One completion result for a request."""

    index: int
    text: str
    token_ids: Sequence[int]
    finish_reason: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={list(self.token_ids)}, "
            f"finish_reason={self.finish_reason!r})"
        )


@dataclass
class RequestOutput:
    """The full output for one request."""

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: Optional[List[int]]
    outputs: List[CompletionOutput]
    finished: bool

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id!r}, "
            f"prompt={self.prompt!r}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished})"
        )
