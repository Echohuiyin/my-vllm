"""Tokenizer wrapper around HuggingFace tokenizers."""

from __future__ import annotations

from typing import List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Tokenizer:
    """Wrapper around HuggingFace tokenizer."""

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        )

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
