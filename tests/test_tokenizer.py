"""Tests for the tokenizer wrapper.

These tests require a network connection to download tokenizer files
on the first run. They use a small model (Qwen/Qwen2-0.5B) for speed.
"""

import pytest

from my_vllm.tokenizer import Tokenizer

MODEL_NAME = "Qwen/Qwen2-0.5B"


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer(MODEL_NAME)


class TestTokenizer:
    def test_encode(self, tokenizer: Tokenizer):
        ids = tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode(self, tokenizer: Tokenizer):
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
        assert "Hello" in text
        assert "world" in text

    def test_roundtrip(self, tokenizer: Tokenizer):
        original = "The quick brown fox"
        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded.strip() == original

    def test_eos_token_id(self, tokenizer: Tokenizer):
        assert tokenizer.eos_token_id is not None
        assert isinstance(tokenizer.eos_token_id, int)

    def test_vocab_size(self, tokenizer: Tokenizer):
        assert tokenizer.vocab_size > 0
