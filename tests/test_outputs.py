"""Tests for output data structures."""

from my_vllm.outputs import CompletionOutput, RequestOutput


class TestCompletionOutput:
    def test_not_finished(self):
        out = CompletionOutput(index=0, text="hello", token_ids=[1, 2])
        assert not out.finished
        assert out.finish_reason is None

    def test_finished(self):
        out = CompletionOutput(
            index=0, text="hello", token_ids=[1, 2], finish_reason="stop"
        )
        assert out.finished

    def test_repr(self):
        out = CompletionOutput(index=0, text="hi", token_ids=[1])
        r = repr(out)
        assert "CompletionOutput" in r
        assert "hi" in r


class TestRequestOutput:
    def test_basic(self):
        comp = CompletionOutput(index=0, text="world", token_ids=[5, 6])
        out = RequestOutput(
            request_id="req-0",
            prompt="hello",
            prompt_token_ids=[1, 2, 3],
            outputs=[comp],
            finished=False,
        )
        assert out.request_id == "req-0"
        assert out.prompt == "hello"
        assert len(out.outputs) == 1
        assert not out.finished

    def test_finished(self):
        comp = CompletionOutput(
            index=0, text="done", token_ids=[7], finish_reason="length"
        )
        out = RequestOutput(
            request_id="req-1",
            prompt=None,
            prompt_token_ids=None,
            outputs=[comp],
            finished=True,
        )
        assert out.finished

    def test_repr(self):
        comp = CompletionOutput(index=0, text="x", token_ids=[1])
        out = RequestOutput(
            request_id="r",
            prompt="p",
            prompt_token_ids=[1],
            outputs=[comp],
            finished=False,
        )
        r = repr(out)
        assert "RequestOutput" in r
