"""Tests for the scheduler."""

import pytest

from my_vllm.block_manager import BlockManager
from my_vllm.config import CacheConfig, SchedulerConfig
from my_vllm.sampling_params import SamplingParams
from my_vllm.scheduler import Scheduler, SchedulerOutput
from my_vllm.sequence import Request, SequenceStatus


def _make_scheduler(
    block_size: int = 4,
    num_gpu_blocks: int = 20,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 32,
) -> Scheduler:
    sched_cfg = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    cache_cfg = CacheConfig(block_size=block_size)
    bm = BlockManager(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    return Scheduler(sched_cfg, cache_cfg, bm)


def _make_request(
    request_id: str = "r0",
    prompt_len: int = 4,
    max_tokens: int = 10,
    stop_token_ids: list = None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids or [],
        ),
    )


class TestSchedulerOutput:
    def test_empty(self):
        so = SchedulerOutput(scheduled_requests=[])
        assert so.is_empty
        assert so.num_tokens == 0

    def test_not_empty(self):
        so = SchedulerOutput(
            scheduled_requests=[_make_request()],
            num_prefill_tokens=4,
            num_decode_tokens=0,
        )
        assert not so.is_empty
        assert so.num_tokens == 4


class TestSchedulerAddAndAbort:
    def test_add_request(self):
        sched = _make_scheduler()
        req = _make_request()
        sched.add_request(req)
        assert sched.has_unfinished()
        assert len(sched.waiting) == 1

    def test_abort_waiting(self):
        sched = _make_scheduler()
        req = _make_request()
        sched.add_request(req)
        sched.abort_request("r0")
        assert not sched.has_unfinished()

    def test_abort_nonexistent(self):
        sched = _make_scheduler()
        sched.abort_request("nonexistent")
        assert not sched.has_unfinished()


class TestSchedulerSchedule:
    def test_schedule_single_prefill(self):
        sched = _make_scheduler()
        req = _make_request(prompt_len=4)
        sched.add_request(req)

        out = sched.schedule()
        assert len(out.scheduled_requests) == 1
        assert out.num_prefill_tokens == 4
        assert out.num_decode_tokens == 0
        assert req.status == SequenceStatus.RUNNING

    def test_schedule_multiple_prefills(self):
        sched = _make_scheduler(max_num_batched_tokens=32)
        r1 = _make_request("r1", prompt_len=4)
        r2 = _make_request("r2", prompt_len=4)
        sched.add_request(r1)
        sched.add_request(r2)

        out = sched.schedule()
        assert len(out.scheduled_requests) == 2
        assert out.num_prefill_tokens == 8

    def test_schedule_decode(self):
        sched = _make_scheduler()
        req = _make_request(prompt_len=4)
        sched.add_request(req)

        sched.schedule()
        req.num_computed_tokens = 4
        req.append_token(99)

        out = sched.schedule()
        assert len(out.scheduled_requests) == 1
        assert out.num_decode_tokens == 1
        assert out.num_prefill_tokens == 0

    def test_max_num_seqs_limit(self):
        sched = _make_scheduler(max_num_seqs=2, max_num_batched_tokens=100)
        for i in range(5):
            sched.add_request(_make_request(f"r{i}", prompt_len=2))

        out = sched.schedule()
        assert len(out.scheduled_requests) == 2
        assert len(sched.waiting) == 3

    def test_token_budget_limit(self):
        sched = _make_scheduler(max_num_batched_tokens=6)
        r1 = _make_request("r1", prompt_len=4)
        r2 = _make_request("r2", prompt_len=4)
        sched.add_request(r1)
        sched.add_request(r2)

        out = sched.schedule()
        assert len(out.scheduled_requests) == 1
        assert out.num_prefill_tokens == 4


class TestSchedulerPreemption:
    def test_preempt_when_no_blocks(self):
        # block_size=4, num_gpu_blocks=2 => max 8 slots total.
        # A prompt of 4 uses 1 block; decoding can fill 1 more block (4 slots).
        # Once all 8 slots are used the next decode step must preempt.
        sched = _make_scheduler(
            block_size=4, num_gpu_blocks=2, max_num_batched_tokens=100
        )
        req = _make_request(prompt_len=4, max_tokens=20)
        sched.add_request(req)
        sched.schedule()  # prefill: allocates 1 block

        preempted = False
        for i in range(10):
            sched.update_from_output([req], [100 + i])
            out = sched.schedule()
            if req.status == SequenceStatus.WAITING:
                preempted = True
                break

        assert preempted
        assert len(sched.preempted) == 1
        assert req.output_token_ids == []
        assert req.num_computed_tokens == 0


class TestSchedulerUpdateFromOutput:
    def test_append_token(self):
        sched = _make_scheduler()
        req = _make_request(prompt_len=4, max_tokens=10)
        sched.add_request(req)
        sched.schedule()

        finished = sched.update_from_output([req], [42])
        assert req.output_token_ids == [42]
        assert req.num_computed_tokens == 4
        assert len(finished) == 0

    def test_stop_on_max_tokens(self):
        sched = _make_scheduler()
        req = _make_request(prompt_len=4, max_tokens=1)
        sched.add_request(req)
        sched.schedule()

        finished = sched.update_from_output([req], [42])
        assert len(finished) == 1
        assert req.status == SequenceStatus.FINISHED_LENGTH

    def test_stop_on_eos(self):
        sched = _make_scheduler()
        req = _make_request(prompt_len=4, max_tokens=10, stop_token_ids=[999])
        sched.add_request(req)
        sched.schedule()

        finished = sched.update_from_output([req], [999])
        assert len(finished) == 1
        assert req.status == SequenceStatus.FINISHED_EOS

    def test_finished_request_freed(self):
        sched = _make_scheduler()
        req = _make_request(prompt_len=4, max_tokens=1)
        sched.add_request(req)
        sched.schedule()
        free_before = sched.block_manager.get_num_free_blocks()

        sched.update_from_output([req], [42])
        assert sched.block_manager.get_num_free_blocks() > free_before
        assert req not in sched.running

    def test_decode_increments_computed(self):
        sched = _make_scheduler()
        req = _make_request(prompt_len=4, max_tokens=5)
        sched.add_request(req)
        sched.schedule()
        sched.update_from_output([req], [10])
        assert req.num_computed_tokens == 4

        sched.schedule()
        sched.update_from_output([req], [20])
        assert req.num_computed_tokens == 5

    def test_multiple_requests(self):
        sched = _make_scheduler(max_num_batched_tokens=100)
        r1 = _make_request("r1", prompt_len=4, max_tokens=1)
        r2 = _make_request("r2", prompt_len=4, max_tokens=10)
        sched.add_request(r1)
        sched.add_request(r2)
        sched.schedule()

        finished = sched.update_from_output([r1, r2], [42, 43])
        assert len(finished) == 1
        assert r1 in finished
        assert r2 not in finished
