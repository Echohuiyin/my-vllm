"""Tests for block manager."""

import pytest

from my_vllm.sampling_params import SamplingParams
from my_vllm.sequence import Request
from my_vllm.block_manager import PhysicalBlock, BlockAllocator, BlockManager


def _make_request(request_id: str = "r0", prompt_len: int = 5) -> Request:
    return Request(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(),
    )


class TestPhysicalBlock:
    def test_new_block_is_free(self):
        b = PhysicalBlock(block_id=0)
        assert b.is_free()

    def test_allocated_block_not_free(self):
        b = PhysicalBlock(block_id=0, ref_count=1)
        assert not b.is_free()


class TestBlockAllocator:
    def test_initial_free_count(self):
        alloc = BlockAllocator(num_blocks=10)
        assert alloc.get_num_free_blocks() == 10

    def test_allocate(self):
        alloc = BlockAllocator(num_blocks=5)
        bid = alloc.allocate()
        assert 0 <= bid < 5
        assert alloc.get_num_free_blocks() == 4

    def test_allocate_all(self):
        alloc = BlockAllocator(num_blocks=3)
        ids = [alloc.allocate() for _ in range(3)]
        assert len(set(ids)) == 3
        assert alloc.get_num_free_blocks() == 0

    def test_allocate_out_of_blocks(self):
        alloc = BlockAllocator(num_blocks=1)
        alloc.allocate()
        with pytest.raises(RuntimeError, match="Out of free blocks"):
            alloc.allocate()

    def test_free(self):
        alloc = BlockAllocator(num_blocks=2)
        bid = alloc.allocate()
        alloc.free(bid)
        assert alloc.get_num_free_blocks() == 2

    def test_double_free_raises(self):
        alloc = BlockAllocator(num_blocks=2)
        bid = alloc.allocate()
        alloc.free(bid)
        with pytest.raises(ValueError, match="already free"):
            alloc.free(bid)

    def test_allocate_after_free(self):
        alloc = BlockAllocator(num_blocks=1)
        bid1 = alloc.allocate()
        alloc.free(bid1)
        bid2 = alloc.allocate()
        assert bid2 == bid1


class TestBlockManager:
    def test_can_allocate(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=8)
        assert bm.can_allocate(req)

    def test_cannot_allocate_insufficient_blocks(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=1)
        req = _make_request(prompt_len=8)
        assert not bm.can_allocate(req)

    def test_allocate(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=5)
        bt = bm.allocate(req)
        assert len(bt) == 2  # ceil(5/4) = 2
        assert bm.get_num_free_blocks() == 8

    def test_allocate_exact_block_boundary(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=8)
        bt = bm.allocate(req)
        assert len(bt) == 2

    def test_free(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=5)
        bm.allocate(req)
        bm.free(req)
        assert bm.get_num_free_blocks() == 10
        assert req.block_table == []

    def test_can_append_slot_has_room(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=3)
        bm.allocate(req)
        req.append_token(99)
        assert bm.can_append_slot(req)

    def test_append_slot_no_new_block_needed(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=3)
        bm.allocate(req)
        req.append_token(99)
        new_block = bm.append_slot(req)
        assert new_block is None
        assert bm.get_num_free_blocks() == 9

    def test_append_slot_new_block_needed(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=4)
        bm.allocate(req)
        req.append_token(99)
        new_block = bm.append_slot(req)
        assert new_block is not None
        assert bm.get_num_free_blocks() == 8

    def test_get_block_table(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=5)
        bt = bm.allocate(req)
        assert bm.get_block_table(req) == bt

    def test_get_block_table_unknown_request(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=5)
        assert bm.get_block_table(req) == []

    def test_get_slot_mapping(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = _make_request(prompt_len=5)
        bt = bm.allocate(req)
        sm = bm.get_slot_mapping(req)
        assert len(sm) == 5
        assert sm[0] == bt[0] * 4 + 0
        assert sm[3] == bt[0] * 4 + 3
        assert sm[4] == bt[1] * 4 + 0

    def test_multiple_requests(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        r1 = _make_request("r1", prompt_len=4)
        r2 = _make_request("r2", prompt_len=4)
        bm.allocate(r1)
        bm.allocate(r2)
        assert bm.get_num_free_blocks() == 8
        bm.free(r1)
        assert bm.get_num_free_blocks() == 9
        bm.free(r2)
        assert bm.get_num_free_blocks() == 10
