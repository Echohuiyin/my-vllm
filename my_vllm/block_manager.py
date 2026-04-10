"""Block manager for paged KV cache management.

The block manager is responsible for allocating and freeing physical
blocks that hold KV cache data.  Each block has a fixed number of
slots (``block_size``), and sequences are assigned blocks on demand.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from my_vllm.sequence import Request
from my_vllm.utils import cdiv


@dataclass
class PhysicalBlock:
    """A physical block in GPU memory holding KV cache slots."""
    block_id: int
    ref_count: int = 0

    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockAllocator:
    """Manages a pool of physical blocks.

    Blocks are allocated from a free queue and returned when freed.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.blocks = [PhysicalBlock(block_id=i) for i in range(num_blocks)]
        self.free_queue: deque[int] = deque(range(num_blocks))

    def allocate(self) -> int:
        """Allocate a single block, returning its block_id."""
        if not self.free_queue:
            raise RuntimeError("Out of free blocks")
        block_id = self.free_queue.popleft()
        self.blocks[block_id].ref_count = 1
        return block_id

    def free(self, block_id: int) -> None:
        """Free a single block by its block_id."""
        block = self.blocks[block_id]
        if block.ref_count <= 0:
            raise ValueError(f"Block {block_id} is already free")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_queue.append(block_id)

    def get_num_free_blocks(self) -> int:
        return len(self.free_queue)


class BlockManager:
    """High-level block manager that maps requests to physical blocks.

    Provides allocation, slot appending, and freeing of blocks for
    individual requests.  The ``block_size`` determines how many KV
    cache slots each physical block holds.
    """

    def __init__(self, block_size: int, num_gpu_blocks: int) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.allocator = BlockAllocator(num_gpu_blocks)
        self._request_blocks: Dict[str, List[int]] = {}

    def can_allocate(self, request: Request) -> bool:
        """Check whether there are enough free blocks for the request's prompt."""
        num_required = cdiv(request.get_len(), self.block_size)
        num_already = len(self._request_blocks.get(request.request_id, []))
        return self.allocator.get_num_free_blocks() >= (num_required - num_already)

    def allocate(self, request: Request) -> List[int]:
        """Allocate blocks for a request's full current length.

        Returns the updated block table.
        """
        num_required = cdiv(request.get_len(), self.block_size)
        block_table = self._request_blocks.get(request.request_id, [])

        while len(block_table) < num_required:
            block_id = self.allocator.allocate()
            block_table.append(block_id)

        self._request_blocks[request.request_id] = block_table
        request.block_table = list(block_table)
        return list(block_table)

    def can_append_slot(self, request: Request) -> bool:
        """Check if we can append a new slot for one more token."""
        current_slots_used = request.get_len()
        current_blocks = len(self._request_blocks.get(request.request_id, []))
        slots_available = current_blocks * self.block_size
        if current_slots_used < slots_available:
            return True
        return self.allocator.get_num_free_blocks() >= 1

    def append_slot(self, request: Request) -> Optional[int]:
        """Allocate a new block if the current last block is full.

        Returns the new block_id if allocated, None if existing block has room.
        """
        block_table = self._request_blocks.get(request.request_id, [])
        slots_used = request.get_len()
        slots_available = len(block_table) * self.block_size

        if slots_used <= slots_available:
            return None

        block_id = self.allocator.allocate()
        block_table.append(block_id)
        self._request_blocks[request.request_id] = block_table
        request.block_table = list(block_table)
        return block_id

    def free(self, request: Request) -> None:
        """Free all blocks held by a request."""
        block_table = self._request_blocks.pop(request.request_id, [])
        for block_id in block_table:
            self.allocator.free(block_id)
        request.block_table = []

    def get_block_table(self, request: Request) -> List[int]:
        """Return the current block table for a request."""
        return list(self._request_blocks.get(request.request_id, []))

    def get_num_free_blocks(self) -> int:
        return self.allocator.get_num_free_blocks()

    def get_slot_mapping(self, request: Request) -> List[int]:
        """Compute the physical slot index for each logical token position.

        slot = block_table[pos // block_size] * block_size + pos % block_size
        """
        block_table = self._request_blocks.get(request.request_id, [])
        mapping = []
        for pos in range(request.get_len()):
            block_idx = pos // self.block_size
            block_offset = pos % self.block_size
            if block_idx < len(block_table):
                mapping.append(
                    block_table[block_idx] * self.block_size + block_offset
                )
        return mapping
