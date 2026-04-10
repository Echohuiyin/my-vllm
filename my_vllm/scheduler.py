"""Scheduler for continuous batching.

Implements FCFS (first-come-first-served) scheduling with preemption.
Each ``schedule()`` call decides which requests to run in the next step,
handling both prefill (new requests) and decode (ongoing generation).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from my_vllm.block_manager import BlockManager
from my_vllm.config import CacheConfig, SchedulerConfig
from my_vllm.sequence import Request, SequenceStatus


@dataclass
class SchedulerOutput:
    """The output of a scheduling step.

    Contains everything the model runner needs to execute a forward pass.
    """
    scheduled_requests: List[Request]
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0


class Scheduler:
    """FCFS scheduler with continuous batching and preemption.

    Manages three queues:
      - waiting:   requests that have not started prefill yet
      - running:   requests currently being decoded
      - preempted: requests that were preempted due to memory pressure
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        block_manager: BlockManager,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.block_manager = block_manager

        self.waiting: deque[Request] = deque()
        self.running: List[Request] = []
        self.preempted: deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        request.status = SequenceStatus.WAITING
        self.waiting.append(request)

    def abort_request(self, request_id: str) -> None:
        """Remove a request from all queues and free its blocks."""
        for queue in [self.waiting, self.running, self.preempted]:
            for req in list(queue):
                if req.request_id == request_id:
                    queue.remove(req)
                    self.block_manager.free(req)
                    return

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running or self.preempted)

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling step.

        1. Try to keep all running requests going (decode).
        2. If a running request cannot get a slot, preempt it.
        3. Promote waiting/preempted requests if there is capacity.
        """
        scheduled: List[Request] = []
        num_prefill_tokens = 0
        num_decode_tokens = 0
        token_budget = self.scheduler_config.max_num_batched_tokens

        preempted_in_this_step = False

        # --- Phase 1: schedule running (decode) requests ---
        kept_running: List[Request] = []
        for req in self.running:
            if not self.block_manager.can_append_slot(req):
                self._preempt(req)
                preempted_in_this_step = True
                continue

            decode_tokens = 1
            if num_decode_tokens + decode_tokens > token_budget:
                self._preempt(req)
                preempted_in_this_step = True
                continue

            self.block_manager.append_slot(req)
            kept_running.append(req)
            scheduled.append(req)
            num_decode_tokens += decode_tokens

        self.running = kept_running

        # --- Phase 2: schedule waiting / preempted requests (prefill) ---
        if not preempted_in_this_step:
            budget_remaining = token_budget - num_decode_tokens - num_prefill_tokens
            new_running = self._schedule_prefills(
                budget_remaining, scheduled, num_prefill_tokens
            )
            num_prefill_tokens += new_running

        return SchedulerOutput(
            scheduled_requests=scheduled,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
        )

    def _schedule_prefills(
        self,
        budget: int,
        scheduled: List[Request],
        current_prefill_tokens: int,
    ) -> int:
        """Try to schedule requests from preempted and waiting queues.

        Returns total new prefill tokens added.
        """
        added_tokens = 0

        for source in [self.preempted, self.waiting]:
            while source:
                if len(self.running) + len(scheduled) - len(self.running) >= \
                        self.scheduler_config.max_num_seqs:
                    break
                if len(scheduled) >= self.scheduler_config.max_num_seqs:
                    break

                req = source[0]
                num_tokens = req.get_num_new_tokens()
                if num_tokens == 0:
                    num_tokens = req.get_len()

                if added_tokens + num_tokens > budget:
                    break

                if not self.block_manager.can_allocate(req):
                    break

                source.popleft()
                self.block_manager.allocate(req)
                req.status = SequenceStatus.RUNNING
                self.running.append(req)
                scheduled.append(req)
                added_tokens += num_tokens

        return added_tokens

    def _preempt(self, request: Request) -> None:
        """Preempt a running request: free blocks and move to preempted queue."""
        request.status = SequenceStatus.WAITING
        request.num_computed_tokens = 0
        request.output_token_ids = []
        self.block_manager.free(request)
        self.preempted.appendleft(request)

    def update_from_output(
        self,
        scheduled_requests: List[Request],
        sampled_token_ids: List[int],
    ) -> List[Request]:
        """Update request state after model output and return finished requests.

        Each request gets its new token appended, then we check stop conditions.
        """
        finished: List[Request] = []

        for req, token_id in zip(scheduled_requests, sampled_token_ids):
            if req.is_prefill():
                req.num_computed_tokens = req.get_prompt_len()
            else:
                req.num_computed_tokens += 1

            req.append_token(token_id)

            if self._check_stop(req):
                finished.append(req)

        for req in finished:
            if req in self.running:
                self.running.remove(req)
            self.block_manager.free(req)

        return finished

    def _check_stop(self, request: Request) -> bool:
        """Check if a request should stop generating."""
        sp = request.sampling_params

        if request.get_last_token_id() in sp._all_stop_token_ids:
            request.status = SequenceStatus.FINISHED_EOS
            return True

        if sp.max_tokens is not None and \
                request.get_output_len() >= sp.max_tokens:
            request.status = SequenceStatus.FINISHED_LENGTH
            return True

        return False
