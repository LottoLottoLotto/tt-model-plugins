# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.prefix_resume import REGISTRY

logger = init_tt_logger(__name__)


@dataclass
class _PendingOutputs:
    """Counts a request's decode tokens that are still in the pipeline.

    Async scheduling starts new decode steps before the tokens from earlier
    steps have come back. If a request is preempted, it has to redo its prefill
    from scratch, and we have to throw the tokens still in the pipeline away instead
    of adding them to the output.

    ``outstanding`` counts tokens that were scheduled but have not come back
    yet. ``stale`` counts how many of those we have decided to throw away.
    """

    PENDING_ATTR: ClassVar[str] = "_tt_pending_outputs"

    outstanding: int = 0
    stale: int = 0

    def record(self) -> None:
        """Count one decode step; its token comes back a few steps later.

        A step counts once even if it speculates several tokens, because it
        still sends back a single output.
        """
        self.outstanding += 1

    def discard_outstanding(self) -> None:
        """On preempt, mark every token now in the pipeline to be thrown away."""
        self.stale = self.outstanding

    def is_next_stale(self) -> bool:
        """Take the next returned token; True means throw it away.

        Tokens come back in the order they were scheduled, so the stale ones
        always arrive before any fresh token from after the preempt.
        """
        if not self.outstanding:
            return False
        self.outstanding -= 1
        if not self.stale:
            return False
        self.stale -= 1
        return True

    @classmethod
    def for_request(cls, request: Request) -> "_PendingOutputs":
        pending = getattr(request, cls.PENDING_ATTR, None)
        if pending is None:
            pending = cls()
            setattr(request, cls.PENDING_ATTR, pending)
        return pending


class TTSchedulingMode(Enum):
    DEFAULT = "default"
    DECODE_ONLY = "decode_only"
    PREFILL_ONLY = "prefill_only"

    @classmethod
    def from_prefill_intent(cls, prefill_intent: int) -> "TTSchedulingMode":
        if prefill_intent == 0:
            return cls.DECODE_ONLY
        if prefill_intent == 1:
            return cls.PREFILL_ONLY
        raise ValueError(f"Invalid TT scheduling intent: {prefill_intent}")


class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - Token-chunked prefill: a long prefill may be split across scheduler
      steps.  After a partial chunk the request moves to ``running`` with
      ``is_prefill_chunk=True``; subsequent steps schedule the next chunk
      until the prompt is fully computed.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap
    - under async_scheduling, preemption invalidates that request's
      scheduled-but-unreturned outputs (see ``_preempt_request``)

    Supports ``set_forced_mode`` for DP-gather coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when waiting is empty).
    - ``TTSchedulingMode.DEFAULT`` normally prefers prefill. Single-device
      batched Qwen instead runs a decode turn before each bounded prefill
      turn whenever both phases have work; neither phase can starve the other.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT
        if self.cache_config.enable_prefix_caching:
            # Prefix hits come from exact model-side resume snapshots, not raw block hashes.
            REGISTRY.attach(self.kv_cache_manager)
        self._qwen_batched_prefill = bool(
            (self.vllm_config.additional_config or {}).get("_tt_qwen_batched_prefill", False)
        )
        self._prefill_turn_due = False
        if self._qwen_batched_prefill:
            # Reuse the upstream pre-allocation alignment hook. Qwen's state
            # is model-owned, not a Mamba cache group, so its split rule below
            # deliberately does not inspect mamba_spec or cache mode.
            self.need_mamba_block_aligned_split = True

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if not self._qwen_batched_prefill:
            return super()._mamba_block_aligned_split(
                request, num_new_tokens,
                num_new_local_computed_tokens, num_external_computed_tokens,
            )
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        if start + num_new_tokens < request.num_tokens:
            # Every non-final chunk ends on a GDN scan boundary (128) and
            # therefore also a paged-KV boundary (64). A short remaining turn
            # budget simply leaves this request for the next prefill turn.
            return max(0, (start + num_new_tokens) // 128 * 128 - start)
        return num_new_tokens

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    def _has_pending_prefill(self) -> bool:
        """Whether any request needs prefill work.

        True when the waiting queue is non-empty or any running request
        is a partial-prefill continuation (``is_prefill_chunk`` set by the
        base scheduler after the previous step).
        """
        return bool(self.waiting) or any(r.is_prefill_chunk for r in self.running)

    def schedule(self) -> SchedulerOutput:
        has_pending_prefill = self._has_pending_prefill()
        has_running = any(not r.is_prefill_chunk for r in self.running)
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide waiting and partial-prefill continuations.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result)

        if self._qwen_batched_prefill:
            if not has_pending_prefill:
                self._prefill_turn_due = False
            elif has_running and not self._prefill_turn_due:
                result = self._schedule_decode_only()
                if result.total_num_scheduled_tokens:
                    self._prefill_turn_due = True
                    return self._finalize_scheduler_output(result)
            # At most one prefill turn may intervene between decode turns.
            self._prefill_turn_due = False

        # Default mode:
        # Prefer prefill whenever there is pending prefill work - either new
        # requests in the waiting queue or partial-prefill continuations in
        # the running list.
        if has_pending_prefill:
            prefill_result = self._schedule_prefill_only()
            # If prefill cannot make progress (e.g., KV pressure) but running
            # decode requests exist, fall back to decode-only so they can
            # advance and free capacity.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running:
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No pending prefill work: run decode-only naturally.
        result = super().schedule()
        return self._finalize_scheduler_output(result)

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        return scheduler_output

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests + partial-prefill continuations.

        Hides running decode requests (``is_prefill_chunk=False``) so the base
        scheduler's running loop only processes partial-prefill continuations and
        the waiting loop admits new prefills.  Adjusts ``max_num_running_reqs`` to
        account for hidden decode slots.
        """
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        saved_budget = self.max_num_scheduled_tokens
        if self._qwen_batched_prefill:
            self.max_num_scheduled_tokens = min(saved_budget, 2048)
        self.running = partial_prefills
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
            self.max_num_scheduled_tokens = saved_budget
        return result

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running decode requests.

        Hides the waiting queue **and** any partial-prefill continuations
        so the base scheduler only sees decode-phase requests.  Preempted
        requests are merged back into the original waiting queue afterwards.
        """
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]

        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            self.waiting = saved_waiting
            if partial_prefills:
                self.running.extend(partial_prefills)

        return result

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and drop any decode tokens still in the pipeline.

        The base class frees the request's KV cache and queues it to redo its
        prefill. Under async scheduling, the tokens it already scheduled are
        still on their way back; those were built on the now-freed cache, so we
        mark them to be thrown away and reset the placeholder count so the base
        scheduler treats the request as a fresh prefill.
        """
        super()._preempt_request(request, timestamp)

        if not self.scheduler_config.async_scheduling:
            return

        _PendingOutputs.for_request(request).discard_outstanding()
        request.num_output_placeholders = 0

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """After scheduling, count the decode tokens this step put in the pipeline.

        Only decode steps produce a token to track; prefill chunks do not. This
        running count is what lets a later preempt know how many in-flight
        tokens it has to throw away.
        """
        super()._update_after_schedule(scheduler_output)

        if self.scheduler_config.async_scheduling:
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests[req_id]
                if not request.is_prefill_chunk:
                    _PendingOutputs.for_request(request).record()

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """Add a returned token to the request, unless a preempt invalidated it.

        When a token comes back for a request that was preempted while the token
        was in the pipeline, adding it would corrupt the output, so we throw it
        away here instead of handing it to the base class.
        """
        if not self.scheduler_config.async_scheduling:
            return super()._update_request_with_output(request, new_token_ids)

        if _PendingOutputs.for_request(request).is_next_stale():
            request.discard_latest_async_tokens = False
            return [], False

        return super()._update_request_with_output(request, new_token_ids)
