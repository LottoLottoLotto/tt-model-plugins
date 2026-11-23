# SPDX-License-Identifier: Apache-2.0
"""Exact prefill resume for models whose prefill carries recurrent state across fixed chunks.

vLLM's block-hash prefix cache is not enough for such models: the recurrent state at the hit
position is not in the paged KV cache, and the chunked kernels are only bitwise reproducible
when a resumed prefill starts on the same chunk boundary a full prefill would cross. So:

* The model, while prefilling a request with its normal fixed-chunk schedule, publishes a
  snapshot at its last completed full-chunk boundary ``S`` (``put``): the model-owned host
  payload (recurrent state, ...) plus the physical KV block ids covering ``[0, S)`` and the
  allocation epoch of each block. A block's content is only ever written while it is freshly
  allocated to a request (prefix blocks handed out by this module are read-only), so an
  unchanged epoch proves the block still holds exactly what that full prefill wrote.
* ``attach`` wraps the scheduler's KV cache manager: the prefix "hit" of a new request is the
  longest snapshot whose token prefix matches and whose blocks are untouched, and the request
  is given exactly those blocks with ``num_computed_tokens = S``. Everything from ``S`` on is
  freshly prefilled into newly allocated blocks, with the same chunk schedule as a full prefill.
* The model takes the snapshot back by token prefix (``take``) when the prefill starts at ``S``.

Scheduler and model share this module-level registry: TT runs the worker in the engine-core
process (uni executor).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def token_key(tokens, length: int) -> bytes:
    """Digest of the first ``length`` token ids (int32, host order)."""
    array = np.asarray(tokens[:length], dtype=np.int32)
    if array.shape != (length,):
        raise ValueError(f"token prefix has shape {array.shape}, expected ({length},)")
    return hashlib.sha256(array.tobytes()).digest()


@dataclass
class _Snapshot:
    length: int
    block_ids: tuple[int, ...]
    epochs: tuple[int, ...]
    payload: Any
    nbytes: int


class PrefixResume:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.align: int | None = None  # model chunk size; None = model cannot resume
        self.block_size: int | None = None
        self.max_snapshots = 0
        self.enabled = False  # set by the scheduler when prefix caching is on
        self._snapshots: OrderedDict[bytes, _Snapshot] = OrderedDict()
        self._lengths: dict[int, int] = {}  # length -> number of snapshots with it
        self._pending: dict[str, bytes] = {}  # request id -> key handed to the scheduler
        self._epoch: dict[int, int] = {}
        self.fork_hints = False  # scheduler asks for snapshots where cached prompts fork
        self._wanted: OrderedDict[bytes, int] = OrderedDict()  # prefix key -> boundary
        self.on_drop = None  # model callback(payload) when a snapshot leaves the registry
        self._private: set[str] = set()  # requests whose state must not be shared (cache_salt, LoRA, multimodal)
        self.stats = {"queries": 0, "hits": 0, "hit_tokens": 0, "query_tokens": 0, "puts": 0,
                      "evicted": 0, "stale": 0}

    # ---- model side -------------------------------------------------------------------------
    def configure(self, align: int, block_size: int, max_snapshots: int, fork_hints: bool = False,
                  on_drop=None) -> None:
        """``on_drop(payload)``: called (under the registry lock; it must not call back into the
        registry) whenever a snapshot leaves the registry: evicted, stale, replaced. A payload with a
        ``valid()`` method is checked at every lookup (False = stale: the model's own state behind it
        was overwritten). A new configuration drops every snapshot of the previous one (through the
        previous on_drop)."""
        if align <= 0 or block_size <= 0 or align % block_size:
            raise ValueError("resume alignment must be a positive multiple of the KV block size")
        with self._lock:
            for key in list(self._snapshots):
                self._drop(key)
            self._pending.clear()
            self._wanted.clear()
            self.align, self.block_size, self.max_snapshots = align, block_size, max(0, max_snapshots)
            self.fork_hints = fork_hints
            self.on_drop = on_drop
        logger.info("Prefix resume: chunk %d, block %d, max snapshots %d", align, block_size, max_snapshots)

    @property
    def active(self) -> bool:
        return self.enabled and self.align is not None and self.max_snapshots > 0

    def put(self, tokens, length: int, block_ids, payload: Any, nbytes: int) -> None:
        """Publish the state after a full prefill of ``tokens[:length]`` into ``block_ids``."""
        if not self.active:
            return
        if length <= 0 or length % self.align:
            raise ValueError(f"snapshot length {length} is not a positive multiple of {self.align}")
        nblocks = length // self.block_size
        block_ids = tuple(int(b) for b in block_ids[:nblocks])
        if len(block_ids) != nblocks or any(b <= 0 for b in block_ids):
            raise ValueError("snapshot needs one real (non-null) KV block per prefix block")
        key = token_key(tokens, length)
        with self._lock:
            if key in self._snapshots:
                self._drop(key)
            self._snapshots[key] = _Snapshot(
                length, block_ids, tuple(self._epoch.get(b, 0) for b in block_ids), payload, int(nbytes)
            )
            self._lengths[length] = self._lengths.get(length, 0) + 1
            self.stats["puts"] += 1
            pinned = set(self._pending.values())
            for victim in [k for k in self._snapshots if k not in pinned][: max(0, len(self._snapshots) - self.max_snapshots)]:
                self._drop(victim)
                self.stats["evicted"] += 1

    def evict_one(self) -> bool:
        """Drop the least recently used snapshot no scheduled request is waiting for (a model whose
        snapshot storage is full makes room); False when every snapshot is pinned."""
        with self._lock:
            pinned = set(self._pending.values())
            for key in self._snapshots:
                if key not in pinned:
                    self._drop(key)
                    self.stats["evicted"] += 1
                    return True
            return False

    def release(self, request_id: str) -> None:
        """The request is gone (finished, aborted): it no longer pins the snapshot it was found at."""
        with self._lock:
            self._pending.pop(request_id, None)
            self._private.discard(request_id)

    def shareable(self, request_id: str | None) -> bool:
        """Whether the state of this request's prefill may be published for other requests (False for cache_salt,
        LoRA and multimodal requests: they neither resume nor publish)."""
        return request_id not in self._private

    def _drop(self, key: bytes) -> None:
        snapshot = self._snapshots.pop(key)
        self._forget_length(snapshot.length)
        if self.on_drop is not None:
            self.on_drop(snapshot.payload)

    def take(self, tokens, length: int, request_id: str | None = None) -> Any:
        """Payload of the snapshot the scheduler resumed this prefill from. With ``request_id``
        only that request is unpinned: concurrent requests may resume from one snapshot."""
        key = token_key(tokens, length)
        with self._lock:
            snapshot = self._snapshots.get(key)
            if request_id is not None:
                if self._pending.pop(request_id, None) != key:
                    raise RuntimeError(f"request {request_id} was not scheduled to resume at {length}")
            else:
                for pinned in [r for r, k in self._pending.items() if k == key]:
                    del self._pending[pinned]
            if snapshot is None or snapshot.length != length:
                raise RuntimeError(f"no prefix-resume snapshot for the scheduled {length}-token prefix")
            self._snapshots.move_to_end(key)
            return snapshot.payload

    def resuming(self, request_id: str) -> bool:
        """Whether the scheduler resumed this request from a snapshot (not yet taken)."""
        with self._lock:
            return request_id in self._pending

    def has(self, tokens, length: int) -> bool:
        if length not in self._lengths:
            return False
        key = token_key(tokens, length)
        with self._lock:
            return key in self._snapshots

    def wanted(self, tokens, length: int) -> bool:
        """Consume a scheduler request to snapshot ``tokens[:length]`` (a fork boundary)."""
        with self._lock:
            if length not in self._wanted.values():
                return False
            return self._wanted.pop(token_key(tokens, length), None) is not None

    # ---- scheduler side ---------------------------------------------------------------------
    def want(self, tokens, length: int) -> None:
        """Ask the model to snapshot ``tokens[:length]`` when a prefill crosses that boundary."""
        key = token_key(tokens, length)
        with self._lock:
            if key in self._snapshots:
                return
            self._wanted[key] = length
            self._wanted.move_to_end(key)
            while len(self._wanted) > 4 * max(1, self.max_snapshots):
                self._wanted.popitem(last=False)

    def on_new_blocks(self, blocks) -> None:
        epoch = self._epoch
        for block in blocks:
            epoch[block.block_id] = epoch.get(block.block_id, 0) + 1

    def find(self, request_id: str, tokens, max_length: int) -> tuple[int, tuple[int, ...]]:
        """Longest valid snapshot prefix of ``tokens`` no longer than ``max_length``."""
        with self._lock:
            self.stats["queries"] += 1
            self.stats["query_tokens"] += max_length + 1
            self._pending.pop(request_id, None)
            lengths = sorted((n for n in self._lengths if n <= max_length), reverse=True)
            if not lengths:
                return 0, ()
            array = np.asarray(tokens[: lengths[0]], dtype=np.int32)
            for length in lengths:
                key = hashlib.sha256(array[:length].tobytes()).digest()
                snapshot = self._snapshots.get(key)
                if snapshot is None:
                    continue
                valid = getattr(snapshot.payload, "valid", None)
                if any(self._epoch.get(b, 0) != e for b, e in zip(snapshot.block_ids, snapshot.epochs)) or (
                    valid is not None and not valid()
                ):
                    # A prefix block was reallocated and rewritten (or the model's own state behind
                    # the snapshot was): the snapshot can never be used.
                    self._drop(key)
                    self.stats["stale"] += 1
                    continue
                self._snapshots.move_to_end(key)
                self._pending[request_id] = key
                self.stats["hits"] += 1
                self.stats["hit_tokens"] += length
                return length, snapshot.block_ids
            return 0, ()

    def _forget_length(self, length: int) -> None:
        count = self._lengths.get(length, 0) - 1
        if count > 0:
            self._lengths[length] = count
        else:
            self._lengths.pop(length, None)

    def attach(self, kv_cache_manager) -> None:
        """Route the scheduler's prefix-cache lookups through this registry."""
        groups = kv_cache_manager.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError("prefix resume supports a single KV cache group")
        pool = kv_cache_manager.block_pool
        get_new_blocks = pool.get_new_blocks

        def tracked_get_new_blocks(num_blocks):
            blocks = get_new_blocks(num_blocks)
            self.on_new_blocks(blocks)
            return blocks

        pool.get_new_blocks = tracked_get_new_blocks
        vllm_get_computed_blocks = kv_cache_manager.get_computed_blocks

        def get_computed_blocks(request):
            if self.align is None:
                # The model never configured resume: plain vLLM prefix caching semantics.
                return vllm_get_computed_blocks(request)
            # cache_salt / LoRA / multimodal requests neither resume nor publish (their state is not another
            # request's); prompt-logprobs requests (skip_reading_prefix_cache) prefill fully but may publish
            private = bool(
                request.mm_features or request.lora_request is not None or getattr(request, "cache_salt", None)
            )
            with self._lock:
                if private:
                    self._private.add(request.request_id)
                else:
                    self._private.discard(request.request_id)
            if private or request.skip_reading_prefix_cache:
                length, block_ids = 0, ()
            else:
                length, block_ids = self.find(request.request_id, request.all_token_ids, request.num_tokens - 1)
            if kv_cache_manager.log_stats and kv_cache_manager.prefix_cache_stats is not None:
                kv_cache_manager.prefix_cache_stats.record(
                    num_tokens=request.num_tokens, num_hits=length, preempted=request.num_preemptions > 0
                )
            if self.fork_hints and self.max_snapshots and not private:
                # vLLM's block hashes only locate where this prompt leaves another cached prompt
                # (shared system prompt / document); their KV is never reused. The prefill
                # publishes a snapshot at the last chunk boundary inside the shared part, so
                # later requests with that prefix resume there.
                shared = kv_cache_manager.coordinator.find_longest_cache_hit(
                    request.block_hashes, request.num_tokens - 1
                )[1]
                fork = shared // self.align * self.align
                if fork > length:
                    self.want(request.all_token_ids, fork)
            if not length:
                return kv_cache_manager.empty_kv_cache_blocks, 0
            logger.info("Prefix resume: request %s resumes at %d of %d tokens", request.request_id, length,
                        request.num_tokens)
            return kv_cache_manager.create_kv_cache_blocks(([pool.blocks[b] for b in block_ids],)), length

        kv_cache_manager.get_computed_blocks = get_computed_blocks
        self.enabled = True
        logger.info("Prefix resume attached to the KV cache manager")


REGISTRY = PrefixResume()
