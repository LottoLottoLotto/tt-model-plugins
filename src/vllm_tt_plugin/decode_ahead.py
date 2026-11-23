# SPDX-License-Identifier: Apache-2.0
"""Decode-ahead handshake between the TT scheduler and a model that launches the next decode step of its running
rows before the scheduler has run (LFM3: models/demos/blackhole/lfm3/tt/lfm3_vllm.py).  vllm_tt_plugin runs vLLM's
uni executor, so the scheduler and the model share this module's state.

- The model calls set_pending(True) when it launches a step ahead and set_pending(False) when it consumes or drops
  it.
- While a step is pending and a prefill could be admitted, TTScheduler.schedule runs the decode turn first (the
  launched step belongs to the running batch: a prefill turn in between would leave it unused while the rows' state
  has already advanced) and calls set_prefill_due(True); the model launches nothing ahead of the prefill turn that
  follows.  Every other schedule() clears prefill_due.
- lookahead(): KV slots the scheduler allocates beyond each running request's scheduled token (1 when the model
  declares model_capabilities["supports_decode_ahead"]), so a step launched ahead at position + 1 has its page.
"""

_pending = False
_prefill_due = False
_lookahead = 0


def set_pending(flag: bool) -> None:
    global _pending
    _pending = bool(flag)


def pending() -> bool:
    return _pending


def set_prefill_due(flag: bool) -> None:
    global _prefill_due
    _prefill_due = bool(flag)


def prefill_due() -> bool:
    return _prefill_due


def configure(lookahead: int) -> None:
    global _lookahead
    _lookahead = int(lookahead)


def lookahead() -> int:
    return _lookahead
