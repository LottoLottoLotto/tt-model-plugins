# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""The plugin's side of exact on-device sampling.

A model that samples on the device exactly (vLLM's sampler semantics, seeded by request) declares
``device_sampling_eligible(params, num_output_tokens)`` and ``device_row_sampling(params, seed)``; the model runner
then hands it one ``SamplingRequest`` per decode row (model_runner._exact_sampling_request).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingRequest:
    """A request routed to the device sampler: identity, parameters, live output ids, prompt ids (repetition)."""

    owner: object
    spec: object  # the model's device_row_sampling(params, seed)
    outputs: list
    prompt: object = None
