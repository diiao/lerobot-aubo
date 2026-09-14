#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auditable, offline-only policy prediction contract for Phase B.

``checkpoint_sha256`` is the digest of an immutable checkpoint manifest, not
an authorization token. A valid prediction still requires the independent
thin safety gate and a live policy-execution authorization before actuation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

from .contracts import (
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
    validate_action_vector,
)

POLICY_PREDICTION_SCHEMA_VERSION: Final = "PolicyPredictionV1"


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class PolicyPredictionV1:
    """One immutable action chunk plus its model and input provenance."""

    schema_version: str
    prediction_id: str
    policy_id: str
    checkpoint_ref: str
    checkpoint_sha256: str
    action_schema_version: str
    observation_id: str
    instruction_schema_version: str
    instruction_id: str
    instruction_text_sha256: str
    chunk_created_monotonic_s: float
    actions: tuple[tuple[object, ...], ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_PREDICTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported prediction schema {self.schema_version!r}; "
                f"expected {POLICY_PREDICTION_SCHEMA_VERSION!r}"
            )
        for name in ("prediction_id", "policy_id", "checkpoint_ref", "observation_id"):
            _require_nonempty_string(name, getattr(self, name))
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)

        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported action schema {self.action_schema_version!r}; "
                f"expected {ACTION_SCHEMA_VERSION!r}"
            )
        if self.instruction_schema_version != INSTRUCTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported instruction schema {self.instruction_schema_version!r}; "
                f"expected {INSTRUCTION_SCHEMA_VERSION!r}"
            )
        try:
            instruction = INSTRUCTION_SPECS[self.instruction_id]
        except KeyError as exc:
            raise ValueError(f"Unknown instruction_id {self.instruction_id!r}") from exc
        if instruction.is_control:
            raise ValueError("Control instructions cannot produce deployable policy predictions")
        if self.instruction_text_sha256 != instruction.text_sha256:
            raise ValueError("instruction_text_sha256 does not match the canonical instruction")

        timestamp = self.chunk_created_monotonic_s
        if isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("chunk_created_monotonic_s must be a finite non-negative number")
        if not self.actions:
            raise ValueError("actions must contain a non-empty action chunk")
        normalized = tuple(validate_action_vector(action) for action in self.actions)
        object.__setattr__(self, "actions", normalized)

        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not math.isfinite(self.confidence)
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise ValueError("confidence must be a finite value in [0, 1]")

    def to_manifest_record(self) -> dict[str, object]:
        """Return JSON-serializable provenance without granting authorization."""

        record = asdict(self)
        record["actions"] = [list(action) for action in self.actions]
        record["serialized_record_grants_live_authorization"] = False
        return record
