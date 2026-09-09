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

"""Auditable authorization fields for teleoperation and policy actions.

This module only validates records. A serialized record can never create live
permission to operate hardware; the runtime must separately hold a current,
same-process authorization latch for the active on-site session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

EXECUTION_AUTHORIZATION_SCHEMA_VERSION: Final = "ExecutionAuthorizationV1"
CONTROL_SOURCES: Final = frozenset({"teleop", "policy"})


@dataclass(frozen=True)
class ExecutionAuthorizationV1:
    """One action's authorization audit record, separate from safety checks."""

    schema_version: str
    session_id: str
    phase_id: str
    control_source: str
    teleop_session_authorized: bool
    policy_execution_authorized: bool
    action_gate_passed: bool
    observation_id: str
    action_chunk_id: str
    policy_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_AUTHORIZATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported authorization schema {self.schema_version!r}; "
                f"expected {EXECUTION_AUTHORIZATION_SCHEMA_VERSION!r}"
            )
        for name in ("session_id", "phase_id", "observation_id", "action_chunk_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.control_source not in CONTROL_SOURCES:
            raise ValueError(f"control_source must be one of {sorted(CONTROL_SOURCES)}")
        for name in (
            "teleop_session_authorized",
            "policy_execution_authorized",
            "action_gate_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")

        if self.control_source == "policy":
            if not isinstance(self.policy_id, str) or not self.policy_id:
                raise ValueError("policy control requires a non-empty policy_id")
        elif self.policy_id is not None:
            raise ValueError("teleop control must not claim a policy_id")

    @property
    def recorded_preconditions_satisfied(self) -> bool:
        """Check recorded fields without claiming live execution permission."""

        source_authorized = (
            self.teleop_session_authorized
            if self.control_source == "teleop"
            else self.policy_execution_authorized
        )
        return source_authorized and self.action_gate_passed

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable audit record."""

        record = asdict(self)
        record["serialized_record_grants_live_authorization"] = False
        return record


def build_authorization_manifest() -> dict[str, object]:
    """Describe field meanings for dataset and experiment manifests."""

    return {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        "control_sources": sorted(CONTROL_SOURCES),
        "fields": {
            "teleop_session_authorized": "explicit authorization for this on-site teleop session",
            "policy_execution_authorized": "explicit authorization for this policy and on-site stage",
            "action_gate_passed": "thin safety gate accepted this exact action chunk",
        },
        "recorded_preconditions": {
            "teleop": ["teleop_session_authorized", "action_gate_passed"],
            "policy": ["policy_execution_authorized", "action_gate_passed"],
        },
        "serialized_record_grants_live_authorization": False,
    }
