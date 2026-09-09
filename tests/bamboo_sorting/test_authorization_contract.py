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

from dataclasses import replace

import pytest

from lerobot.bamboo_sorting.authorization_contract import (
    EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
    ExecutionAuthorizationV1,
    build_authorization_manifest,
)


def _teleop_record(**changes: object) -> ExecutionAuthorizationV1:
    values = {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        "session_id": "session-dev-0001",
        "phase_id": "phase-c-teleop",
        "control_source": "teleop",
        "teleop_session_authorized": False,
        "policy_execution_authorized": False,
        "action_gate_passed": False,
        "observation_id": "observation-0001",
        "action_chunk_id": "action-chunk-0001",
    }
    values.update(changes)
    return ExecutionAuthorizationV1(**values)


def test_gate_pass_is_not_teleop_authorization() -> None:
    record = _teleop_record(action_gate_passed=True)

    assert not record.recorded_preconditions_satisfied
    assert not record.to_manifest_record()["serialized_record_grants_live_authorization"]


def test_teleop_requires_both_session_authorization_and_gate() -> None:
    record = _teleop_record(teleop_session_authorized=True, action_gate_passed=True)

    assert record.recorded_preconditions_satisfied


def test_policy_authorization_does_not_authorize_teleop_source() -> None:
    record = _teleop_record(policy_execution_authorized=True, action_gate_passed=True)

    assert not record.recorded_preconditions_satisfied


def test_policy_requires_model_identity_authorization_and_gate() -> None:
    record = _teleop_record(
        control_source="policy",
        policy_id="smolvla-rgb-dev-0001",
        policy_execution_authorized=True,
        action_gate_passed=True,
    )

    assert record.recorded_preconditions_satisfied


def test_teleop_and_policy_records_reject_ambiguous_identity() -> None:
    with pytest.raises(ValueError, match="requires a non-empty policy_id"):
        _teleop_record(control_source="policy")

    with pytest.raises(ValueError, match="must not claim a policy_id"):
        _teleop_record(policy_id="smolvla-rgb-dev-0001")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "ExecutionAuthorizationV2"}, "Unsupported authorization schema"),
        ({"session_id": ""}, "session_id must be a non-empty string"),
        ({"control_source": "automatic"}, "control_source must be one of"),
        ({"action_gate_passed": 1}, "action_gate_passed must be bool"),
    ],
)
def test_authorization_record_rejects_contract_drift(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_teleop_record(), **changes)


def test_authorization_manifest_separates_source_permissions_and_gate() -> None:
    manifest = build_authorization_manifest()

    assert manifest["recorded_preconditions"]["teleop"] == [
        "teleop_session_authorized",
        "action_gate_passed",
    ]
    assert manifest["recorded_preconditions"]["policy"] == [
        "policy_execution_authorized",
        "action_gate_passed",
    ]
    assert not manifest["serialized_record_grants_live_authorization"]
