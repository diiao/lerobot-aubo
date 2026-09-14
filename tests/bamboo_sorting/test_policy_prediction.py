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

from lerobot.bamboo_sorting.contracts import (
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
)
from lerobot.bamboo_sorting.policy_prediction import (
    POLICY_PREDICTION_SCHEMA_VERSION,
    PolicyPredictionV1,
)


def _prediction(**changes: object) -> PolicyPredictionV1:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    values = {
        "schema_version": POLICY_PREDICTION_SCHEMA_VERSION,
        "prediction_id": "action-chunk-0001",
        "policy_id": "smolvla-camerasetv1-offline-phase-b",
        "checkpoint_ref": "checkpoints/001000/pretrained_model",
        "checkpoint_sha256": "a" * 64,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_id": "observation-0001",
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": instruction.instruction_id,
        "instruction_text_sha256": instruction.text_sha256,
        "chunk_created_monotonic_s": 9.99,
        "actions": (
            (0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.02, -0.5, 0.2, 0.0, 0.0, 0.0, 100.0),
        ),
        "confidence": None,
    }
    values.update(changes)
    return PolicyPredictionV1(**values)


def test_prediction_normalizes_actions_and_serializes_without_authorizing() -> None:
    prediction = _prediction(confidence=0.75)

    assert prediction.actions[1][-1] == 100.0
    record = prediction.to_manifest_record()
    assert record["actions"][1][-1] == 100.0
    assert record["confidence"] == 0.75
    assert record["serialized_record_grants_live_authorization"] is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "PolicyPredictionV2"}, "Unsupported prediction schema"),
        ({"prediction_id": ""}, "prediction_id must be a non-empty string"),
        ({"checkpoint_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"action_schema_version": "ActionSchemaV2"}, "Unsupported action schema"),
        ({"instruction_text_sha256": "b" * 64}, "canonical instruction"),
        ({"chunk_created_monotonic_s": float("nan")}, "finite non-negative"),
        ({"actions": ()}, "non-empty action chunk"),
        ({"confidence": 1.01}, r"\[0, 1\]"),
    ],
)
def test_prediction_rejects_contract_and_provenance_drift(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_prediction(), **changes)


def test_prediction_rejects_nonbinary_gripper_target() -> None:
    with pytest.raises(ValueError, match="ee.gripper_pos"):
        _prediction(actions=((0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 50.0),))
