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

import json

import pytest

from lerobot.bamboo_sorting.contracts import (
    ACTION_CONTROL_MODE,
    ACTION_FIELD_NAMES,
    ACTION_SCHEMA_VERSION,
    CONTROL_FIXED_ID,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
    build_action_manifest,
    build_instruction_manifest,
    validate_action_vector,
    validate_instruction_fields,
)


EXPECTED_TEXT_HASHES = {
    "pick_any_collection": "08f63b16667b9ce6463942838f712e2ceadea31f848898859f68a09487b99e6e",
    "pick_long_zone_a": "06861eef66336c72e3a988b54375a4ff4410c2e79d12e68df50c73cbbb7d0d19",
    "pick_long_zone_b": "58d54055f0e79a7c8ffc2cdc0dcae8a23ee49224df395f88f695d490f622837f",
    "pick_short_zone_a": "f0271d1546e0046912e0d02996911dabba05b14f7d47fed67abda6fe842e547d",
    "pick_short_zone_b": "b3a7bafecbb19404e4bbc7a8b5b7500e3df3280b2cf7b769426b181d6a8765ff",
    CONTROL_FIXED_ID: "2039b6b919216ce783b48d92de453cab7d9110a5dead12ed0be75f07a50daa8f",
}


def test_instruction_text_hashes_are_frozen() -> None:
    assert {key: spec.text_sha256 for key, spec in INSTRUCTION_SPECS.items()} == EXPECTED_TEXT_HASHES


def test_validate_canonical_demonstration_instruction() -> None:
    spec = INSTRUCTION_SPECS["pick_any_collection"]

    validated = validate_instruction_fields(
        schema_version=INSTRUCTION_SCHEMA_VERSION,
        instruction_id=spec.instruction_id,
        instruction_text=spec.text,
        instruction_language=INSTRUCTION_LANGUAGE,
        instruction_text_sha256=spec.text_sha256,
    )

    assert validated is spec


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "InstructionSchemaV2", "Unsupported instruction schema"),
        ("instruction_language", "zh", "Unsupported instruction language"),
        ("instruction_id", "unknown", "Unknown instruction_id"),
        ("instruction_text", "Pick one strip and place it in the collection area", "does not match"),
        ("instruction_text", " Pick one strip and place it in the collection area.", "does not match"),
        ("instruction_text", "pick one strip and place it in the collection area.", "does not match"),
        ("instruction_text_sha256", "0" * 64, "SHA-256"),
    ],
)
def test_validate_rejects_contract_drift(field: str, value: str, message: str) -> None:
    spec = INSTRUCTION_SPECS["pick_any_collection"]
    fields = {
        "schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": spec.instruction_id,
        "instruction_text": spec.text,
        "instruction_language": INSTRUCTION_LANGUAGE,
        "instruction_text_sha256": spec.text_sha256,
    }
    fields[field] = value

    with pytest.raises(ValueError, match=message):
        validate_instruction_fields(**fields)


def test_control_instruction_requires_explicit_opt_in() -> None:
    spec = INSTRUCTION_SPECS[CONTROL_FIXED_ID]
    fields = {
        "schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": spec.instruction_id,
        "instruction_text": spec.text,
        "instruction_language": INSTRUCTION_LANGUAGE,
        "instruction_text_sha256": spec.text_sha256,
    }

    with pytest.raises(ValueError, match="not valid for demonstrations"):
        validate_instruction_fields(**fields)

    assert validate_instruction_fields(**fields, allow_control=True) is spec


def test_manifest_is_serializable_and_complete() -> None:
    manifest = build_instruction_manifest()
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)

    assert manifest["schema_version"] == INSTRUCTION_SCHEMA_VERSION
    assert manifest["language"] == INSTRUCTION_LANGUAGE
    assert manifest["encoding"] == "UTF-8"
    assert manifest["hash_algorithm"] == "SHA-256"
    assert {
        item["instruction_id"]: item["instruction_text_sha256"] for item in manifest["instructions"]
    } == EXPECTED_TEXT_HASHES
    assert "InstructionSchemaV1" in serialized


def test_action_schema_matches_run06_order_and_units() -> None:
    manifest = build_action_manifest()

    assert ACTION_FIELD_NAMES == (
        "ee.j6_target",
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "ee.gripper_pos",
    )
    assert manifest["schema_version"] == ACTION_SCHEMA_VERSION
    assert manifest["runtime_control_mode"] == ACTION_CONTROL_MODE
    assert [field["unit"] for field in manifest["fields"]] == [
        "rad",
        "m",
        "m",
        "m",
        "rad",
        "rad",
        "rad",
        "legacy_binary",
    ]
    assert manifest["gripper_physical_mapping_verified"] is False


@pytest.mark.parametrize("gripper_value", [0.0, 100.0])
def test_validate_action_vector_accepts_both_legacy_binary_values(gripper_value: float) -> None:
    action = [-2.5, 0.1, -0.7, 0.2, 3.14, 0.0, 0.0, gripper_value]

    assert validate_action_vector(action) == tuple(action)


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ([0.0] * 7, "requires 8"),
        ([0.0] * 7 + [50.0], "must be one of"),
        ([float("nan")] + [0.0] * 7, "must be finite"),
        ([0.0, float("inf")] + [0.0] * 6, "must be finite"),
        ([False] + [0.0] * 7, "not bool"),
        (["invalid"] + [0.0] * 7, "must be numeric"),
    ],
)
def test_validate_action_vector_rejects_invalid_representation(
    action: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_action_vector(action)
