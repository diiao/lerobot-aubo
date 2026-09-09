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

"""Versioned language contract for the AUBO depth-enhanced VLA study.

The exact English text is part of the experimental input. Changing whitespace,
capitalization, or punctuation changes the experiment and therefore requires a
new schema version.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Final, Mapping

INSTRUCTION_SCHEMA_VERSION: Final = "InstructionSchemaV1"
INSTRUCTION_LANGUAGE: Final = "en"
CONTROL_FIXED_ID: Final = "control_fixed"
ACTION_SCHEMA_VERSION: Final = "ActionSchemaV1"
ACTION_CONTROL_MODE: Final = "abs_j6yaw"
GRIPPER_BINARY_VALUES: Final = (0.0, 100.0)


@dataclass(frozen=True)
class InstructionSpec:
    """One immutable instruction in the experiment contract."""

    instruction_id: str
    text: str
    is_control: bool = False

    @property
    def text_sha256(self) -> str:
        """Return SHA-256 for the exact UTF-8 text, without an added newline."""

        return sha256(self.text.encode("utf-8")).hexdigest()


_INSTRUCTION_SPECS = {
    "pick_any_collection": InstructionSpec(
        instruction_id="pick_any_collection",
        text="Pick one strip and place it in the collection area.",
    ),
    "pick_long_zone_a": InstructionSpec(
        instruction_id="pick_long_zone_a",
        text="Place the long strip in zone A.",
    ),
    "pick_long_zone_b": InstructionSpec(
        instruction_id="pick_long_zone_b",
        text="Place the long strip in zone B.",
    ),
    "pick_short_zone_a": InstructionSpec(
        instruction_id="pick_short_zone_a",
        text="Place the short strip in zone A.",
    ),
    "pick_short_zone_b": InstructionSpec(
        instruction_id="pick_short_zone_b",
        text="Place the short strip in zone B.",
    ),
    CONTROL_FIXED_ID: InstructionSpec(
        instruction_id=CONTROL_FIXED_ID,
        text="Complete the task.",
        is_control=True,
    ),
}

INSTRUCTION_SPECS: Final[Mapping[str, InstructionSpec]] = MappingProxyType(_INSTRUCTION_SPECS)


@dataclass(frozen=True)
class ActionFieldSpec:
    """One ordered scalar in the absolute end-effector action vector."""

    name: str
    unit: str
    semantics: str


ACTION_FIELD_SPECS: Final = (
    ActionFieldSpec("ee.j6_target", "rad", "absolute AUBO J6 joint target"),
    ActionFieldSpec("ee.x", "m", "absolute base-frame TCP x"),
    ActionFieldSpec("ee.y", "m", "absolute base-frame TCP y"),
    ActionFieldSpec("ee.z", "m", "absolute base-frame TCP z"),
    ActionFieldSpec("ee.wx", "rad", "absolute base-frame TCP rotation-vector x"),
    ActionFieldSpec("ee.wy", "rad", "absolute base-frame TCP rotation-vector y"),
    ActionFieldSpec("ee.wz", "rad", "absolute base-frame TCP rotation-vector z"),
    ActionFieldSpec(
        "ee.gripper_pos",
        "legacy_binary",
        "legacy 0/100 command; physical open/close mapping is not yet verified",
    ),
)
ACTION_FIELD_NAMES: Final = tuple(field.name for field in ACTION_FIELD_SPECS)


def validate_instruction_fields(
    *,
    schema_version: str,
    instruction_id: str,
    instruction_text: str,
    instruction_language: str,
    instruction_text_sha256: str | None = None,
    allow_control: bool = False,
) -> InstructionSpec:
    """Validate instruction fields and return their canonical specification.

    Control instructions are rejected by default so ``control_fixed`` cannot be
    recorded as a normal demonstration instruction by accident.
    """

    if schema_version != INSTRUCTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported instruction schema {schema_version!r}; expected {INSTRUCTION_SCHEMA_VERSION!r}"
        )
    if instruction_language != INSTRUCTION_LANGUAGE:
        raise ValueError(
            f"Unsupported instruction language {instruction_language!r}; expected {INSTRUCTION_LANGUAGE!r}"
        )

    try:
        spec = INSTRUCTION_SPECS[instruction_id]
    except KeyError as exc:
        raise ValueError(f"Unknown instruction_id {instruction_id!r}") from exc

    if spec.is_control and not allow_control:
        raise ValueError(f"Control instruction {instruction_id!r} is not valid for demonstrations")
    if instruction_text != spec.text:
        raise ValueError(f"Text for {instruction_id!r} does not match {INSTRUCTION_SCHEMA_VERSION}")
    if instruction_text_sha256 is not None and instruction_text_sha256 != spec.text_sha256:
        raise ValueError(f"SHA-256 for {instruction_id!r} does not match its canonical UTF-8 text")

    return spec


def build_instruction_manifest() -> dict[str, object]:
    """Build a serializable manifest for dataset and experiment metadata."""

    return {
        "schema_version": INSTRUCTION_SCHEMA_VERSION,
        "language": INSTRUCTION_LANGUAGE,
        "encoding": "UTF-8",
        "hash_algorithm": "SHA-256",
        "instructions": [
            {
                "instruction_id": spec.instruction_id,
                "instruction_text": spec.text,
                "instruction_text_sha256": spec.text_sha256,
                "is_control": spec.is_control,
            }
            for spec in INSTRUCTION_SPECS.values()
        ],
    }


def validate_action_vector(action: Sequence[object]) -> tuple[float, ...]:
    """Validate an ordered ``ActionSchemaV1`` vector without applying safety limits.

    This contract checks representation only. Workspace, step-size, speed, age,
    and IK checks belong to the separate execution safety gate.
    """

    if isinstance(action, (str, bytes)) or len(action) != len(ACTION_FIELD_SPECS):
        raise ValueError(
            f"{ACTION_SCHEMA_VERSION} requires {len(ACTION_FIELD_SPECS)} ordered scalar values"
        )

    values: list[float] = []
    for field, raw_value in zip(ACTION_FIELD_SPECS, action, strict=True):
        if isinstance(raw_value, bool):
            raise ValueError(f"{field.name} must be numeric, not bool")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field.name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field.name} must be finite")
        values.append(value)

    if values[-1] not in GRIPPER_BINARY_VALUES:
        raise ValueError(
            f"ee.gripper_pos must be one of {GRIPPER_BINARY_VALUES}; physical open/close mapping is unverified"
        )

    return tuple(values)


def build_action_manifest() -> dict[str, object]:
    """Build the machine-readable action representation contract."""

    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "representation": "absolute_end_effector_with_absolute_j6",
        "runtime_control_mode": ACTION_CONTROL_MODE,
        "fields": [
            {"name": field.name, "unit": field.unit, "semantics": field.semantics}
            for field in ACTION_FIELD_SPECS
        ],
        "gripper_binary_values": list(GRIPPER_BINARY_VALUES),
        "gripper_physical_mapping_verified": False,
    }
