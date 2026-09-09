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

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Final, Mapping

INSTRUCTION_SCHEMA_VERSION: Final = "InstructionSchemaV1"
INSTRUCTION_LANGUAGE: Final = "en"
CONTROL_FIXED_ID: Final = "control_fixed"


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
