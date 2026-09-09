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

"""Immutable raw-capture manifest contract for Phase A evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

from .authorization_contract import EXECUTION_AUTHORIZATION_SCHEMA_VERSION
from .contracts import (
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_SCHEMA_VERSION,
    validate_instruction_fields,
)
from .observation_contract import OBSERVATION_SCHEMA_VERSION

RAW_CAPTURE_MANIFEST_SCHEMA_VERSION: Final = "RawCaptureManifestV1"
CAPTURE_PURPOSES: Final = frozenset(
    {"depth_go_no_go_static", "depth_go_no_go_motion", "teleop_demonstration"}
)
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must use only letters, digits, dot, underscore, and hyphen")
    return value


def _validate_utc_timestamp(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("created_at_utc must be a non-empty ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at_utc must be valid ISO 8601") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at_utc must include the UTC timezone")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class RawArtifactRecord:
    """Integrity metadata for one finalized raw artifact."""

    logical_name: str
    relative_path: str
    sha256: str
    byte_count: int
    sample_count: int

    def __post_init__(self) -> None:
        _validate_identifier("logical_name", self.logical_name)
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("relative_path must be a non-empty POSIX path")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.relative_path:
            raise ValueError(
                "relative_path must be normalized, relative, and remain inside the capture directory"
            )
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        for name in ("byte_count", "sample_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class RawCaptureManifestV1:
    """Versioned capture metadata; it does not grant permission to capture."""

    schema_version: str
    capture_id: str
    session_id: str
    capture_purpose: str
    created_at_utc: str
    authorization_evidence_ref: str
    scene_id: str
    calibration_version: str
    tool_version: str
    capture_profile_id: str
    go_no_go_threshold_version: str
    representative_frame_rule_version: str
    depth_requested: bool
    raw_data_immutable: bool
    finalized: bool
    sensor_identifiers: Mapping[str, str]
    capture_parameters: Mapping[str, object]
    artifacts: Sequence[RawArtifactRecord] = ()
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION
    instruction_schema_version: str = INSTRUCTION_SCHEMA_VERSION
    authorization_schema_version: str = EXECUTION_AUTHORIZATION_SCHEMA_VERSION
    instruction_id: str | None = None
    instruction_text: str | None = None
    instruction_language: str | None = None
    instruction_text_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RAW_CAPTURE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported capture manifest schema {self.schema_version!r}; "
                f"expected {RAW_CAPTURE_MANIFEST_SCHEMA_VERSION!r}"
            )
        expected_versions = {
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
            "authorization_schema_version": EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        }
        for name, expected in expected_versions.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r}")

        _validate_identifier("capture_id", self.capture_id)
        _validate_identifier("session_id", self.session_id)
        if self.capture_purpose not in CAPTURE_PURPOSES:
            raise ValueError(f"capture_purpose must be one of {sorted(CAPTURE_PURPOSES)}")
        _validate_utc_timestamp(self.created_at_utc)
        for name in (
            "authorization_evidence_ref",
            "scene_id",
            "calibration_version",
            "tool_version",
            "capture_profile_id",
            "go_no_go_threshold_version",
            "representative_frame_rule_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("depth_requested", "raw_data_immutable", "finalized"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if not self.raw_data_immutable:
            raise ValueError("RawCaptureManifestV1 requires raw_data_immutable=true")

        instruction_values = (
            self.instruction_id,
            self.instruction_text,
            self.instruction_language,
            self.instruction_text_sha256,
        )
        has_instruction = any(value is not None for value in instruction_values)
        if has_instruction and not all(value is not None for value in instruction_values):
            raise ValueError("All instruction fields must be present together")
        if self.capture_purpose == "teleop_demonstration" and not has_instruction:
            raise ValueError("teleop_demonstration requires canonical instruction fields")
        if has_instruction:
            validate_instruction_fields(
                schema_version=self.instruction_schema_version,
                instruction_id=self.instruction_id,
                instruction_text=self.instruction_text,
                instruction_language=self.instruction_language,
                instruction_text_sha256=self.instruction_text_sha256,
            )

        sensors = dict(self.sensor_identifiers)
        if not sensors or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in sensors.items()
        ):
            raise ValueError("sensor_identifiers must contain non-empty string keys and values")
        object.__setattr__(self, "sensor_identifiers", MappingProxyType(sensors))

        parameters = dict(self.capture_parameters)
        try:
            json.dumps(parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("capture_parameters must be finite JSON-compatible data") from exc
        object.__setattr__(self, "capture_parameters", _freeze_json(parameters))

        artifacts = tuple(self.artifacts)
        if not all(isinstance(artifact, RawArtifactRecord) for artifact in artifacts):
            raise ValueError("artifacts must contain RawArtifactRecord instances")
        logical_names = [artifact.logical_name for artifact in artifacts]
        relative_paths = [artifact.relative_path for artifact in artifacts]
        if len(set(logical_names)) != len(logical_names) or len(set(relative_paths)) != len(relative_paths):
            raise ValueError("artifact logical names and relative paths must be unique")
        if self.finalized and not artifacts:
            raise ValueError("A finalized capture manifest requires at least one artifact")
        if not self.finalized and artifacts:
            raise ValueError("Artifacts can only be attached to a finalized capture manifest")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable copy of the manifest."""

        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["sensor_identifiers"] = dict(self.sensor_identifiers)
        result["capture_parameters"] = _thaw_json(self.capture_parameters)
        result["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        return result


def reserve_capture_directory(root: Path, capture_id: str) -> Path:
    """Create one new capture directory and fail instead of overwriting it."""

    _validate_identifier("capture_id", capture_id)
    root = Path(root)
    if not root.is_dir():
        raise ValueError("Capture root must already exist and be a directory")
    capture_directory = root / capture_id
    capture_directory.mkdir(exist_ok=False)
    return capture_directory
