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

"""Load versioned camera records. CameraSetV2 is the current configuration;
CameraSetV1 remains readable as historical evidence. No device access occurs here."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final

CAMERA_SET_V1_SCHEMA_VERSION: Final = "CameraSetV1"

CAMERA_SET_V2_SCHEMA_VERSION: Final = "CameraSetV2"

CAMERA_SET_SCHEMA_VERSION: Final = CAMERA_SET_V1_SCHEMA_VERSION

GLOBAL_RGB: Final = "global_rgb"

GRASP_RGB: Final = "grasp_rgb"

WRIST_RGB: Final = "wrist_rgb"

ROBOT_STATE: Final = "robot_state"

FIXED_RGB_STREAMS: Final = (GLOBAL_RGB, GRASP_RGB)

THREE_RGB_STREAMS: Final = (*FIXED_RGB_STREAMS, WRIST_RGB)


class CameraSetDecision(str, Enum):
    BLOCKED = "blocked"
    TWO_RGB = "two_rgb"
    THREE_RGB = "three_rgb"


@dataclass(frozen=True)
class CameraSetV1Record:
    """Immutable audit record that makes C0 eligibility explicit."""

    schema_version: str
    frozen_at_utc: str
    decision: CameraSetDecision
    camera_streams: tuple[str, ...]
    physical_roles: Mapping[str, str]
    capture_profiles: Mapping[str, Mapping[str, object]]
    timestamp_methods: Mapping[str, str]
    evidence_sha256: Mapping[str, str]
    global_roi_evidence_ref: str
    grasp_roi_evidence_ref: str
    immutable: bool = True
    c0_eligible: bool = True

    def __post_init__(self) -> None:
        if self.schema_version not in {
            CAMERA_SET_V1_SCHEMA_VERSION,
            CAMERA_SET_V2_SCHEMA_VERSION,
        }:
            raise ValueError(
                "schema_version must be "
                f"{CAMERA_SET_V1_SCHEMA_VERSION!r} or {CAMERA_SET_V2_SCHEMA_VERSION!r}"
            )
        if self.decision is CameraSetDecision.TWO_RGB:
            expected_streams = FIXED_RGB_STREAMS
        elif self.decision is CameraSetDecision.THREE_RGB:
            expected_streams = THREE_RGB_STREAMS
        else:
            raise ValueError("A blocked camera-set decision cannot be frozen")
        if tuple(self.camera_streams) != expected_streams:
            raise ValueError(f"camera_streams must be {expected_streams} for {self.decision.value}")
        try:
            parsed = datetime.fromisoformat(self.frozen_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("frozen_at_utc must be valid ISO 8601") from exc
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("frozen_at_utc must include the UTC timezone")
        roles = dict(self.physical_roles)
        timestamp_methods = dict(self.timestamp_methods)
        profiles = dict(self.capture_profiles)
        for name, mapping in (
            ("physical_roles", roles),
            ("capture_profiles", profiles),
            ("timestamp_methods", timestamp_methods),
        ):
            if set(mapping) != set(expected_streams):
                raise ValueError(f"{name} must contain exactly the frozen camera streams")
        if any(not isinstance(value, str) or not value for value in (*roles.values(), *timestamp_methods.values())):
            raise ValueError("physical roles and timestamp methods must be non-empty strings")
        frozen_profiles: dict[str, Mapping[str, object]] = {}
        for name, profile in profiles.items():
            if not isinstance(profile, Mapping) or not profile:
                raise ValueError(f"capture profile for {name} must be a non-empty mapping")
            profile_copy = dict(profile)
            try:
                json.dumps(profile_copy, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"capture profile for {name} must be finite JSON data") from exc
            frozen_profiles[name] = MappingProxyType(profile_copy)
        object.__setattr__(self, "physical_roles", MappingProxyType(roles))
        object.__setattr__(self, "capture_profiles", MappingProxyType(frozen_profiles))
        object.__setattr__(self, "timestamp_methods", MappingProxyType(timestamp_methods))
        evidence = dict(self.evidence_sha256)
        if not evidence or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for key, value in evidence.items()
        ):
            raise ValueError("evidence_sha256 must contain lowercase SHA-256 digests")
        object.__setattr__(self, "evidence_sha256", MappingProxyType(evidence))
        for name in ("global_roi_evidence_ref", "grasp_roi_evidence_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not self.immutable or not self.c0_eligible:
            raise ValueError("A frozen camera set must be immutable and C0-eligible")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_utc": self.frozen_at_utc,
            "decision": self.decision.value,
            "camera_streams": list(self.camera_streams),
            "physical_roles": dict(self.physical_roles),
            "capture_profiles": {
                name: dict(profile) for name, profile in self.capture_profiles.items()
            },
            "timestamp_methods": dict(self.timestamp_methods),
            "evidence_sha256": dict(self.evidence_sha256),
            "global_roi_evidence_ref": self.global_roi_evidence_ref,
            "grasp_roi_evidence_ref": self.grasp_roi_evidence_ref,
            "immutable": self.immutable,
            "c0_eligible": self.c0_eligible,
        }


DEFAULT_CAMERA_SET_V1_PATH: Final = Path("configs/aubo_i10/CameraSetV1.json")

FROZEN_CAMERA_SET_V1_SHA256: Final = (
    "9d57ed90803d35dedcd33920bfa9dec9370a59f04a9e757e510d8ba44795fc6b"
)

DEFAULT_CAMERA_SET_V2_PATH: Final = Path("configs/aubo_i10/CameraSetV2.json")

FROZEN_CAMERA_SET_V2_SHA256: Final = (
    "20de7adfd6ed9734c3a4329d86ab7e0ad08df9c6758d878d3c1302e2ac6423b4"
)


def _load_camera_set(
    path: Path, *, expected_filename: str, expected_schema_version: str
) -> tuple[CameraSetV1Record, str]:
    if path.name != expected_filename:
        raise ValueError(f"The frozen record filename must be {expected_filename}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"{expected_filename} must contain a JSON object")
    record = CameraSetV1Record(
        schema_version=str(data["schema_version"]),
        frozen_at_utc=str(data["frozen_at_utc"]),
        decision=CameraSetDecision(data["decision"]),
        camera_streams=tuple(data["camera_streams"]),
        physical_roles=data["physical_roles"],
        capture_profiles=data["capture_profiles"],
        timestamp_methods=data["timestamp_methods"],
        evidence_sha256=data["evidence_sha256"],
        global_roi_evidence_ref=str(data["global_roi_evidence_ref"]),
        grasp_roi_evidence_ref=str(data["grasp_roi_evidence_ref"]),
        immutable=bool(data.get("immutable", True)),
        c0_eligible=bool(data.get("c0_eligible", True)),
    )
    if record.schema_version != expected_schema_version:
        raise ValueError(
            f"{expected_filename} schema_version must be {expected_schema_version!r}"
        )
    if record.decision is CameraSetDecision.TWO_RGB and WRIST_RGB in record.camera_streams:
        raise ValueError(f"A two-RGB {expected_schema_version} cannot contain wrist_rgb")
    return record, digest


def load_camera_set_v1(path: Path | None = None) -> tuple[CameraSetV1Record, str]:
    """Load the immutable CameraSetV1 record and return it with its SHA-256."""

    if path is None:
        path = Path(__file__).resolve().parents[3] / DEFAULT_CAMERA_SET_V1_PATH
    else:
        path = Path(path)
    return _load_camera_set(
        path,
        expected_filename="CameraSetV1.json",
        expected_schema_version=CAMERA_SET_V1_SCHEMA_VERSION,
    )


def load_camera_set_v2(path: Path | None = None) -> tuple[CameraSetV1Record, str]:
    """Load the corrected immutable CameraSetV2 record and return its SHA-256."""

    if path is None:
        path = Path(__file__).resolve().parents[3] / DEFAULT_CAMERA_SET_V2_PATH
    else:
        path = Path(path)
    record, digest = _load_camera_set(
        path,
        expected_filename="CameraSetV2.json",
        expected_schema_version=CAMERA_SET_V2_SCHEMA_VERSION,
    )
    if digest != FROZEN_CAMERA_SET_V2_SHA256:
        raise ValueError("CameraSetV2.json does not match the frozen SHA-256")
    return record, digest
