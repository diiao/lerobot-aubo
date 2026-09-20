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

"""Frozen Phase A2 RGB gates plus immutable CameraSetV1/V2 records.

This module is deliberately hardware-free. Hardware runners must reduce raw
samples to the metrics below before applying the pre-registered thresholds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final

RGB_GATE_SCHEMA_VERSION: Final = "RgbGateV1"
CAMERA_SET_V1_SCHEMA_VERSION: Final = "CameraSetV1"
CAMERA_SET_V2_SCHEMA_VERSION: Final = "CameraSetV2"
# Historical compatibility name. New formal capture code must use
# CURRENT_CAMERA_SET_SCHEMA_VERSION explicitly.
CAMERA_SET_SCHEMA_VERSION: Final = CAMERA_SET_V1_SCHEMA_VERSION
CURRENT_CAMERA_SET_SCHEMA_VERSION: Final = CAMERA_SET_V2_SCHEMA_VERSION
RGB_GATE_MIN_EFFECTIVE_FPS: Final = 9.0
RGB_GATE_MAX_DROP_RATE: Final = 0.01
RGB_GATE_MAX_DUPLICATE_RATE: Final = 0.01
RGB_GATE_MAX_FRAME_AGE_P95_MS: Final = 100.0
RGB_GATE_MAX_ALIGNMENT_P95_MS: Final = 50.0
RGB_GATE_MIN_CONCURRENT_DURATION_S: Final = 60.0

GLOBAL_RGB: Final = "global_rgb"
GRASP_RGB: Final = "grasp_rgb"
WRIST_RGB: Final = "wrist_rgb"
ROBOT_STATE: Final = "robot_state"
FIXED_RGB_STREAMS: Final = (GLOBAL_RGB, GRASP_RGB)
THREE_RGB_STREAMS: Final = (*FIXED_RGB_STREAMS, WRIST_RGB)


class RGBGateDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class CameraSetDecision(str, Enum):
    BLOCKED = "blocked"
    TWO_RGB = "two_rgb"
    THREE_RGB = "three_rgb"


def _finite_non_negative(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class RGBStreamMetrics:
    """Metrics for one RGB stream in either the isolated or concurrent run."""

    stream_name: str
    duration_s: float
    sample_count: int
    unique_sample_count: int
    effective_unique_fps: float
    drop_rate_fraction: float
    duplicate_rate_fraction: float
    frame_age_p95_ms: float
    capture_error_count: int
    timestamp_method: str
    timestamp_traceable: bool
    timestamp_monotonic: bool
    frame_identity_method: str

    def __post_init__(self) -> None:
        if self.stream_name not in THREE_RGB_STREAMS:
            raise ValueError(f"stream_name must be one of {THREE_RGB_STREAMS}")
        for name in ("duration_s", "effective_unique_fps", "frame_age_p95_ms"):
            _finite_non_negative(name, getattr(self, name))
        for name in ("drop_rate_fraction", "duplicate_rate_fraction"):
            value = getattr(self, name)
            _finite_non_negative(name, value)
            if value > 1:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in ("sample_count", "unique_sample_count", "capture_error_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.unique_sample_count > self.sample_count:
            raise ValueError("unique_sample_count cannot exceed sample_count")
        for name in ("timestamp_method", "frame_identity_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("timestamp_traceable", "timestamp_monotonic"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")


@dataclass(frozen=True)
class RGBStreamGateResult:
    schema_version: str
    stream_name: str
    decision: RGBGateDecision
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["decision"] = self.decision.value
        return result


@dataclass(frozen=True)
class ConcurrentRGBMetrics:
    """Timing evidence from a simultaneous RGB and robot-state run.

    Pair keys are canonical ``left|right`` strings produced by :func:`pair_key`.
    All values must already share the host-monotonic clock domain.
    """

    duration_s: float
    policy_tick_count: int
    stream_metrics: Mapping[str, RGBStreamMetrics]
    pairwise_skew_p95_ms: Mapping[str, float]
    robot_state_sample_count: int
    robot_state_error_count: int
    robot_state_timestamp_traceable: bool
    robot_state_timestamp_monotonic: bool

    def __post_init__(self) -> None:
        _finite_non_negative("duration_s", self.duration_s)
        for name in ("policy_tick_count", "robot_state_sample_count", "robot_state_error_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        streams = dict(self.stream_metrics)
        if not streams or any(name != metrics.stream_name for name, metrics in streams.items()):
            raise ValueError("stream_metrics keys must match their stream_name")
        if any(name not in THREE_RGB_STREAMS for name in streams):
            raise ValueError("stream_metrics contains an unsupported RGB stream")
        object.__setattr__(self, "stream_metrics", MappingProxyType(streams))
        skews = dict(self.pairwise_skew_p95_ms)
        for key, value in skews.items():
            if not isinstance(key, str) or "|" not in key:
                raise ValueError("pairwise skew keys must be canonical pair strings")
            _finite_non_negative(f"pairwise_skew_p95_ms[{key}]", value)
        object.__setattr__(self, "pairwise_skew_p95_ms", MappingProxyType(skews))
        for name in ("robot_state_timestamp_traceable", "robot_state_timestamp_monotonic"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")


@dataclass(frozen=True)
class ConcurrentRGBGateResult:
    schema_version: str
    required_rgb_streams: tuple[str, ...]
    decision: RGBGateDecision
    stream_results: Mapping[str, RGBStreamGateResult]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_results", MappingProxyType(dict(self.stream_results)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "required_rgb_streams": list(self.required_rgb_streams),
            "decision": self.decision.value,
            "stream_results": {
                name: result.to_dict() for name, result in self.stream_results.items()
            },
            "reasons": list(self.reasons),
        }


def pair_key(left: str, right: str) -> str:
    if left == right:
        raise ValueError("A timing pair requires two different streams")
    return "|".join(sorted((left, right)))


def evaluate_rgb_stream(metrics: RGBStreamMetrics) -> RGBStreamGateResult:
    reasons: list[str] = []
    if metrics.effective_unique_fps < RGB_GATE_MIN_EFFECTIVE_FPS:
        reasons.append(
            f"effective unique FPS {metrics.effective_unique_fps:.3f} < {RGB_GATE_MIN_EFFECTIVE_FPS:g}"
        )
    if metrics.drop_rate_fraction >= RGB_GATE_MAX_DROP_RATE:
        reasons.append(f"drop rate {metrics.drop_rate_fraction:.3%} is not below 1%")
    if metrics.duplicate_rate_fraction >= RGB_GATE_MAX_DUPLICATE_RATE:
        reasons.append(f"duplicate rate {metrics.duplicate_rate_fraction:.3%} is not below 1%")
    if metrics.frame_age_p95_ms > RGB_GATE_MAX_FRAME_AGE_P95_MS:
        reasons.append(f"frame age p95 {metrics.frame_age_p95_ms:.3f} ms > 100 ms")
    if metrics.capture_error_count:
        reasons.append(f"capture errors observed: {metrics.capture_error_count}")
    if not metrics.timestamp_traceable:
        reasons.append("timestamp origin or host-receive stamping point is not traceable")
    if not metrics.timestamp_monotonic:
        reasons.append("timestamps are not monotonic")
    decision = RGBGateDecision.FAIL if reasons else RGBGateDecision.PASS
    if not reasons:
        reasons.append("all frozen per-stream RGB criteria passed")
    return RGBStreamGateResult(
        schema_version=RGB_GATE_SCHEMA_VERSION,
        stream_name=metrics.stream_name,
        decision=decision,
        reasons=tuple(reasons),
    )


def evaluate_isolated_wrist_rgb(metrics: RGBStreamMetrics) -> RGBStreamGateResult:
    if metrics.stream_name != WRIST_RGB:
        raise ValueError("The isolated Mech-Eye gate requires wrist_rgb metrics")
    return evaluate_rgb_stream(metrics)


def evaluate_concurrent_rgb_gate(
    metrics: ConcurrentRGBMetrics,
    *,
    required_rgb_streams: tuple[str, ...],
) -> ConcurrentRGBGateResult:
    if required_rgb_streams not in (FIXED_RGB_STREAMS, THREE_RGB_STREAMS):
        raise ValueError("required_rgb_streams must be the frozen two- or three-camera candidate tuple")

    reasons: list[str] = []
    if metrics.duration_s < RGB_GATE_MIN_CONCURRENT_DURATION_S:
        reasons.append(
            f"concurrent duration {metrics.duration_s:.3f} s < {RGB_GATE_MIN_CONCURRENT_DURATION_S:g} s"
        )
    missing = [name for name in required_rgb_streams if name not in metrics.stream_metrics]
    if missing:
        reasons.append(f"missing required RGB streams: {missing}")

    stream_results = {
        name: evaluate_rgb_stream(metrics.stream_metrics[name])
        for name in required_rgb_streams
        if name in metrics.stream_metrics
    }
    for name, result in stream_results.items():
        if result.decision is RGBGateDecision.FAIL:
            reasons.append(f"{name} failed its per-stream gate")

    required_pairs = [
        pair_key(required_rgb_streams[index], required_rgb_streams[other])
        for index in range(len(required_rgb_streams))
        for other in range(index + 1, len(required_rgb_streams))
    ]
    required_pairs.extend(pair_key(stream, ROBOT_STATE) for stream in required_rgb_streams)
    for key in required_pairs:
        if key not in metrics.pairwise_skew_p95_ms:
            reasons.append(f"missing P95 alignment evidence for {key}")
        elif metrics.pairwise_skew_p95_ms[key] > RGB_GATE_MAX_ALIGNMENT_P95_MS:
            reasons.append(
                f"alignment p95 {key}={metrics.pairwise_skew_p95_ms[key]:.3f} ms > 50 ms"
            )

    if metrics.robot_state_sample_count <= 0:
        reasons.append("no AUBO state samples were recorded")
    if metrics.robot_state_error_count:
        reasons.append(f"AUBO state read errors observed: {metrics.robot_state_error_count}")
    if not metrics.robot_state_timestamp_traceable:
        reasons.append("AUBO state host-receive timestamp is not traceable")
    if not metrics.robot_state_timestamp_monotonic:
        reasons.append("AUBO state timestamps are not monotonic")

    decision = RGBGateDecision.FAIL if reasons else RGBGateDecision.PASS
    if not reasons:
        reasons.append("all frozen concurrent RGB/state criteria passed")
    return ConcurrentRGBGateResult(
        schema_version=RGB_GATE_SCHEMA_VERSION,
        required_rgb_streams=required_rgb_streams,
        decision=decision,
        stream_results=stream_results,
        reasons=tuple(reasons),
    )


def decide_camera_set_v1(
    *,
    isolated_wrist_result: RGBStreamGateResult,
    fixed_concurrent_result: ConcurrentRGBGateResult,
    three_concurrent_result: ConcurrentRGBGateResult | None,
    global_roi_accepted: bool,
    grasp_roi_accepted: bool,
) -> CameraSetDecision:
    """Apply the one-way CameraSetV1 decision without hiding failed evidence."""

    if not isinstance(global_roi_accepted, bool) or not isinstance(grasp_roi_accepted, bool):
        raise ValueError("ROI acceptance values must be bool")
    if not global_roi_accepted or not grasp_roi_accepted:
        return CameraSetDecision.BLOCKED
    if fixed_concurrent_result.required_rgb_streams != FIXED_RGB_STREAMS:
        raise ValueError("fixed_concurrent_result must evaluate the two fixed RGB streams")
    if fixed_concurrent_result.decision is not RGBGateDecision.PASS:
        return CameraSetDecision.BLOCKED
    if isolated_wrist_result.stream_name != WRIST_RGB:
        raise ValueError("isolated_wrist_result must describe wrist_rgb")
    if isolated_wrist_result.decision is RGBGateDecision.FAIL:
        return CameraSetDecision.TWO_RGB
    if three_concurrent_result is None:
        return CameraSetDecision.BLOCKED
    if three_concurrent_result.required_rgb_streams != THREE_RGB_STREAMS:
        raise ValueError("three_concurrent_result must evaluate all three RGB streams")
    if three_concurrent_result.decision is RGBGateDecision.PASS:
        return CameraSetDecision.THREE_RGB
    return CameraSetDecision.TWO_RGB


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
CURRENT_CAMERA_SET_SHA256: Final = FROZEN_CAMERA_SET_V2_SHA256


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


def _freeze_camera_set(
    path: Path,
    record: CameraSetV1Record,
    *,
    expected_filename: str,
    expected_schema_version: str,
) -> str:
    path = Path(path)
    if path.name != expected_filename:
        raise ValueError(f"The frozen record filename must be {expected_filename}")
    if record.schema_version != expected_schema_version:
        raise ValueError(
            f"{expected_filename} requires schema_version {expected_schema_version!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def freeze_camera_set_v1(path: Path, record: CameraSetV1Record) -> str:
    """Create CameraSetV1 once, never overwrite it, and return its SHA-256."""

    return _freeze_camera_set(
        path,
        record,
        expected_filename="CameraSetV1.json",
        expected_schema_version=CAMERA_SET_V1_SCHEMA_VERSION,
    )


def freeze_camera_set_v2(path: Path, record: CameraSetV1Record) -> str:
    """Create CameraSetV2 once, never overwrite it, and return its SHA-256."""

    return _freeze_camera_set(
        path,
        record,
        expected_filename="CameraSetV2.json",
        expected_schema_version=CAMERA_SET_V2_SCHEMA_VERSION,
    )
