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

"""Offline observation and timestamp contract for the AUBO VLA study.

This module contains no device SDK calls. Drivers must first convert their raw
timestamps into the common host-monotonic ``sync_timestamp_s`` domain before
constructing an observation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .contracts import ACTION_SCHEMA_VERSION, INSTRUCTION_SCHEMA_VERSION, validate_instruction_fields

OBSERVATION_SCHEMA_VERSION: Final = "EmbodiedObservationV1"
RGB_STREAM_KEYS: Final = ("global_rgb", "grasp_rgb", "wrist_rgb")
ROBOT_STATE_STREAM_KEY: Final = "robot_state"
DEPTH_STREAM_KEY: Final = "wrist_depth_m"
BASE_TIMESTAMP_STREAMS: Final = (*RGB_STREAM_KEYS, ROBOT_STATE_STREAM_KEY)
SYNC_METHODS: Final = frozenset({"host_receive", "device_to_host_calibrated"})


@dataclass(frozen=True)
class ObservationStateFieldSpec:
    """One ordered scalar in the run06-compatible robot state vector."""

    name: str
    unit: str
    semantics: str


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics bound to one stored RGB image layout."""

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.width_px, bool) or not isinstance(self.width_px, int) or self.width_px <= 0:
            raise ValueError("width_px must be a positive integer")
        if isinstance(self.height_px, bool) or not isinstance(self.height_px, int) or self.height_px <= 0:
            raise ValueError("height_px must be a positive integer")
        for name in ("fx_px", "fy_px", "cx_px", "cy_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("fx_px and fy_px must be positive")
        if not isinstance(self.distortion_model, str) or not self.distortion_model:
            raise ValueError("distortion_model must be a non-empty string")
        coefficients = tuple(self.distortion_coefficients)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in coefficients):
            raise ValueError("distortion_coefficients must contain only finite numbers")
        object.__setattr__(self, "distortion_coefficients", coefficients)


OBSERVATION_STATE_FIELD_SPECS: Final = (
    *(ObservationStateFieldSpec(f"J{index}", "deg", f"measured AUBO joint {index}") for index in range(1, 7)),
    ObservationStateFieldSpec("ee.x", "m", "measured base-frame TCP x"),
    ObservationStateFieldSpec("ee.y", "m", "measured base-frame TCP y"),
    ObservationStateFieldSpec("ee.z", "m", "measured base-frame TCP z"),
    ObservationStateFieldSpec("ee.wx", "rad", "measured base-frame TCP rotation-vector x"),
    ObservationStateFieldSpec("ee.wy", "rad", "measured base-frame TCP rotation-vector y"),
    ObservationStateFieldSpec("ee.wz", "rad", "measured base-frame TCP rotation-vector z"),
    ObservationStateFieldSpec(
        "gripper_pos",
        "legacy_binary",
        "last commanded 0/100 state; not measured gripper feedback",
    ),
)
OBSERVATION_STATE_FIELD_NAMES: Final = tuple(field.name for field in OBSERVATION_STATE_FIELD_SPECS)


@dataclass(frozen=True)
class SensorTimestamp:
    """Raw and normalized timestamps for one sensor sample.

    ``sync_timestamp_s`` and ``host_receive_monotonic_s`` must share the host
    monotonic clock domain. ``device_timestamp_s`` remains raw evidence and is
    never compared across devices without a calibrated clock conversion.
    """

    sync_timestamp_s: float
    host_receive_monotonic_s: float
    sync_method: str
    device_timestamp_s: float | None = None
    device_clock_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("sync_timestamp_s", "host_receive_monotonic_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")

        if self.sync_method not in SYNC_METHODS:
            raise ValueError(f"sync_method must be one of {sorted(SYNC_METHODS)}")

        has_device_time = self.device_timestamp_s is not None
        has_clock_id = self.device_clock_id is not None
        if has_device_time != has_clock_id:
            raise ValueError("device_timestamp_s and device_clock_id must be provided together")
        if has_device_time:
            if isinstance(self.device_timestamp_s, bool) or not math.isfinite(self.device_timestamp_s):
                raise ValueError("device_timestamp_s must be finite")
            if not self.device_clock_id:
                raise ValueError("device_clock_id must be a non-empty string")

        if self.sync_method == "host_receive" and not math.isclose(
            self.sync_timestamp_s, self.host_receive_monotonic_s, abs_tol=1e-9
        ):
            raise ValueError("host_receive sync must use host_receive_monotonic_s as sync_timestamp_s")
        if self.sync_method == "device_to_host_calibrated" and not has_device_time:
            raise ValueError("device_to_host_calibrated sync requires a raw device timestamp")


@dataclass(frozen=True)
class TemporalAlignmentMetrics:
    """Measured cross-stream timing quality in milliseconds."""

    streams: tuple[str, ...]
    max_pairwise_skew_ms: float
    oldest_sample_age_ms: float


def validate_temporal_alignment(
    timestamps: Mapping[str, SensorTimestamp],
    *,
    required_streams: Sequence[str],
    reference_time_s: float,
    max_skew_ms: float,
    max_age_ms: float,
) -> TemporalAlignmentMetrics:
    """Validate freshness and skew after timestamps share one clock domain."""

    if isinstance(reference_time_s, bool) or not math.isfinite(reference_time_s) or reference_time_s < 0:
        raise ValueError("reference_time_s must be finite and non-negative")
    if isinstance(max_skew_ms, bool) or max_skew_ms < 0 or not math.isfinite(max_skew_ms):
        raise ValueError("max_skew_ms must be finite and non-negative")
    if isinstance(max_age_ms, bool) or max_age_ms < 0 or not math.isfinite(max_age_ms):
        raise ValueError("max_age_ms must be finite and non-negative")

    streams = tuple(required_streams)
    if not streams or len(set(streams)) != len(streams):
        raise ValueError("required_streams must be non-empty and contain no duplicates")

    missing = [stream for stream in streams if stream not in timestamps]
    if missing:
        raise ValueError(f"Missing timestamp streams: {missing}")

    sync_times = [timestamps[stream].sync_timestamp_s for stream in streams]
    newest_time = max(sync_times)
    oldest_time = min(sync_times)
    if newest_time > reference_time_s:
        raise ValueError("Sensor sync timestamp cannot be in the future")

    skew_ms = (newest_time - oldest_time) * 1000.0
    oldest_age_ms = (reference_time_s - oldest_time) * 1000.0
    if skew_ms > max_skew_ms:
        raise ValueError(f"Cross-stream skew {skew_ms:.3f} ms exceeds {max_skew_ms:.3f} ms")
    if oldest_age_ms > max_age_ms:
        raise ValueError(f"Oldest sample age {oldest_age_ms:.3f} ms exceeds {max_age_ms:.3f} ms")

    return TemporalAlignmentMetrics(
        streams=streams,
        max_pairwise_skew_ms=skew_ms,
        oldest_sample_age_ms=oldest_age_ms,
    )


def _validate_rgb(name: str, image: NDArray[np.uint8]) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have uint8 HxWx3 RGB layout")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} must have non-zero height and width")


def _validate_transform(name: str, transform: NDArray[np.floating]) -> None:
    if not isinstance(transform, np.ndarray) or transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 numpy array")
    if not np.issubdtype(transform.dtype, np.floating) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5
    ):
        raise ValueError(f"{name} rotation must be orthonormal with determinant +1")


@dataclass(frozen=True)
class EmbodiedObservationV1:
    """Validated multi-view observation without any execution permission."""

    global_rgb: NDArray[np.uint8]
    grasp_rgb: NDArray[np.uint8]
    wrist_rgb: NDArray[np.uint8]
    robot_state: Sequence[object]
    timestamps: Mapping[str, SensorTimestamp]
    observation_schema_version: str
    instruction_schema_version: str
    instruction_id: str
    instruction_text: str
    instruction_language: str
    scene_id: str
    calibration_version: str
    tool_version: str
    model_input_ref: str
    action_schema_version: str
    camera_intrinsics: Mapping[str, CameraIntrinsics]
    t_base_global_camera: NDArray[np.floating]
    t_base_grasp_camera: NDArray[np.floating]
    t_ee_camera: NDArray[np.floating]
    t_base_ee: NDArray[np.floating]
    wrist_depth_m: NDArray[np.floating] | None = None
    wrist_depth_valid: NDArray[np.bool_] | None = None
    wrist_xyz_m: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported observation schema {self.observation_schema_version!r}; "
                f"expected {OBSERVATION_SCHEMA_VERSION!r}"
            )
        for name in ("scene_id", "calibration_version", "tool_version", "model_input_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported action schema {self.action_schema_version!r}; "
                f"expected {ACTION_SCHEMA_VERSION!r}"
            )

        validate_instruction_fields(
            schema_version=self.instruction_schema_version,
            instruction_id=self.instruction_id,
            instruction_text=self.instruction_text,
            instruction_language=self.instruction_language,
        )

        for key in RGB_STREAM_KEYS:
            _validate_rgb(key, getattr(self, key))

        intrinsics_copy = dict(self.camera_intrinsics)
        missing_intrinsics = sorted(set(RGB_STREAM_KEYS) - intrinsics_copy.keys())
        if missing_intrinsics:
            raise ValueError(f"Missing camera intrinsics: {missing_intrinsics}")
        if not all(isinstance(value, CameraIntrinsics) for value in intrinsics_copy.values()):
            raise ValueError("camera_intrinsics values must be CameraIntrinsics instances")
        for key in RGB_STREAM_KEYS:
            intrinsics = intrinsics_copy[key]
            image = getattr(self, key)
            if (intrinsics.height_px, intrinsics.width_px) != image.shape[:2]:
                raise ValueError(f"camera_intrinsics.{key} dimensions must match its RGB image")
        object.__setattr__(self, "camera_intrinsics", MappingProxyType(intrinsics_copy))

        if isinstance(self.robot_state, (str, bytes)) or len(self.robot_state) != len(
            OBSERVATION_STATE_FIELD_SPECS
        ):
            raise ValueError(
                f"robot_state requires {len(OBSERVATION_STATE_FIELD_SPECS)} ordered scalar values"
            )
        state_values: list[float] = []
        for field, raw_value in zip(OBSERVATION_STATE_FIELD_SPECS, self.robot_state, strict=True):
            if isinstance(raw_value, bool):
                raise ValueError(f"robot_state.{field.name} must be numeric, not bool")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"robot_state.{field.name} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"robot_state.{field.name} must be finite")
            state_values.append(value)
        if state_values[-1] not in (0.0, 100.0):
            raise ValueError("robot_state.gripper_pos must be the last commanded legacy 0/100 state")
        object.__setattr__(self, "robot_state", tuple(state_values))

        _validate_transform("t_base_global_camera", self.t_base_global_camera)
        _validate_transform("t_base_grasp_camera", self.t_base_grasp_camera)
        _validate_transform("t_ee_camera", self.t_ee_camera)
        _validate_transform("t_base_ee", self.t_base_ee)

        timestamp_copy = dict(self.timestamps)
        required_timestamps = set(BASE_TIMESTAMP_STREAMS)
        if self.depth_enabled:
            required_timestamps.add(DEPTH_STREAM_KEY)
        missing_timestamps = sorted(required_timestamps - timestamp_copy.keys())
        if missing_timestamps:
            raise ValueError(f"Missing observation timestamps: {missing_timestamps}")
        if not all(isinstance(value, SensorTimestamp) for value in timestamp_copy.values()):
            raise ValueError("timestamps values must be SensorTimestamp instances")
        object.__setattr__(self, "timestamps", MappingProxyType(timestamp_copy))

        self._validate_depth_bundle()

    @property
    def depth_enabled(self) -> bool:
        """Return whether the complete metric depth bundle is present."""

        return self.wrist_depth_m is not None

    @property
    def t_base_camera(self) -> NDArray[np.floating]:
        """Derive the time-varying wrist-camera pose in the robot base frame."""

        return self.t_base_ee @ self.t_ee_camera

    @property
    def required_timestamp_streams(self) -> tuple[str, ...]:
        """Return streams that must be aligned for this observation."""

        if self.depth_enabled:
            return (*BASE_TIMESTAMP_STREAMS, DEPTH_STREAM_KEY)
        return BASE_TIMESTAMP_STREAMS

    def validate_timing(
        self, *, reference_time_s: float, max_skew_ms: float, max_age_ms: float
    ) -> TemporalAlignmentMetrics:
        """Validate this observation against one control-cycle timing gate."""

        return validate_temporal_alignment(
            self.timestamps,
            required_streams=self.required_timestamp_streams,
            reference_time_s=reference_time_s,
            max_skew_ms=max_skew_ms,
            max_age_ms=max_age_ms,
        )

    def _validate_depth_bundle(self) -> None:
        present = (
            self.wrist_depth_m is not None,
            self.wrist_depth_valid is not None,
            self.wrist_xyz_m is not None,
        )
        if any(present) and not all(present):
            raise ValueError("wrist_depth_m, wrist_depth_valid, and wrist_xyz_m must be present together")
        if not any(present):
            return

        depth = self.wrist_depth_m
        valid = self.wrist_depth_valid
        xyz = self.wrist_xyz_m
        assert depth is not None and valid is not None and xyz is not None
        if not isinstance(depth, np.ndarray) or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError("wrist_depth_m must be a floating-point numpy array")
        if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
            raise ValueError("wrist_depth_valid must be a boolean numpy array")
        if not isinstance(xyz, np.ndarray) or not np.issubdtype(xyz.dtype, np.floating):
            raise ValueError("wrist_xyz_m must be a floating-point numpy array")

        expected_shape = self.wrist_rgb.shape[:2]
        if (
            depth.shape != expected_shape
            or valid.shape != expected_shape
            or xyz.shape != (*expected_shape, 3)
        ):
            raise ValueError("Depth, valid mask, and XYZ shapes must match wrist_rgb")
        if not valid.any():
            raise ValueError("wrist_depth_valid must contain at least one valid point")
        if not np.isfinite(depth[valid]).all() or (depth[valid] <= 0).any():
            raise ValueError("Valid wrist_depth_m pixels must be finite and positive")
        if not np.isfinite(xyz[valid]).all():
            raise ValueError("Valid wrist_xyz_m points must be finite")


def build_observation_manifest(*, depth_enabled: bool) -> dict[str, object]:
    """Build a serializable structural manifest without hardware assumptions."""

    if not isinstance(depth_enabled, bool):
        raise ValueError("depth_enabled must be bool")
    timestamp_streams = list(BASE_TIMESTAMP_STREAMS)
    if depth_enabled:
        timestamp_streams.append(DEPTH_STREAM_KEY)

    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "rgb_streams": list(RGB_STREAM_KEYS),
        "depth_fields": (
            ["wrist_depth_m", "wrist_depth_valid", "wrist_xyz_m"] if depth_enabled else []
        ),
        "robot_state_fields": [
            {"name": field.name, "unit": field.unit, "semantics": field.semantics}
            for field in OBSERVATION_STATE_FIELD_SPECS
        ],
        "timestamp_streams": timestamp_streams,
        "timestamp_domain": "host_monotonic_seconds",
        "camera_intrinsics": list(RGB_STREAM_KEYS),
        "stored_transforms": [
            "t_base_global_camera",
            "t_base_grasp_camera",
            "t_ee_camera",
            "t_base_ee",
        ],
        "derived_transforms": ["t_base_camera"],
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
