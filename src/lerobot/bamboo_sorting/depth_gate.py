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

"""Pre-registered Phase A Depth Go/No-Go decision rules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

DEPTH_GATE_SCHEMA_VERSION: Final = "DepthGateV1"
STATIC_GROUP_MINIMUMS: Final = MappingProxyType(
    {"single_strip": 8, "two_strips": 8, "three_to_five": 8, "five_to_ten": 8}
)


class DepthGateDecision(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GO_10HZ = "go_10hz"
    CONDITIONAL_GO_5HZ = "conditional_go_5hz"
    NO_GO = "no_go"


@dataclass(frozen=True)
class DepthGateEvidence:
    """Independent sampling counts required before evaluating performance."""

    static_placements_by_group: Mapping[str, int]
    minimum_poses_per_placement: int
    minimum_frames_per_pose: int
    representative_mask_count: int
    motion_sequence_count: int

    def __post_init__(self) -> None:
        counts = dict(self.static_placements_by_group)
        if any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in counts.items()
        ):
            raise ValueError("static placement counts must be non-negative integers")
        object.__setattr__(self, "static_placements_by_group", MappingProxyType(counts))
        for name in (
            "minimum_poses_per_placement",
            "minimum_frames_per_pose",
            "representative_mask_count",
            "motion_sequence_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def required_representative_mask_count(self) -> int:
        return sum(STATIC_GROUP_MINIMUMS.values()) * 3

    def insufficiency_reasons(self) -> tuple[str, ...]:
        reasons = []
        for group, minimum in STATIC_GROUP_MINIMUMS.items():
            actual = self.static_placements_by_group.get(group, 0)
            if actual < minimum:
                reasons.append(f"{group} placements {actual} < {minimum}")
        if self.minimum_poses_per_placement < 3:
            reasons.append(f"minimum poses per placement {self.minimum_poses_per_placement} < 3")
        if self.minimum_frames_per_pose < 20:
            reasons.append(f"minimum frames per pose {self.minimum_frames_per_pose} < 20")
        required_masks = self.required_representative_mask_count
        if self.representative_mask_count < required_masks:
            reasons.append(f"representative masks {self.representative_mask_count} < {required_masks}")
        if self.motion_sequence_count < 12:
            reasons.append(f"motion sequences {self.motion_sequence_count} < 12")
        return tuple(reasons)


@dataclass(frozen=True)
class DepthGateMetrics:
    """Aggregate metrics computed from independent scenes and motion sequences."""

    effective_3d_fps: float
    drop_rate_fraction: float
    frame_age_p95_ms: float
    rgbd_state_offset_p95_ms: float
    layer_order_accuracy_fraction: float
    minimum_layer_gap_to_noise_ratio: float
    valid_depth_median_fraction: float
    valid_depth_p10_fraction: float
    timestamp_traceable: bool
    frame_order_valid: bool
    long_blocking_observed: bool
    systematic_contact_holes_observed: bool

    def __post_init__(self) -> None:
        for name in (
            "effective_3d_fps",
            "drop_rate_fraction",
            "frame_age_p95_ms",
            "rgbd_state_offset_p95_ms",
            "layer_order_accuracy_fraction",
            "minimum_layer_gap_to_noise_ratio",
            "valid_depth_median_fraction",
            "valid_depth_p10_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.effective_3d_fps < 0 or self.frame_age_p95_ms < 0 or self.rgbd_state_offset_p95_ms < 0:
            raise ValueError("frequency and timing metrics must be non-negative")
        for name in (
            "drop_rate_fraction",
            "layer_order_accuracy_fraction",
            "valid_depth_median_fraction",
            "valid_depth_p10_fraction",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.minimum_layer_gap_to_noise_ratio < 0:
            raise ValueError("minimum_layer_gap_to_noise_ratio must be non-negative")
        for name in (
            "timestamp_traceable",
            "frame_order_valid",
            "long_blocking_observed",
            "systematic_contact_holes_observed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")


@dataclass(frozen=True)
class DepthGateResult:
    schema_version: str
    decision: DepthGateDecision
    reasons: tuple[str, ...]
    rgb_only_mainline_preserved: bool = True

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["decision"] = self.decision.value
        return result


def _shared_depth_quality_failures(metrics: DepthGateMetrics) -> list[str]:
    reasons = []
    if metrics.drop_rate_fraction >= 0.01:
        reasons.append(f"drop rate {metrics.drop_rate_fraction:.3%} is not below 1%")
    if metrics.layer_order_accuracy_fraction < 0.95:
        reasons.append(f"layer order accuracy {metrics.layer_order_accuracy_fraction:.3%} < 95%")
    if metrics.minimum_layer_gap_to_noise_ratio <= 3.0:
        reasons.append(
            f"minimum layer gap/noise ratio {metrics.minimum_layer_gap_to_noise_ratio:.3f} is not above 3"
        )
    if metrics.valid_depth_median_fraction < 0.80:
        reasons.append(f"valid depth median {metrics.valid_depth_median_fraction:.3%} < 80%")
    if metrics.valid_depth_p10_fraction < 0.60:
        reasons.append(f"valid depth P10 {metrics.valid_depth_p10_fraction:.3%} < 60%")
    if not metrics.timestamp_traceable:
        reasons.append("timestamps are not traceable")
    if not metrics.frame_order_valid:
        reasons.append("frame order is invalid")
    if metrics.long_blocking_observed:
        reasons.append("long blocking was observed")
    if metrics.systematic_contact_holes_observed:
        reasons.append("systematic contact-region holes were observed")
    return reasons


def evaluate_depth_gate(evidence: DepthGateEvidence, metrics: DepthGateMetrics) -> DepthGateResult:
    """Apply frozen thresholds without considering downstream model results."""

    insufficiency = evidence.insufficiency_reasons()
    if insufficiency:
        return DepthGateResult(
            schema_version=DEPTH_GATE_SCHEMA_VERSION,
            decision=DepthGateDecision.INSUFFICIENT_EVIDENCE,
            reasons=insufficiency,
        )

    failures = _shared_depth_quality_failures(metrics)
    if failures:
        return DepthGateResult(
            schema_version=DEPTH_GATE_SCHEMA_VERSION,
            decision=DepthGateDecision.NO_GO,
            reasons=tuple(failures),
        )

    ten_hz_failures = []
    if metrics.effective_3d_fps < 9.0:
        ten_hz_failures.append(f"effective 3D FPS {metrics.effective_3d_fps:.3f} < 9")
    if metrics.frame_age_p95_ms > 100.0:
        ten_hz_failures.append(f"frame age p95 {metrics.frame_age_p95_ms:.3f} ms > 100 ms")
    if metrics.rgbd_state_offset_p95_ms > 50.0:
        ten_hz_failures.append(
            f"RGB-D/state offset p95 {metrics.rgbd_state_offset_p95_ms:.3f} ms > 50 ms"
        )
    if not ten_hz_failures:
        return DepthGateResult(
            schema_version=DEPTH_GATE_SCHEMA_VERSION,
            decision=DepthGateDecision.GO_10HZ,
            reasons=("all frozen 10 Hz criteria passed",),
        )

    five_hz_failures = []
    if metrics.effective_3d_fps < 4.5:
        five_hz_failures.append(f"effective 3D FPS {metrics.effective_3d_fps:.3f} < 4.5")
    if metrics.frame_age_p95_ms > 200.0:
        five_hz_failures.append(f"frame age p95 {metrics.frame_age_p95_ms:.3f} ms > 200 ms")
    if metrics.rgbd_state_offset_p95_ms > 100.0:
        five_hz_failures.append(
            f"RGB-D/state offset p95 {metrics.rgbd_state_offset_p95_ms:.3f} ms > 100 ms"
        )
    if five_hz_failures:
        return DepthGateResult(
            schema_version=DEPTH_GATE_SCHEMA_VERSION,
            decision=DepthGateDecision.NO_GO,
            reasons=tuple(ten_hz_failures + five_hz_failures),
        )
    return DepthGateResult(
        schema_version=DEPTH_GATE_SCHEMA_VERSION,
        decision=DepthGateDecision.CONDITIONAL_GO_5HZ,
        reasons=tuple(ten_hz_failures),
    )


def build_depth_gate_manifest() -> dict[str, object]:
    """Return the pre-registered sample sizes and decision thresholds."""

    return {
        "schema_version": DEPTH_GATE_SCHEMA_VERSION,
        "static_group_minimums": dict(STATIC_GROUP_MINIMUMS),
        "minimum_poses_per_placement": 3,
        "minimum_frames_per_pose": 20,
        "minimum_motion_sequences": 12,
        "representative_masks_per_pose": 1,
        "continuous_frames_are_not_independent_masks": True,
        "thresholds": {
            "go_10hz": {"fps_min": 9.0, "frame_age_p95_ms_max": 100.0, "offset_p95_ms_max": 50.0},
            "conditional_go_5hz": {
                "fps_min": 4.5,
                "frame_age_p95_ms_max": 200.0,
                "offset_p95_ms_max": 100.0,
            },
            "shared": {
                "drop_rate_strictly_below": 0.01,
                "layer_order_accuracy_min": 0.95,
                "layer_gap_to_noise_ratio_strictly_above": 3.0,
                "valid_depth_median_min": 0.80,
                "valid_depth_p10_min": 0.60,
            },
        },
        "no_go_fallback": "rgb_only_language_conditioned_smolvla",
    }
