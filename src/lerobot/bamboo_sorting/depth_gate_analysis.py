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

"""Pure offline aggregation for the preregistered Phase A depth gate."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .depth_gate import (
    DEPTH_GATE_SCHEMA_VERSION,
    STATIC_GROUP_MINIMUMS,
    DepthGateDecision,
    DepthGateEvidence,
    DepthGateMetrics,
    DepthGateResult,
    evaluate_depth_gate,
)

DEPTH_GATE_ANALYSIS_SCHEMA_VERSION = "DepthGateAnalysisV1"


def _finite_non_negative_tuple(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(values)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in result
    ):
        raise ValueError(f"{name} must contain only finite non-negative numbers")
    return tuple(float(value) for value in result)


@dataclass(frozen=True)
class StaticPoseMeasurements:
    """Measurements from one pose; its burst frames are not independent masks."""

    group: str
    placement_id: str
    pose_id: str
    captured_frame_count: int
    representative_mask_annotated: bool
    valid_depth_fraction: float | None = None
    layer_order_correct: bool | None = None
    layer_gap_to_noise_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.group not in STATIC_GROUP_MINIMUMS:
            raise ValueError(f"group must be one of {sorted(STATIC_GROUP_MINIMUMS)}")
        for name in ("placement_id", "pose_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.captured_frame_count, bool)
            or not isinstance(self.captured_frame_count, int)
            or self.captured_frame_count < 0
        ):
            raise ValueError("captured_frame_count must be a non-negative integer")
        if not isinstance(self.representative_mask_annotated, bool):
            raise ValueError("representative_mask_annotated must be bool")
        if self.representative_mask_annotated != (self.valid_depth_fraction is not None):
            raise ValueError("An annotated representative mask requires valid_depth_fraction")
        if self.valid_depth_fraction is not None and not 0.0 <= self.valid_depth_fraction <= 1.0:
            raise ValueError("valid_depth_fraction must be within [0, 1]")
        has_order = self.layer_order_correct is not None
        has_ratio = self.layer_gap_to_noise_ratio is not None
        if has_order != has_ratio:
            raise ValueError("layer_order_correct and layer_gap_to_noise_ratio must be provided together")
        if has_order and not isinstance(self.layer_order_correct, bool):
            raise ValueError("layer_order_correct must be bool when provided")
        if has_ratio and (
            isinstance(self.layer_gap_to_noise_ratio, bool)
            or not isinstance(self.layer_gap_to_noise_ratio, (int, float))
            or not math.isfinite(self.layer_gap_to_noise_ratio)
            or self.layer_gap_to_noise_ratio < 0
        ):
            raise ValueError("layer_gap_to_noise_ratio must be finite and non-negative")


@dataclass(frozen=True)
class MotionSequenceMeasurements:
    """Timing records from one independently planned, no-contact motion sequence."""

    sequence_id: str
    expected_frame_count: int
    frame_receive_monotonic_s: Sequence[float]
    frame_age_ms: Sequence[float]
    rgbd_state_offset_ms: Sequence[float]
    frame_ids: Sequence[int]
    timestamps_traceable: bool
    long_blocking_observed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if (
            isinstance(self.expected_frame_count, bool)
            or not isinstance(self.expected_frame_count, int)
            or self.expected_frame_count <= 0
        ):
            raise ValueError("expected_frame_count must be a positive integer")
        receive_times = _finite_non_negative_tuple(
            "frame_receive_monotonic_s", self.frame_receive_monotonic_s
        )
        ages = _finite_non_negative_tuple("frame_age_ms", self.frame_age_ms)
        offsets = _finite_non_negative_tuple("rgbd_state_offset_ms", self.rgbd_state_offset_ms)
        frame_ids = tuple(self.frame_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in frame_ids):
            raise ValueError("frame_ids must contain non-negative integers")
        received_count = len(receive_times)
        if received_count > self.expected_frame_count:
            raise ValueError("received frame count cannot exceed expected_frame_count")
        if not (received_count == len(ages) == len(offsets) == len(frame_ids)):
            raise ValueError("motion sequence per-frame fields must have equal lengths")
        for name in ("timestamps_traceable", "long_blocking_observed"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        object.__setattr__(self, "frame_receive_monotonic_s", receive_times)
        object.__setattr__(self, "frame_age_ms", ages)
        object.__setattr__(self, "rgbd_state_offset_ms", offsets)
        object.__setattr__(self, "frame_ids", frame_ids)

    @property
    def frame_order_valid(self) -> bool:
        return all(left < right for left, right in zip(self.frame_ids, self.frame_ids[1:])) and all(
            left < right
            for left, right in zip(
                self.frame_receive_monotonic_s,
                self.frame_receive_monotonic_s[1:],
            )
        )


@dataclass(frozen=True)
class DepthGateAnalysisReport:
    schema_version: str
    depth_gate_schema_version: str
    evidence: DepthGateEvidence
    metrics: DepthGateMetrics | None
    result: DepthGateResult
    static_pose_count: int
    layer_label_count: int
    motion_received_frame_count: int
    motion_expected_frame_count: int

    def to_dict(self) -> dict[str, object]:
        evidence = {
            "static_placements_by_group": dict(self.evidence.static_placements_by_group),
            "minimum_poses_per_placement": self.evidence.minimum_poses_per_placement,
            "minimum_frames_per_pose": self.evidence.minimum_frames_per_pose,
            "representative_mask_count": self.evidence.representative_mask_count,
            "motion_sequence_count": self.evidence.motion_sequence_count,
        }
        return {
            "schema_version": self.schema_version,
            "depth_gate_schema_version": self.depth_gate_schema_version,
            "evidence": evidence,
            "metrics": None if self.metrics is None else asdict(self.metrics),
            "result": self.result.to_dict(),
            "static_pose_count": self.static_pose_count,
            "layer_label_count": self.layer_label_count,
            "motion_received_frame_count": self.motion_received_frame_count,
            "motion_expected_frame_count": self.motion_expected_frame_count,
        }


def _build_evidence(
    static_poses: tuple[StaticPoseMeasurements, ...],
    motion_sequences: tuple[MotionSequenceMeasurements, ...],
) -> DepthGateEvidence:
    placements_by_group = {
        group: len({pose.placement_id for pose in static_poses if pose.group == group})
        for group in STATIC_GROUP_MINIMUMS
    }
    poses_by_placement: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pose in static_poses:
        poses_by_placement[(pose.group, pose.placement_id)].add(pose.pose_id)
    return DepthGateEvidence(
        static_placements_by_group=placements_by_group,
        minimum_poses_per_placement=min(map(len, poses_by_placement.values()), default=0),
        minimum_frames_per_pose=min(
            (pose.captured_frame_count for pose in static_poses),
            default=0,
        ),
        representative_mask_count=sum(pose.representative_mask_annotated for pose in static_poses),
        motion_sequence_count=len(motion_sequences),
    )


def _aggregate_motion_metrics(
    motion_sequences: tuple[MotionSequenceMeasurements, ...],
) -> tuple[float, float, float, float, bool, bool, bool]:
    expected_count = sum(sequence.expected_frame_count for sequence in motion_sequences)
    received_count = sum(len(sequence.frame_ids) for sequence in motion_sequences)
    drop_rate = (expected_count - received_count) / expected_count

    observed_intervals = 0
    observed_duration_s = 0.0
    for sequence in motion_sequences:
        if len(sequence.frame_receive_monotonic_s) >= 2:
            observed_intervals += len(sequence.frame_receive_monotonic_s) - 1
            observed_duration_s += (
                sequence.frame_receive_monotonic_s[-1] - sequence.frame_receive_monotonic_s[0]
            )
    if observed_duration_s <= 0:
        raise ValueError("Complete evidence requires positive motion capture duration")

    ages = tuple(value for sequence in motion_sequences for value in sequence.frame_age_ms)
    offsets = tuple(value for sequence in motion_sequences for value in sequence.rgbd_state_offset_ms)
    if not ages or not offsets:
        raise ValueError("Complete evidence requires per-frame timing measurements")
    return (
        observed_intervals / observed_duration_s,
        drop_rate,
        float(np.percentile(ages, 95)),
        float(np.percentile(offsets, 95)),
        all(sequence.timestamps_traceable for sequence in motion_sequences),
        all(sequence.frame_order_valid for sequence in motion_sequences),
        any(sequence.long_blocking_observed for sequence in motion_sequences),
    )


def analyze_depth_gate(
    static_poses: Sequence[StaticPoseMeasurements],
    motion_sequences: Sequence[MotionSequenceMeasurements],
    *,
    contact_hole_review_complete: bool,
    systematic_contact_holes_observed: bool,
) -> DepthGateAnalysisReport:
    """Aggregate independent Phase A evidence and apply the frozen gate."""

    if not isinstance(contact_hole_review_complete, bool) or not isinstance(
        systematic_contact_holes_observed, bool
    ):
        raise ValueError("contact-hole review flags must be bool")
    static_records = tuple(static_poses)
    motion_records = tuple(motion_sequences)
    if not all(isinstance(record, StaticPoseMeasurements) for record in static_records):
        raise ValueError("static_poses must contain StaticPoseMeasurements")
    if not all(isinstance(record, MotionSequenceMeasurements) for record in motion_records):
        raise ValueError("motion_sequences must contain MotionSequenceMeasurements")
    if len({record.sequence_id for record in motion_records}) != len(motion_records):
        raise ValueError("motion sequence IDs must be unique")
    pose_keys = [(record.group, record.placement_id, record.pose_id) for record in static_records]
    duplicate_pose_keys = [key for key, count in Counter(pose_keys).items() if count > 1]
    if duplicate_pose_keys:
        raise ValueError(f"static pose IDs must be unique: {duplicate_pose_keys}")

    evidence = _build_evidence(static_records, motion_records)
    insufficiency = evidence.insufficiency_reasons()
    if insufficiency:
        result = DepthGateResult(
            schema_version=DEPTH_GATE_SCHEMA_VERSION,
            decision=DepthGateDecision.INSUFFICIENT_EVIDENCE,
            reasons=insufficiency,
        )
        return DepthGateAnalysisReport(
            schema_version=DEPTH_GATE_ANALYSIS_SCHEMA_VERSION,
            depth_gate_schema_version=DEPTH_GATE_SCHEMA_VERSION,
            evidence=evidence,
            metrics=None,
            result=result,
            static_pose_count=len(static_records),
            layer_label_count=sum(record.layer_order_correct is not None for record in static_records),
            motion_received_frame_count=sum(len(record.frame_ids) for record in motion_records),
            motion_expected_frame_count=sum(record.expected_frame_count for record in motion_records),
        )

    if not contact_hole_review_complete:
        raise ValueError("Complete sampling requires a completed contact-hole review")
    valid_fractions = tuple(
        record.valid_depth_fraction
        for record in static_records
        if record.valid_depth_fraction is not None
    )
    layer_records = tuple(record for record in static_records if record.layer_order_correct is not None)
    if not layer_records:
        raise ValueError("Complete sampling requires layer-order and gap/noise annotations")

    fps, drop_rate, age_p95, offset_p95, traceable, ordered, long_blocking = (
        _aggregate_motion_metrics(motion_records)
    )
    metrics = DepthGateMetrics(
        effective_3d_fps=fps,
        drop_rate_fraction=drop_rate,
        frame_age_p95_ms=age_p95,
        rgbd_state_offset_p95_ms=offset_p95,
        layer_order_accuracy_fraction=(
            sum(bool(record.layer_order_correct) for record in layer_records) / len(layer_records)
        ),
        minimum_layer_gap_to_noise_ratio=min(
            float(record.layer_gap_to_noise_ratio) for record in layer_records
        ),
        valid_depth_median_fraction=float(np.median(valid_fractions)),
        valid_depth_p10_fraction=float(np.percentile(valid_fractions, 10)),
        timestamp_traceable=traceable,
        frame_order_valid=ordered,
        long_blocking_observed=long_blocking,
        systematic_contact_holes_observed=systematic_contact_holes_observed,
    )
    result = evaluate_depth_gate(evidence, metrics)
    return DepthGateAnalysisReport(
        schema_version=DEPTH_GATE_ANALYSIS_SCHEMA_VERSION,
        depth_gate_schema_version=DEPTH_GATE_SCHEMA_VERSION,
        evidence=evidence,
        metrics=metrics,
        result=result,
        static_pose_count=len(static_records),
        layer_label_count=len(layer_records),
        motion_received_frame_count=sum(len(record.frame_ids) for record in motion_records),
        motion_expected_frame_count=sum(record.expected_frame_count for record in motion_records),
    )
