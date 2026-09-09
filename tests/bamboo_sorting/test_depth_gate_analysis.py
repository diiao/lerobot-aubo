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

from lerobot.bamboo_sorting.depth_gate import DepthGateDecision
from lerobot.bamboo_sorting.depth_gate_analysis import (
    MotionSequenceMeasurements,
    StaticPoseMeasurements,
    analyze_depth_gate,
)


def _static_records() -> list[StaticPoseMeasurements]:
    records = []
    groups = ("single_strip", "two_strips", "three_to_five", "five_to_ten")
    for group in groups:
        for placement in range(8):
            for pose in range(3):
                has_layers = group != "single_strip"
                records.append(
                    StaticPoseMeasurements(
                        group=group,
                        placement_id=f"{group}-{placement}",
                        pose_id=f"pose-{pose}",
                        captured_frame_count=20,
                        representative_mask_annotated=True,
                        valid_depth_fraction=0.9,
                        layer_order_correct=True if has_layers else None,
                        layer_gap_to_noise_ratio=4.0 if has_layers else None,
                    )
                )
    return records


def _motion_records(*, period_s: float = 0.1) -> list[MotionSequenceMeasurements]:
    return [
        MotionSequenceMeasurements(
            sequence_id=f"motion-{index}",
            expected_frame_count=20,
            frame_receive_monotonic_s=tuple(index * 10 + frame * period_s for frame in range(20)),
            frame_age_ms=(80.0,) * 20,
            rgbd_state_offset_ms=(40.0,) * 20,
            frame_ids=tuple(index * 100 + frame for frame in range(20)),
            timestamps_traceable=True,
            long_blocking_observed=False,
        )
        for index in range(12)
    ]


def test_complete_independent_evidence_produces_10hz_go() -> None:
    report = analyze_depth_gate(
        _static_records(),
        _motion_records(),
        contact_hole_review_complete=True,
        systematic_contact_holes_observed=False,
    )

    assert report.result.decision is DepthGateDecision.GO_10HZ
    assert report.evidence.static_placements_by_group["single_strip"] == 8
    assert report.evidence.representative_mask_count == 96
    assert report.metrics.effective_3d_fps == pytest.approx(10.0)
    assert report.metrics.layer_order_accuracy_fraction == 1.0
    assert report.motion_received_frame_count == report.motion_expected_frame_count == 240
    assert report.to_dict()["result"]["rgb_only_mainline_preserved"] is True


def test_five_hz_measurements_produce_conditional_go() -> None:
    report = analyze_depth_gate(
        _static_records(),
        _motion_records(period_s=0.2),
        contact_hole_review_complete=True,
        systematic_contact_holes_observed=False,
    )

    assert report.result.decision is DepthGateDecision.CONDITIONAL_GO_5HZ


def test_incomplete_evidence_stops_before_metric_interpretation() -> None:
    report = analyze_depth_gate(
        _static_records()[:2],
        [],
        contact_hole_review_complete=False,
        systematic_contact_holes_observed=False,
    )

    assert report.result.decision is DepthGateDecision.INSUFFICIENT_EVIDENCE
    assert report.metrics is None
    assert any("single_strip placements" in reason for reason in report.result.reasons)


def test_duplicate_static_pose_or_motion_sequence_is_rejected() -> None:
    static = _static_records()
    with pytest.raises(ValueError, match="static pose IDs"):
        analyze_depth_gate(
            [static[0], static[0]],
            [],
            contact_hole_review_complete=False,
            systematic_contact_holes_observed=False,
        )

    motion = _motion_records()
    with pytest.raises(ValueError, match="motion sequence IDs"):
        analyze_depth_gate(
            [],
            [motion[0], motion[0]],
            contact_hole_review_complete=False,
            systematic_contact_holes_observed=False,
        )


def test_frame_loss_and_reordered_ids_are_derived_from_records() -> None:
    motion = _motion_records()
    motion[0] = replace(
        motion[0],
        expected_frame_count=21,
        frame_ids=(0, 2, 1, *range(3, 20)),
    )
    report = analyze_depth_gate(
        _static_records(),
        motion,
        contact_hole_review_complete=True,
        systematic_contact_holes_observed=False,
    )

    assert report.metrics.drop_rate_fraction == pytest.approx(1 / 241)
    assert not report.metrics.frame_order_valid
    assert report.result.decision is DepthGateDecision.NO_GO


def test_complete_sampling_requires_layer_and_contact_hole_review() -> None:
    static = [
        replace(record, layer_order_correct=None, layer_gap_to_noise_ratio=None)
        for record in _static_records()
    ]
    with pytest.raises(ValueError, match="contact-hole review"):
        analyze_depth_gate(
            static,
            _motion_records(),
            contact_hole_review_complete=False,
            systematic_contact_holes_observed=False,
        )
    with pytest.raises(ValueError, match="layer-order"):
        analyze_depth_gate(
            static,
            _motion_records(),
            contact_hole_review_complete=True,
            systematic_contact_holes_observed=False,
        )
