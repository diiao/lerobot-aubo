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

from lerobot.bamboo_sorting.depth_gate import (
    DepthGateDecision,
    DepthGateEvidence,
    DepthGateMetrics,
    build_depth_gate_manifest,
    evaluate_depth_gate,
)


def _evidence(**changes: object) -> DepthGateEvidence:
    values = {
        "static_placements_by_group": {
            "single_strip": 8,
            "two_strips": 8,
            "three_to_five": 8,
            "five_to_ten": 8,
        },
        "minimum_poses_per_placement": 3,
        "minimum_frames_per_pose": 20,
        "representative_mask_count": 96,
        "motion_sequence_count": 12,
    }
    values.update(changes)
    return DepthGateEvidence(**values)


def _metrics(**changes: object) -> DepthGateMetrics:
    values = {
        "effective_3d_fps": 9.0,
        "drop_rate_fraction": 0.009,
        "frame_age_p95_ms": 100.0,
        "rgbd_state_offset_p95_ms": 50.0,
        "layer_order_accuracy_fraction": 0.95,
        "minimum_layer_gap_to_noise_ratio": 3.01,
        "valid_depth_median_fraction": 0.80,
        "valid_depth_p10_fraction": 0.60,
        "timestamp_traceable": True,
        "frame_order_valid": True,
        "long_blocking_observed": False,
        "systematic_contact_holes_observed": False,
    }
    values.update(changes)
    return DepthGateMetrics(**values)


def test_exact_10hz_boundaries_pass_but_drop_and_noise_are_strict() -> None:
    result = evaluate_depth_gate(_evidence(), _metrics())

    assert result.decision is DepthGateDecision.GO_10HZ
    assert result.rgb_only_mainline_preserved
    assert result.to_dict()["decision"] == "go_10hz"

    assert evaluate_depth_gate(
        _evidence(), replace(_metrics(), drop_rate_fraction=0.01)
    ).decision is DepthGateDecision.NO_GO
    assert evaluate_depth_gate(
        _evidence(), replace(_metrics(), minimum_layer_gap_to_noise_ratio=3.0)
    ).decision is DepthGateDecision.NO_GO


def test_five_hz_conditional_go_uses_independent_frequency_thresholds() -> None:
    result = evaluate_depth_gate(
        _evidence(),
        _metrics(effective_3d_fps=4.5, frame_age_p95_ms=200.0, rgbd_state_offset_p95_ms=100.0),
    )

    assert result.decision is DepthGateDecision.CONDITIONAL_GO_5HZ
    assert any("< 9" in reason for reason in result.reasons)


def test_below_five_hz_or_failed_depth_quality_is_no_go_but_vla_remains() -> None:
    slow = evaluate_depth_gate(_evidence(), _metrics(effective_3d_fps=4.49))
    poor_depth = evaluate_depth_gate(_evidence(), _metrics(valid_depth_p10_fraction=0.59))

    assert slow.decision is DepthGateDecision.NO_GO
    assert poor_depth.decision is DepthGateDecision.NO_GO
    assert slow.rgb_only_mainline_preserved
    assert poor_depth.rgb_only_mainline_preserved


@pytest.mark.parametrize(
    "changes",
    [
        {"timestamp_traceable": False},
        {"frame_order_valid": False},
        {"long_blocking_observed": True},
        {"systematic_contact_holes_observed": True},
        {"layer_order_accuracy_fraction": 0.949},
        {"valid_depth_median_fraction": 0.799},
    ],
)
def test_shared_hard_failures_are_no_go(changes: dict[str, object]) -> None:
    assert evaluate_depth_gate(_evidence(), _metrics(**changes)).decision is DepthGateDecision.NO_GO


def test_incomplete_sampling_is_not_mislabeled_as_sensor_failure() -> None:
    evidence = _evidence(
        static_placements_by_group={"single_strip": 8},
        representative_mask_count=24,
        motion_sequence_count=2,
    )
    result = evaluate_depth_gate(evidence, _metrics())

    assert result.decision is DepthGateDecision.INSUFFICIENT_EVIDENCE
    assert any("two_strips" in reason for reason in result.reasons)
    assert any("motion sequences" in reason for reason in result.reasons)


def test_manifest_freezes_representative_frame_and_fallback_rules() -> None:
    manifest = build_depth_gate_manifest()

    assert manifest["representative_masks_per_pose"] == 1
    assert manifest["continuous_frames_are_not_independent_masks"] is True
    assert manifest["no_go_fallback"] == "rgb_only_language_conditioned_smolvla"
