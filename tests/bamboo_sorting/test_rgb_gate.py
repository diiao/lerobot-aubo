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

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_SCHEMA_VERSION,
    CAMERA_SET_V2_SCHEMA_VERSION,
    FIXED_RGB_STREAMS,
    FROZEN_CAMERA_SET_V1_SHA256,
    FROZEN_CAMERA_SET_V2_SHA256,
    THREE_RGB_STREAMS,
    CameraSetDecision,
    CameraSetV1Record,
    ConcurrentRGBMetrics,
    RGBGateDecision,
    RGBStreamMetrics,
    decide_camera_set_v1,
    evaluate_concurrent_rgb_gate,
    evaluate_isolated_wrist_rgb,
    freeze_camera_set_v1,
    freeze_camera_set_v2,
    load_camera_set_v1,
    load_camera_set_v2,
    pair_key,
    WRIST_RGB,
)


def _stream(name: str, **changes: object) -> RGBStreamMetrics:
    values = {
        "stream_name": name,
        "duration_s": 60.0,
        "sample_count": 600,
        "unique_sample_count": 600,
        "effective_unique_fps": 9.0,
        "drop_rate_fraction": 0.009,
        "duplicate_rate_fraction": 0.009,
        "frame_age_p95_ms": 100.0,
        "capture_error_count": 0,
        "timestamp_method": "host_receive_monotonic_after_sdk_return",
        "timestamp_traceable": True,
        "timestamp_monotonic": True,
        "frame_identity_method": "vendor_frame_id",
    }
    values.update(changes)
    return RGBStreamMetrics(**values)


def _concurrent(*, include_wrist: bool = True, **changes: object) -> ConcurrentRGBMetrics:
    names = THREE_RGB_STREAMS if include_wrist else FIXED_RGB_STREAMS
    skews = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            skews[pair_key(left, right)] = 50.0
        skews[pair_key(left, "robot_state")] = 50.0
    values = {
        "duration_s": 60.0,
        "policy_tick_count": 600,
        "stream_metrics": {name: _stream(name) for name in names},
        "pairwise_skew_p95_ms": skews,
        "robot_state_sample_count": 1500,
        "robot_state_error_count": 0,
        "robot_state_timestamp_traceable": True,
        "robot_state_timestamp_monotonic": True,
    }
    values.update(changes)
    return ConcurrentRGBMetrics(**values)


def test_isolated_gate_uses_strict_drop_and_duplicate_boundaries() -> None:
    passing = evaluate_isolated_wrist_rgb(_stream("wrist_rgb"))
    drop_failure = evaluate_isolated_wrist_rgb(_stream("wrist_rgb", drop_rate_fraction=0.01))
    duplicate_failure = evaluate_isolated_wrist_rgb(
        _stream("wrist_rgb", duplicate_rate_fraction=0.01)
    )

    assert passing.decision is RGBGateDecision.PASS
    assert drop_failure.decision is RGBGateDecision.FAIL
    assert duplicate_failure.decision is RGBGateDecision.FAIL


@pytest.mark.parametrize(
    "changes",
    [
        {"effective_unique_fps": 8.999},
        {"frame_age_p95_ms": 100.001},
        {"capture_error_count": 1},
        {"timestamp_traceable": False},
        {"timestamp_monotonic": False},
    ],
)
def test_isolated_gate_rejects_each_non_rate_failure(changes: dict[str, object]) -> None:
    result = evaluate_isolated_wrist_rgb(_stream("wrist_rgb", **changes))

    assert result.decision is RGBGateDecision.FAIL


def test_concurrent_gate_requires_60_seconds_all_pairs_and_read_only_state_samples() -> None:
    passing = evaluate_concurrent_rgb_gate(
        _concurrent(), required_rgb_streams=THREE_RGB_STREAMS
    )
    too_short = evaluate_concurrent_rgb_gate(
        _concurrent(duration_s=59.999), required_rgb_streams=THREE_RGB_STREAMS
    )
    skews = dict(_concurrent().pairwise_skew_p95_ms)
    skews.pop(pair_key("wrist_rgb", "robot_state"))
    missing_pair = evaluate_concurrent_rgb_gate(
        _concurrent(pairwise_skew_p95_ms=skews), required_rgb_streams=THREE_RGB_STREAMS
    )

    assert passing.decision is RGBGateDecision.PASS
    assert too_short.decision is RGBGateDecision.FAIL
    assert missing_pair.decision is RGBGateDecision.FAIL


def test_fixed_pair_failure_blocks_c0() -> None:
    isolated = evaluate_isolated_wrist_rgb(_stream("wrist_rgb"))
    fixed_metrics = _concurrent(include_wrist=False)
    failed_streams = dict(fixed_metrics.stream_metrics)
    failed_streams["global_rgb"] = replace(failed_streams["global_rgb"], effective_unique_fps=8.0)
    fixed = evaluate_concurrent_rgb_gate(
        replace(fixed_metrics, stream_metrics=failed_streams),
        required_rgb_streams=FIXED_RGB_STREAMS,
    )

    assert (
        decide_camera_set_v1(
            isolated_wrist_result=isolated,
            fixed_concurrent_result=fixed,
            three_concurrent_result=None,
            global_roi_accepted=True,
            grasp_roi_accepted=True,
        )
        is CameraSetDecision.BLOCKED
    )


def test_wrist_failure_freezes_two_rgb_after_fixed_pair_passes() -> None:
    isolated = evaluate_isolated_wrist_rgb(_stream("wrist_rgb", effective_unique_fps=8.0))
    fixed = evaluate_concurrent_rgb_gate(
        _concurrent(include_wrist=False), required_rgb_streams=FIXED_RGB_STREAMS
    )

    decision = decide_camera_set_v1(
        isolated_wrist_result=isolated,
        fixed_concurrent_result=fixed,
        three_concurrent_result=None,
        global_roi_accepted=True,
        grasp_roi_accepted=True,
    )

    assert decision is CameraSetDecision.TWO_RGB


def test_three_rgb_requires_isolated_and_concurrent_passes() -> None:
    isolated = evaluate_isolated_wrist_rgb(_stream("wrist_rgb"))
    concurrent_metrics = _concurrent()
    fixed = evaluate_concurrent_rgb_gate(
        concurrent_metrics, required_rgb_streams=FIXED_RGB_STREAMS
    )
    three = evaluate_concurrent_rgb_gate(
        concurrent_metrics, required_rgb_streams=THREE_RGB_STREAMS
    )

    decision = decide_camera_set_v1(
        isolated_wrist_result=isolated,
        fixed_concurrent_result=fixed,
        three_concurrent_result=three,
        global_roi_accepted=True,
        grasp_roi_accepted=True,
    )

    assert decision is CameraSetDecision.THREE_RGB


def _record() -> CameraSetV1Record:
    return CameraSetV1Record(
        schema_version=CAMERA_SET_SCHEMA_VERSION,
        frozen_at_utc="2026-09-09T09:00:00Z",
        decision=CameraSetDecision.TWO_RGB,
        camera_streams=FIXED_RGB_STREAMS,
        physical_roles={"global_rgb": "fixed global", "grasp_rgb": "fixed close view"},
        capture_profiles={
            "global_rgb": {"width": 640, "height": 480, "fourcc": "MJPG"},
            "grasp_rgb": {"width": 640, "height": 480, "fourcc": "MJPG"},
        },
        timestamp_methods={
            "global_rgb": "host_receive_monotonic",
            "grasp_rgb": "host_receive_monotonic",
        },
        evidence_sha256={"isolated_report": "a" * 64, "concurrent_report": "b" * 64},
        global_roi_evidence_ref="operator-reviewed-global-preview",
        grasp_roi_evidence_ref="operator-reviewed-grasp-preview",
    )


def test_camera_set_v1_is_created_once_and_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "CameraSetV1.json"

    digest = freeze_camera_set_v1(target, _record())

    assert len(digest) == 64
    assert json.loads(target.read_text())["schema_version"] == CAMERA_SET_SCHEMA_VERSION
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        freeze_camera_set_v1(target, _record())


def test_camera_set_v2_freeze_requires_matching_version_and_filename(tmp_path: Path) -> None:
    v2_record = replace(_record(), schema_version=CAMERA_SET_V2_SCHEMA_VERSION)
    target = tmp_path / "CameraSetV2.json"

    digest = freeze_camera_set_v2(target, v2_record)

    assert len(digest) == 64
    assert json.loads(target.read_text())["schema_version"] == CAMERA_SET_V2_SCHEMA_VERSION
    with pytest.raises(ValueError, match="requires schema_version"):
        freeze_camera_set_v2(tmp_path / "other" / "CameraSetV2.json", _record())
    with pytest.raises(ValueError, match="filename must be CameraSetV2.json"):
        freeze_camera_set_v2(tmp_path / "CameraSetV1.json", v2_record)


def test_camera_set_v1_deep_freezes_capture_profiles() -> None:
    profile = {"width": 640, "height": 480, "fourcc": "MJPG"}
    record = CameraSetV1Record(
        schema_version=CAMERA_SET_SCHEMA_VERSION,
        frozen_at_utc="2026-09-09T09:00:00Z",
        decision=CameraSetDecision.TWO_RGB,
        camera_streams=FIXED_RGB_STREAMS,
        physical_roles={"global_rgb": "fixed global", "grasp_rgb": "fixed close view"},
        capture_profiles={"global_rgb": profile, "grasp_rgb": dict(profile)},
        timestamp_methods={
            "global_rgb": "host_receive_monotonic",
            "grasp_rgb": "host_receive_monotonic",
        },
        evidence_sha256={"isolated_report": "a" * 64, "concurrent_report": "b" * 64},
        global_roi_evidence_ref="operator-reviewed-global-preview",
        grasp_roi_evidence_ref="operator-reviewed-grasp-preview",
    )

    profile["width"] = 1

    assert record.capture_profiles["global_rgb"]["width"] == 640
    with pytest.raises(TypeError):
        record.capture_profiles["global_rgb"]["width"] = 1


def test_load_frozen_camera_set_v1_is_two_rgb_without_wrist() -> None:
    record, digest = load_camera_set_v1()

    assert digest == FROZEN_CAMERA_SET_V1_SHA256
    assert record.decision is CameraSetDecision.TWO_RGB
    assert record.camera_streams == FIXED_RGB_STREAMS
    assert WRIST_RGB not in record.camera_streams
    assert record.immutable is True


def test_load_frozen_camera_set_v2_has_corrected_physical_roles() -> None:
    record, digest = load_camera_set_v2()

    assert digest == FROZEN_CAMERA_SET_V2_SHA256
    assert record.schema_version == CAMERA_SET_V2_SCHEMA_VERSION
    assert record.decision is CameraSetDecision.TWO_RGB
    assert record.camera_streams == FIXED_RGB_STREAMS
    assert "external stationary GENERAL WEBCAM" in record.physical_roles["global_rgb"]
    assert "eye-in-hand Sonix USB2.0_CAM1" in record.physical_roles["grasp_rgb"]
    assert record.evidence_sha256["legacy_camera_set_v1"] == FROZEN_CAMERA_SET_V1_SHA256
    assert WRIST_RGB not in record.camera_streams
    assert record.immutable is True
