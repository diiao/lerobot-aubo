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

"""Read-only camera configuration and immutable profile regressions."""

from __future__ import annotations

import pytest

from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_SCHEMA_VERSION,
    CAMERA_SET_V2_SCHEMA_VERSION,
    FIXED_RGB_STREAMS,
    FROZEN_CAMERA_SET_V1_SHA256,
    FROZEN_CAMERA_SET_V2_SHA256,
    CameraSetDecision,
    CameraSetV1Record,
    load_camera_set_v1,
    load_camera_set_v2,
    WRIST_RGB,
)


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
