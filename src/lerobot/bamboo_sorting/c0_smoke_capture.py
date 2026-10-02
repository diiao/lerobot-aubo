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

"""CameraSetV2 device mapping shared by current capture and execution.

The historical module name is retained for its existing callers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from .rgb_gate import GLOBAL_RGB, GRASP_RGB, load_camera_set_v2

C0_SMOKE_IMAGE_KEYS: Final = (GLOBAL_RGB, GRASP_RGB)

C0_SMOKE_CAMERA_DEVICES: Final = {
    GLOBAL_RGB: (
        "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"
    ),
    GRASP_RGB: (
        "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
    ),
}

C0_SMOKE_CAMERA_MODELS: Final = {
    GLOBAL_RGB: "GENERAL WEBCAM",
    GRASP_RGB: "Sonix USB2.0_CAM1",
}

C0_SMOKE_LEGACY_ACT_KEYS: Final = {
    GLOBAL_RGB: "handeye",
    GRASP_RGB: "fixed",
}


class C0SmokeCaptureError(ValueError):
    """The configured streams do not match the current capture devices."""


def frozen_c0_smoke_camera_mapping(
    camera_set_path: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Return the verified CameraSetV2 mapping used by joint capture and execution."""

    # The loader verifies the file hash and schema before returning the record.
    record, _ = load_camera_set_v2(camera_set_path)
    if tuple(record.camera_streams) != C0_SMOKE_IMAGE_KEYS:
        raise C0SmokeCaptureError(
            f"camera streams must be {C0_SMOKE_IMAGE_KEYS}, not {record.camera_streams}"
        )

    mapping: dict[str, dict[str, object]] = {}
    for key in C0_SMOKE_IMAGE_KEYS:
        profile = record.capture_profiles.get(key)
        if not isinstance(profile, Mapping):
            raise C0SmokeCaptureError(f"CameraSetV2 missing capture profile for {key}")
        device = profile.get("device")
        if device != C0_SMOKE_CAMERA_DEVICES[key]:
            raise C0SmokeCaptureError(
                f"{key} device must be {C0_SMOKE_CAMERA_DEVICES[key]!r}, got {device!r}"
            )
        fps = profile.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise C0SmokeCaptureError(f"{key} capture fps is missing")
        mapping[key] = {
            "device": device,
            "fps": float(fps),
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "fourcc": str(profile["fourcc"]),
            "model": C0_SMOKE_CAMERA_MODELS[key],
            "legacy_act_key": C0_SMOKE_LEGACY_ACT_KEYS[key],
            "physical_role": record.physical_roles[key],
        }
    return mapping
