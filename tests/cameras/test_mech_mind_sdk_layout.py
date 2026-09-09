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

"""No-hardware compatibility checks against the installed vendor binding."""

from importlib.metadata import version

import numpy as np
import pytest

sdk = pytest.importorskip("mecheye.area_scan_3d_camera")

from lerobot.cameras.mech_mind.mech_mind_camera import (  # noqa: E402
    _convert_color_to_rgb,
    _reshape_vendor_array,
)


def test_installed_binding_matches_system_sdk_release() -> None:
    assert version("MechEyeAPI") == "2.6.0"


def test_vendor_in_memory_array_layouts_match_adapter_without_camera() -> None:
    color = sdk.Color2DImage()
    depth = sdk.DepthMap()
    xyz = sdk.UntexturedPointCloud()
    color.resize(3, 2)
    depth.resize(3, 2)
    xyz.resize(3, 2)

    assert color.data().shape == (2, 3, 3)
    assert color.data().dtype == np.uint8
    assert depth.data().shape == (2, 3)
    assert depth.data().dtype == np.float32
    assert xyz.data().shape == (2, 3, 3)
    assert xyz.data().dtype == np.float32
    assert _convert_color_to_rgb(color).shape == (2, 3, 3)
    assert _reshape_vendor_array(depth, channels=None).shape == (2, 3)
    assert _reshape_vendor_array(xyz, channels=3).shape == (2, 3, 3)


def test_vendor_frames_expose_ids_but_not_device_timestamps() -> None:
    frame_2d = sdk.Frame2D()
    frame_3d = sdk.Frame3D()

    assert frame_2d.frame_id() == 0
    assert frame_3d.frame_id() == 0
    assert not hasattr(frame_2d, "timestamp")
    assert not hasattr(frame_3d, "timestamp")


def test_vendor_calibration_layout_matches_adapter_contract() -> None:
    intrinsics = sdk.CameraIntrinsics()
    resolutions = sdk.CameraResolutions()

    assert all(hasattr(intrinsics.texture.camera_matrix, name) for name in ("fx", "fy", "cx", "cy"))
    assert all(
        hasattr(intrinsics.texture.camera_distortion, name)
        for name in ("k1", "k2", "p1", "p2", "k3")
    )
    assert np.asarray(intrinsics.depth_to_texture.rotation).shape == (3, 3)
    assert np.asarray(intrinsics.depth_to_texture.translation).shape == (3,)
    assert all(hasattr(resolutions.texture, name) for name in ("width", "height"))
    assert all(hasattr(resolutions.depth, name) for name in ("width", "height"))
