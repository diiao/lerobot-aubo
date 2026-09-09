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

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.mech_mind import MechMindCamera, MechMindCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


class _FakeStatus:
    def __init__(self, ok: bool = True, error_code: int = -6):
        self.ok = ok
        self.error_code = error_code
        self.error_description = "fake-status"

    def is_ok(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return "fake-status"


class _FakeImage:
    def __init__(self, array: np.ndarray):
        self.array = array

    def data(self) -> np.ndarray:
        return self.array

    def width(self) -> int:
        return self.array.shape[1]

    def height(self) -> int:
        return self.array.shape[0]


class _FakeFrame2D:
    def __init__(self):
        self.color = _FakeImage(
            np.array(
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                ],
                dtype=np.uint8,
            )
        )

    def get_color_image(self) -> _FakeImage:
        return self.color


class _FakeFrame3D:
    def __init__(self):
        depth_mm = np.array([[500.0, 0.0], [750.0, 1000.0]], dtype=np.float32)
        xyz_mm = np.zeros((2, 2, 3), dtype=np.float32)
        xyz_mm[..., 2] = depth_mm
        self.depth = _FakeImage(depth_mm)
        self.xyz = _FakeImage(xyz_mm)

    def get_depth_map(self) -> _FakeImage:
        return self.depth

    def get_untextured_point_cloud(self) -> _FakeImage:
        return self.xyz


class _FakeFrame2DAnd3D:
    def __init__(self):
        self.two_d = _FakeFrame2D()
        self.three_d = _FakeFrame3D()

    def frame_2d(self) -> _FakeFrame2D:
        return self.two_d

    def frame_3d(self) -> _FakeFrame3D:
        return self.three_d


class _FakeCamera:
    next_connect_ok = True
    next_capture_ok = True
    instances: list[_FakeCamera] = []

    def __init__(self):
        self.connected = False
        self.connect_args: tuple[str, int] | None = None
        self.capture_timeout_ms: int | None = None
        self.__class__.instances.append(self)

    def connect(self, ip_address: str, timeout_ms: int) -> _FakeStatus:
        self.connect_args = (ip_address, timeout_ms)
        self.connected = self.next_connect_ok
        return _FakeStatus(self.next_connect_ok)

    def capture_2d_and_3d(self, _frame: object, timeout_ms: int) -> _FakeStatus:
        self.capture_timeout_ms = timeout_ms
        return _FakeStatus(self.next_capture_ok)

    def capture_2d(self, _frame: object, timeout_ms: int) -> _FakeStatus:
        self.capture_timeout_ms = timeout_ms
        return _FakeStatus(self.next_capture_ok)

    def disconnect(self) -> None:
        self.connected = False


@pytest.fixture(name="fake_sdk")
def fixture_fake_sdk() -> SimpleNamespace:
    _FakeCamera.next_connect_ok = True
    _FakeCamera.next_capture_ok = True
    _FakeCamera.instances.clear()
    return SimpleNamespace(
        Camera=_FakeCamera,
        Frame2D=_FakeFrame2D,
        Frame2DAnd3D=_FakeFrame2DAnd3D,
    )


def _camera(fake_sdk: SimpleNamespace, *, use_depth: bool = True) -> MechMindCamera:
    config = MechMindCameraConfig(
        ip_address="192.0.2.10",
        use_depth=use_depth,
        connect_timeout_ms=123,
        capture_timeout_ms=456,
    )
    return MechMindCamera(config, _sdk=fake_sdk)


def test_camera_implements_lerobot_interface_and_connects_by_fixed_ip(fake_sdk: SimpleNamespace) -> None:
    camera = _camera(fake_sdk)
    camera.connect(warmup=False)

    assert camera.is_connected
    assert _FakeCamera.instances[-1].connect_args == ("192.0.2.10", 123)
    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_generic_camera_factory_constructs_without_importing_vendor_sdk() -> None:
    config = MechMindCameraConfig(ip_address="192.0.2.10")

    cameras = make_cameras_from_configs({"wrist": config})

    assert isinstance(cameras["wrist"], MechMindCamera)
    assert not cameras["wrist"].is_connected


def test_joint_rgbd_capture_converts_bgr_and_millimeters(fake_sdk: SimpleNamespace) -> None:
    camera = _camera(fake_sdk)
    camera.connect(warmup=False)

    frame = camera.read_rgbd()

    assert frame.rgb.shape == (2, 3, 3)
    np.testing.assert_array_equal(frame.rgb[0, 0], [3, 2, 1])
    assert frame.depth_m.shape == (2, 2)
    assert frame.xyz_m.shape == (2, 2, 3)
    assert frame.depth_m[0, 0] == pytest.approx(0.5)
    assert not frame.depth_valid[0, 1]
    assert np.isnan(frame.depth_m[0, 1])
    assert frame.device_timestamp_s is None
    assert frame.device_clock_id is None
    assert _FakeCamera.instances[-1].capture_timeout_ms == 456


def test_rgb_only_capture_and_latest_frame(fake_sdk: SimpleNamespace) -> None:
    camera = _camera(fake_sdk, use_depth=False)
    camera.connect(warmup=False)

    image = camera.async_read(timeout_ms=12.5)

    assert image.shape == (2, 3, 3)
    assert camera.read_latest().shape == image.shape
    assert _FakeCamera.instances[-1].capture_timeout_ms == 13


def test_read_latest_rejects_missing_or_stale_frame(fake_sdk: SimpleNamespace) -> None:
    camera = _camera(fake_sdk)
    camera.connect(warmup=False)

    with pytest.raises(RuntimeError, match="has not captured"):
        camera.read_latest()
    camera.read_rgbd()
    with pytest.raises(TimeoutError, match="frame is"):
        camera.read_latest(max_age_ms=0)


def test_capture_failure_never_returns_a_frame(fake_sdk: SimpleNamespace) -> None:
    camera = _camera(fake_sdk)
    camera.connect(warmup=False)
    _FakeCamera.next_capture_ok = False

    with pytest.raises(RuntimeError, match="capture_2d_and_3d failed"):
        camera.read_rgbd()


def test_vendor_timeout_is_reported_separately(fake_sdk: SimpleNamespace) -> None:
    camera = _camera(fake_sdk)
    camera.connect(warmup=False)
    fake_camera = _FakeCamera.instances[-1]

    def timeout(_frame: object, timeout_ms: int) -> _FakeStatus:
        fake_camera.capture_timeout_ms = timeout_ms
        return _FakeStatus(False, error_code=-9)

    fake_camera.capture_2d_and_3d = timeout
    with pytest.raises(TimeoutError, match="code -9"):
        camera.read_rgbd(timeout_ms=25)


def test_disconnect_allows_explicit_reconnect(fake_sdk: SimpleNamespace) -> None:
    camera = _camera(fake_sdk)

    with pytest.raises(DeviceNotConnectedError):
        camera.read_rgbd()
    camera.connect(warmup=False)
    camera.disconnect()
    assert not camera.is_connected
    camera.connect(warmup=False)
    assert camera.is_connected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ip_address": "not-an-ip"}, "valid IPv4"),
        ({"ip_address": "::1"}, "valid IPv4"),
        ({"connect_timeout_ms": 0}, "positive integer"),
        ({"capture_timeout_ms": True}, "positive integer"),
        ({"use_depth": 1}, "use_depth must be bool"),
    ],
)
def test_config_rejects_unsafe_or_ambiguous_values(changes: dict[str, object], message: str) -> None:
    values = {"ip_address": "192.0.2.10"}
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        MechMindCameraConfig(**values)
