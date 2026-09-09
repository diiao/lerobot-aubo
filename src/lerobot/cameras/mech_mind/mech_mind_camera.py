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

"""LeRobot adapter for synchronous Mech-Eye 2D + 3D capture.

The vendor SDK is imported only when discovery or connection is explicitly
requested. Importing this module performs no network or hardware operation.
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from .configuration_mech_mind import MechMindCameraConfig

MILLIMETERS_PER_METER = 1000.0


@dataclass(frozen=True)
class MechMindRGBDFrame:
    """One jointly requested Mech-Eye frame in model-facing metric units."""

    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32] | None
    depth_valid: NDArray[np.bool_] | None
    xyz_m: NDArray[np.float32] | None
    capture_started_monotonic_s: float
    host_receive_monotonic_s: float
    device_timestamp_s: float | None = None
    device_clock_id: str | None = None

    @property
    def capture_latency_ms(self) -> float:
        return (self.host_receive_monotonic_s - self.capture_started_monotonic_s) * 1000.0


def _load_mecheye_sdk() -> Any:
    try:
        return importlib.import_module("mecheye.area_scan_3d_camera")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Mech-Eye SDK and the MechEyeAPI Python package are required only for hardware access. "
            "Install a version compatible with the camera firmware before connecting."
        ) from exc


def _require_status_ok(status: object, operation: str) -> None:
    is_ok = getattr(status, "is_ok", None)
    if not callable(is_ok):
        raise RuntimeError(f"Mech-Eye {operation} returned an invalid status object")
    if not is_ok():
        error_code = getattr(status, "error_code", getattr(status, "errorCode", None))
        description = getattr(
            status,
            "error_description",
            getattr(status, "errorDescription", str(status)),
        )
        message = f"Mech-Eye {operation} failed with code {error_code}: {description}"
        try:
            is_timeout = int(error_code) == -9
        except (TypeError, ValueError):
            is_timeout = "TIMEOUT" in str(error_code).upper()
        if is_timeout:
            raise TimeoutError(message)
        raise RuntimeError(message)


def _reshape_vendor_array(container: object, *, channels: int | None) -> NDArray[Any]:
    data_method = getattr(container, "data", None)
    width_method = getattr(container, "width", None)
    height_method = getattr(container, "height", None)
    if not callable(data_method) or not callable(width_method) or not callable(height_method):
        raise RuntimeError("Mech-Eye frame container lacks data/width/height accessors")

    width = int(width_method())
    height = int(height_method())
    if width <= 0 or height <= 0:
        raise RuntimeError("Mech-Eye returned a frame with invalid dimensions")
    array = np.asarray(data_method())

    if channels is None:
        if array.size != height * width:
            raise RuntimeError("Mech-Eye scalar image size does not match its reported dimensions")
        return array.reshape(height, width)

    if array.dtype.names:
        names = {name.lower(): name for name in array.dtype.names}
        if not all(axis in names for axis in ("x", "y", "z")):
            raise RuntimeError("Mech-Eye structured point cloud lacks x/y/z fields")
        array = np.stack([array[names[axis]] for axis in ("x", "y", "z")], axis=-1)
    if array.size != height * width * channels:
        raise RuntimeError("Mech-Eye image size does not match its reported dimensions")
    return array.reshape(height, width, channels)


def _convert_color_to_rgb(color_image: object) -> NDArray[np.uint8]:
    array = _reshape_vendor_array(color_image, channels=3)
    if array.dtype != np.uint8:
        raise RuntimeError("Mech-Eye color image must be uint8 BGR")
    return np.ascontiguousarray(array[..., ::-1])


def _convert_metric_depth(frame_3d: object) -> tuple[
    NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]
]:
    depth_map = frame_3d.get_depth_map()
    point_cloud = frame_3d.get_untextured_point_cloud()
    depth_mm = _reshape_vendor_array(depth_map, channels=None).astype(np.float32, copy=False)
    xyz_mm = _reshape_vendor_array(point_cloud, channels=3).astype(np.float32, copy=False)
    if xyz_mm.shape[:2] != depth_mm.shape:
        raise RuntimeError("Mech-Eye depth map and point cloud dimensions do not match")

    depth_m = depth_mm / MILLIMETERS_PER_METER
    xyz_m = xyz_mm / MILLIMETERS_PER_METER
    valid = np.isfinite(depth_m) & (depth_m > 0) & np.isfinite(xyz_m).all(axis=-1)
    depth_m = np.where(valid, depth_m, np.nan).astype(np.float32, copy=False)
    xyz_m = np.where(valid[..., None], xyz_m, np.nan).astype(np.float32, copy=False)
    return depth_m, valid, xyz_m


class MechMindCamera(Camera):
    """Mech-Eye camera with explicit timeouts and no implicit file writes."""

    def __init__(self, config: MechMindCameraConfig, *, _sdk: Any | None = None):
        super().__init__(config)
        self.config = config
        self._sdk = _sdk
        self._camera: Any | None = None
        self._latest_frame: MechMindRGBDFrame | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.ip_address})"

    @property
    def is_connected(self) -> bool:
        return self._camera is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        sdk = _load_mecheye_sdk()
        camera_infos = sdk.Camera.discover_cameras(5000)
        return [
            {
                "type": "mech_mind",
                "index": index,
                "vendor_info": repr(camera_info),
            }
            for index, camera_info in enumerate(camera_infos)
        ]

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")
        sdk = self._sdk if self._sdk is not None else _load_mecheye_sdk()
        camera = sdk.Camera()
        status = camera.connect(self.config.ip_address, self.config.connect_timeout_ms)
        _require_status_ok(status, "connect")
        self._sdk = sdk
        self._camera = camera
        try:
            if warmup:
                self.read_rgbd()
        except Exception:
            camera.disconnect()
            self._camera = None
            self._latest_frame = None
            raise

    def read(self) -> NDArray[Any]:
        return self.read_rgbd().rgb

    def read_rgbd(self, timeout_ms: int | None = None) -> MechMindRGBDFrame:
        if not self.is_connected or self._camera is None or self._sdk is None:
            raise DeviceNotConnectedError(f"{self} is not connected")
        timeout_ms = self.config.capture_timeout_ms if timeout_ms is None else timeout_ms
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")

        capture_started = time.monotonic()
        if self.config.use_depth:
            vendor_frame = self._sdk.Frame2DAnd3D()
            status = self._camera.capture_2d_and_3d(vendor_frame, timeout_ms)
            host_receive = time.monotonic()
            _require_status_ok(status, "capture_2d_and_3d")
            frame_2d = vendor_frame.frame_2d()
            frame_3d = vendor_frame.frame_3d()
            depth_m, depth_valid, xyz_m = _convert_metric_depth(frame_3d)
        else:
            frame_2d = self._sdk.Frame2D()
            status = self._camera.capture_2d(frame_2d, timeout_ms)
            host_receive = time.monotonic()
            _require_status_ok(status, "capture_2d")
            depth_m = None
            depth_valid = None
            xyz_m = None

        rgb = _convert_color_to_rgb(frame_2d.get_color_image())
        self._validate_or_set_rgb_dimensions(rgb)
        result = MechMindRGBDFrame(
            rgb=rgb,
            depth_m=depth_m,
            depth_valid=depth_valid,
            xyz_m=xyz_m,
            capture_started_monotonic_s=capture_started,
            host_receive_monotonic_s=host_receive,
        )
        self._latest_frame = result
        return result

    def async_read(self, timeout_ms: float = 10000) -> NDArray[Any]:
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, (int, float))
            or not math.isfinite(timeout_ms)
            or timeout_ms <= 0
        ):
            raise ValueError("timeout_ms must be finite and positive")
        return self.read_rgbd(timeout_ms=max(1, math.ceil(timeout_ms))).rgb

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or max_age_ms < 0:
            raise ValueError("max_age_ms must be a non-negative integer")
        if self._latest_frame is None:
            raise RuntimeError(f"{self} has not captured a frame")
        age_ms = (time.monotonic() - self._latest_frame.host_receive_monotonic_s) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(f"Latest {self} frame is {age_ms:.1f} ms old")
        return self._latest_frame.rgb.copy()

    def disconnect(self) -> None:
        if not self.is_connected or self._camera is None:
            raise DeviceNotConnectedError(f"{self} is not connected")
        camera = self._camera
        self._camera = None
        self._latest_frame = None
        camera.disconnect()

    def _validate_or_set_rgb_dimensions(self, rgb: NDArray[np.uint8]) -> None:
        height, width = rgb.shape[:2]
        if self.width is not None and self.width != width:
            raise RuntimeError(f"{self} returned width={width}, expected {self.width}")
        if self.height is not None and self.height != height:
            raise RuntimeError(f"{self} returned height={height}, expected {self.height}")
        self.width = width
        self.height = height
