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

"""Configuration for the Mech-Eye industrial 3D camera adapter."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from ..configs import CameraConfig


@CameraConfig.register_subclass("mech_mind")
@dataclass
class MechMindCameraConfig(CameraConfig):
    """Connection and capture limits; it does not change camera parameters."""

    ip_address: str = ""
    use_depth: bool = True
    connect_timeout_ms: int = 5000
    capture_timeout_ms: int = 10000

    def __post_init__(self) -> None:
        try:
            address = ipaddress.ip_address(self.ip_address)
        except ValueError as exc:
            raise ValueError("ip_address must be a valid IPv4 address") from exc
        if address.version != 4:
            raise ValueError("ip_address must be a valid IPv4 address")
        if not isinstance(self.use_depth, bool):
            raise ValueError("use_depth must be bool")
        for name in ("connect_timeout_ms", "capture_timeout_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fps is not None and (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or self.fps <= 0
        ):
            raise ValueError("fps must be a positive number when provided")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer when provided")
