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

"""Sensor timestamp records retained by the current joint capture journal."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Final

from .rgb_gate import FIXED_RGB_STREAMS, ROBOT_STATE

C0_SENSOR_TIMESTAMP_SCHEMA_VERSION: Final = "C0SensorTimestampV1"

C0_SENSOR_STREAMS: Final = (*FIXED_RGB_STREAMS, ROBOT_STATE)

C0_SYNC_METHODS: Final = frozenset({"host_receive", "device_to_host_calibrated"})


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return converted


@dataclass(frozen=True)
class C0SensorTimestampV1:
    """Timing evidence for one C0 sample from one frozen sensor stream.

    The class is intentionally independent of the legacy three-RGB
    ``EmbodiedObservationV1``. It permits only the two CameraSetV2 RGB streams
    and robot state, so wrist RGB, depth, ``handeye`` and ``fixed`` cannot enter
    C0 through this contract.
    """

    stream_name: str
    sync_timestamp_s: float
    host_receive_monotonic_s: float
    sync_method: str
    device_timestamp_s: float | None = None
    device_clock_id: str | None = None
    schema_version: str = C0_SENSOR_TIMESTAMP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != C0_SENSOR_TIMESTAMP_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_SENSOR_TIMESTAMP_SCHEMA_VERSION!r}"
            )
        if self.stream_name not in C0_SENSOR_STREAMS:
            raise ValueError(f"stream_name must be one of {C0_SENSOR_STREAMS}")

        sync_timestamp = _finite_non_negative("sync_timestamp_s", self.sync_timestamp_s)
        host_timestamp = _finite_non_negative(
            "host_receive_monotonic_s", self.host_receive_monotonic_s
        )
        object.__setattr__(self, "sync_timestamp_s", sync_timestamp)
        object.__setattr__(self, "host_receive_monotonic_s", host_timestamp)

        if self.sync_method not in C0_SYNC_METHODS:
            raise ValueError(f"sync_method must be one of {sorted(C0_SYNC_METHODS)}")

        has_device_time = self.device_timestamp_s is not None
        has_device_clock = self.device_clock_id is not None
        if has_device_time != has_device_clock:
            raise ValueError(
                "device_timestamp_s and device_clock_id must be provided together"
            )
        if has_device_time:
            device_timestamp = _finite_non_negative(
                "device_timestamp_s", self.device_timestamp_s
            )
            _require_nonempty_string("device_clock_id", self.device_clock_id)
            object.__setattr__(self, "device_timestamp_s", device_timestamp)

        if self.sync_method == "host_receive" and not math.isclose(
            sync_timestamp, host_timestamp, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "host_receive sync_timestamp_s must equal host_receive_monotonic_s"
            )
        if self.sync_method == "device_to_host_calibrated" and not has_device_time:
            raise ValueError(
                "device_to_host_calibrated requires device_timestamp_s and device_clock_id"
            )

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible timing record."""

        return copy.deepcopy(asdict(self))
