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

"""Timestamp schema tests for current joint capture."""

import json
import subprocess
import sys

import pytest

from lerobot.bamboo_sorting.c0_capture_contract import C0_SENSOR_TIMESTAMP_SCHEMA_VERSION, C0SensorTimestampV1


def test_valid_sensor_timestamp_modes() -> None:
    host = C0SensorTimestampV1(
        schema_version=C0_SENSOR_TIMESTAMP_SCHEMA_VERSION,
        stream_name="global_rgb",
        sync_timestamp_s=1.25,
        host_receive_monotonic_s=1.25,
        sync_method="host_receive",
    )
    calibrated = C0SensorTimestampV1(
        stream_name="robot_state",
        sync_timestamp_s=2.0,
        host_receive_monotonic_s=2.01,
        sync_method="device_to_host_calibrated",
        device_timestamp_s=20.0,
        device_clock_id="aubo-rtde-clock",
    )

    assert host.to_manifest_record()["schema_version"] == C0_SENSOR_TIMESTAMP_SCHEMA_VERSION
    assert calibrated.device_timestamp_s == 20.0


@pytest.mark.parametrize("stream_name", ["wrist_rgb", "wrist_depth_m", "depth", "handeye", "fixed"])
def test_timestamp_rejects_legacy_or_depth_streams(stream_name: str) -> None:
    with pytest.raises(ValueError, match="stream_name must be one of"):
        C0SensorTimestampV1(
            stream_name=stream_name,
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.0,
            sync_method="host_receive",
        )


@pytest.mark.parametrize("field", ["sync_timestamp_s", "host_receive_monotonic_s"])
@pytest.mark.parametrize("bad_value", [True, -0.1, float("nan"), float("inf")])
def test_timestamp_rejects_invalid_common_times(field: str, bad_value: object) -> None:
    values = {
        "stream_name": "grasp_rgb",
        "sync_timestamp_s": 1.0,
        "host_receive_monotonic_s": 1.0,
        "sync_method": "host_receive",
    }
    values[field] = bad_value
    with pytest.raises(ValueError, match=field):
        C0SensorTimestampV1(**values)


@pytest.mark.parametrize("bad_value", [True, -0.1, float("nan"), float("inf")])
def test_timestamp_rejects_invalid_device_time(bad_value: object) -> None:
    with pytest.raises(ValueError, match="device_timestamp_s"):
        C0SensorTimestampV1(
            stream_name="robot_state",
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.1,
            sync_method="device_to_host_calibrated",
            device_timestamp_s=bad_value,
            device_clock_id="clock-1",
        )


def test_host_receive_requires_equal_timestamp() -> None:
    with pytest.raises(ValueError, match="must equal"):
        C0SensorTimestampV1(
            stream_name="global_rgb",
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.1,
            sync_method="host_receive",
        )


@pytest.mark.parametrize(
    ("device_timestamp_s", "device_clock_id"),
    [(1.0, None), (None, "clock-1"), (1.0, "")],
)
def test_device_timestamp_and_clock_id_are_paired(
    device_timestamp_s: float | None, device_clock_id: str | None
) -> None:
    with pytest.raises(ValueError, match="device_"):
        C0SensorTimestampV1(
            stream_name="robot_state",
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.1,
            sync_method="device_to_host_calibrated",
            device_timestamp_s=device_timestamp_s,
            device_clock_id=device_clock_id,
        )


_IMPORT_PROBE = r"""
import json
import os
import sys
import threading

import lerobot.bamboo_sorting.c0_capture_contract  # noqa: F401

forbidden_modules = [name for name in sys.modules if name.split(".")[0] in {"cv2", "pyaubo_sdk"}]
extra_threads = [
    thread.name for thread in threading.enumerate() if thread is not threading.main_thread()
]
sockets = []
for fd in os.listdir("/proc/self/fd"):
    path = f"/proc/self/fd/{fd}"
    if os.path.islink(path) and "socket" in os.readlink(path):
        sockets.append(fd)
print(
    json.dumps(
        {
            "forbidden_modules": forbidden_modules,
            "extra_threads": extra_threads,
            "sockets": sockets,
        }
    )
)
"""


def test_importing_c0_contract_touches_no_hardware_network_or_threads() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []
