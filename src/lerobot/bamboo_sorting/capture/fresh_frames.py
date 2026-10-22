"""Fresh dual-camera observation helper for joint capture.

The retired C0 batch CLI is no longer provided; this module opens no devices on import.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

from .c0_batch_capture import C0BatchCaptureError
from .c0_smoke_capture import C0_SMOKE_IMAGE_KEYS


def read_fresh_c0_observation(
    *,
    read_observation: Callable[[], dict[str, Any]],
    read_timestamps: Callable[[], object],
    previous_camera_timestamps: dict[str, float],
    timeout_s: float = 0.2,
    poll_interval_s: float = 0.002,
    monotonic: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Read until both formal cameras have a frame newer than the last saved read."""

    deadline = monotonic() + timeout_s
    stale_streams: list[str] = []
    while True:
        observation = read_observation()
        raw_timestamps = read_timestamps()
        current: dict[str, float] = {}
        stale_streams = []
        for stream in C0_SMOKE_IMAGE_KEYS:
            value = raw_timestamps.get(stream) if isinstance(raw_timestamps, Mapping) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                stale_streams.append(stream)
                continue
            current[stream] = float(value)
            previous = previous_camera_timestamps.get(stream)
            if previous is not None and current[stream] <= previous:
                stale_streams.append(stream)

        if not stale_streams:
            previous_camera_timestamps.update(current)
            return observation
        if monotonic() >= deadline:
            joined = ", ".join(stale_streams)
            raise C0BatchCaptureError(
                f"no fresh C0 camera frame within {timeout_s:.3f}s: {joined}"
            )
        sleep(poll_interval_s)
