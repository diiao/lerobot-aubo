"""Read-only NPZ scene replay adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import CameraIntrinsics, SceneFrame


class NpzSceneSensor:
    """Replay immutable captures without importing or connecting a camera SDK."""

    def __init__(self, paths: list[str | Path], *, repeat: bool = False) -> None:
        if not paths:
            raise ValueError("at least one NPZ path is required")
        self.paths = tuple(Path(path) for path in paths)
        self.repeat = repeat
        self._index = 0

    def capture(self) -> SceneFrame:
        if self._index >= len(self.paths):
            if not self.repeat:
                raise StopIteration("scene replay exhausted")
            self._index = 0
        path = self.paths[self._index]
        self._index += 1
        with np.load(path, allow_pickle=False) as data:
            required = {"points_xyz", "timestamp_s", "frame_id"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{path} is missing required arrays: {missing}")
            intrinsics = None
            if "intrinsics" in data:
                values = np.asarray(data["intrinsics"], dtype=float).reshape(-1)
                if len(values) != 6:
                    raise ValueError("intrinsics must contain width,height,fx,fy,cx,cy")
                intrinsics = CameraIntrinsics(int(values[0]), int(values[1]), *values[2:])
            return SceneFrame(
                points_xyz=data["points_xyz"],
                timestamp_s=float(np.asarray(data["timestamp_s"]).item()),
                frame_id=str(np.asarray(data["frame_id"]).item()),
                color=data["color"] if "color" in data else None,
                depth_m=data["depth_m"] if "depth_m" in data else None,
                points_rgb=data["points_rgb"] if "points_rgb" in data else None,
                intrinsics=intrinsics,
                T_base_camera=data["T_base_camera"] if "T_base_camera" in data else None,
                metadata={"replay_path": str(path)},
            )
