"""Deterministic NumPy point-cloud preprocessing for offline development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import PointCloudQualityReport, PreprocessedScene, SceneFrame


@dataclass(frozen=True)
class PreprocessConfig:
    """Measured workspace and conservative quality gates.

    All distances are metres in the AUBO base frame.  The workspace and table
    height intentionally have no guessed defaults: they must come from a saved
    calibration or an explicitly constructed synthetic test scene.
    """

    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    table_height_m: float
    table_clearance_m: float = 0.002
    voxel_size_m: float = 0.003
    outlier_radius_m: float = 0.0
    outlier_min_neighbors: int = 0
    min_finite_ratio: float = 0.80
    min_retained_points: int = 80

    def __post_init__(self) -> None:
        low = np.asarray(self.workspace_min, dtype=np.float64)
        high = np.asarray(self.workspace_max, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
            raise ValueError("workspace_max must be greater than workspace_min on every axis")
        if any(
            value < 0 for value in (self.table_clearance_m, self.voxel_size_m, self.outlier_radius_m)
        ):
            raise ValueError("table clearance, voxel size and outlier radius must be non-negative")
        if self.outlier_min_neighbors < 0:
            raise ValueError("outlier_min_neighbors must be non-negative")
        if (self.outlier_radius_m == 0) != (self.outlier_min_neighbors == 0):
            raise ValueError("outlier radius and minimum neighbors must be enabled or disabled together")
        if not 0.0 <= self.min_finite_ratio <= 1.0:
            raise ValueError("min_finite_ratio must be in [0, 1]")
        if self.min_retained_points < 1:
            raise ValueError("min_retained_points must be positive")


class PointCloudPreprocessor:
    """Transform, crop, remove the table plane and voxel-downsample a scene."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config

    def process(self, frame: SceneFrame) -> PreprocessedScene:
        points, coordinate_frame = self._coordinates(frame)
        colors = frame.points_rgb
        raw_count = len(points)

        finite_mask = np.all(np.isfinite(points), axis=1)
        finite_points = points[finite_mask]
        finite_colors = colors[finite_mask] if colors is not None else None

        low = np.asarray(self.config.workspace_min, dtype=np.float64)
        high = np.asarray(self.config.workspace_max, dtype=np.float64)
        workspace_mask = np.all((finite_points >= low) & (finite_points <= high), axis=1)
        workspace_points = finite_points[workspace_mask]
        workspace_colors = finite_colors[workspace_mask] if finite_colors is not None else None

        above_table = workspace_points[:, 2] >= (
            self.config.table_height_m + self.config.table_clearance_m
        )
        retained = workspace_points[above_table]
        retained_colors = workspace_colors[above_table] if workspace_colors is not None else None
        retained, retained_colors = self._voxel_downsample(retained, retained_colors)
        retained, retained_colors = self._remove_radius_outliers(retained, retained_colors)

        finite_ratio = len(finite_points) / raw_count if raw_count else 0.0
        retained_ratio = len(retained) / raw_count if raw_count else 0.0
        warnings: list[str] = []
        if coordinate_frame != "base":
            warnings.append("missing_base_calibration")
        if finite_ratio < self.config.min_finite_ratio:
            warnings.append("low_finite_point_ratio")
        if len(retained) < self.config.min_retained_points:
            warnings.append("too_few_retained_points")

        quality = PointCloudQualityReport(
            raw_points=raw_count,
            finite_points=len(finite_points),
            workspace_points=len(workspace_points),
            retained_points=len(retained),
            finite_ratio=finite_ratio,
            retained_ratio=retained_ratio,
            warnings=tuple(warnings),
        )
        return PreprocessedScene(
            source=frame,
            points_xyz=retained,
            points_rgb=retained_colors,
            coordinate_frame=coordinate_frame,
            quality=quality,
            workspace_min=low,
            workspace_max=high,
        )

    @staticmethod
    def _coordinates(frame: SceneFrame) -> tuple[np.ndarray, str]:
        if frame.frame_id == "base" or frame.T_base_camera is not None:
            return frame.points_in_base(), "base"
        return frame.points_xyz.copy(), frame.frame_id

    def _voxel_downsample(
        self, points: np.ndarray, colors: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not len(points) or self.config.voxel_size_m == 0:
            return points, colors
        keys = np.floor(points / self.config.voxel_size_m).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = np.bincount(inverse)
        reduced = np.column_stack(
            [np.bincount(inverse, weights=points[:, axis]) / count for axis in range(3)]
        )
        if colors is None:
            return reduced, None
        reduced_colors = np.column_stack(
            [np.bincount(inverse, weights=colors[:, axis]) / count for axis in range(3)]
        )
        return reduced, reduced_colors

    def _remove_radius_outliers(
        self, points: np.ndarray, colors: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        radius = self.config.outlier_radius_m
        minimum = self.config.outlier_min_neighbors
        if not len(points) or radius == 0 or minimum == 0:
            return points, colors

        keys = np.floor(points / radius).astype(np.int64)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, key in enumerate(keys):
            buckets.setdefault(tuple(int(value) for value in key), []).append(index)

        keep = np.zeros(len(points), dtype=bool)
        offsets = (-1, 0, 1)
        radius_squared = radius * radius
        for index, key in enumerate(keys):
            neighbors: list[int] = []
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        neighbors.extend(
                            buckets.get((int(key[0] + dx), int(key[1] + dy), int(key[2] + dz)), ())
                        )
            neighbor_points = points[neighbors]
            distances_squared = np.sum((neighbor_points - points[index]) ** 2, axis=1)
            # Subtract the query point itself from the count.
            keep[index] = int(np.count_nonzero(distances_squared <= radius_squared)) - 1 >= minimum
        return points[keep], colors[keep] if colors is not None else None
