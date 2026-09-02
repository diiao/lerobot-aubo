"""Visible line-segment extraction for sparse bamboo-strip scenes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .types import IntersectionHypothesis, PreprocessedScene, SegmentRelation, VisibleSegment


@dataclass(frozen=True)
class SegmentExtractionConfig:
    distance_threshold_m: float
    min_inlier_points: int
    min_visible_length_m: float
    expected_length_m: float
    max_segments: int = 20
    ransac_iterations: int = 250
    intersection_distance_m: float = 0.012
    workspace_edge_margin_m: float = 0.015
    random_seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.distance_threshold_m,
            self.min_visible_length_m,
            self.expected_length_m,
            self.intersection_distance_m,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("segment distances and expected length must be positive")
        if self.min_inlier_points < 2 or self.max_segments < 1 or self.ransac_iterations < 1:
            raise ValueError("invalid segment count or RANSAC iteration configuration")


class RansacSegmentExtractor:
    """Extract visible straight portions; it does not hallucinate occluded ends."""

    def __init__(self, config: SegmentExtractionConfig) -> None:
        self.config = config

    def extract(
        self, scene: PreprocessedScene
    ) -> tuple[tuple[VisibleSegment, ...], tuple[IntersectionHypothesis, ...]]:
        rng = np.random.default_rng(self.config.random_seed)
        remaining = scene.points_xyz.copy()
        segments: list[VisibleSegment] = []

        while len(remaining) >= self.config.min_inlier_points and len(segments) < self.config.max_segments:
            inliers = self._best_line_inliers(remaining, rng)
            if inliers is None or int(np.count_nonzero(inliers)) < self.config.min_inlier_points:
                break
            cluster = remaining[inliers]
            segment = self._fit_segment(cluster, len(segments), scene)
            remaining = remaining[~inliers]
            if segment is not None:
                segments.append(segment)

        intersections = self._intersections(segments)
        involved = {
            segment_id
            for item in intersections
            for segment_id in (item.first_segment_id, item.second_segment_id)
        }
        marked = tuple(
            replace(segment, near_intersection=segment.segment_id in involved) for segment in segments
        )
        return marked, intersections

    def _best_line_inliers(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        best: np.ndarray | None = None
        best_count = 0
        for _ in range(self.config.ransac_iterations):
            indices = rng.choice(len(points), size=2, replace=False)
            direction = points[indices[1]] - points[indices[0]]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                continue
            direction /= norm
            delta = points - points[indices[0]]
            distances = np.linalg.norm(np.cross(delta, direction), axis=1)
            mask = distances <= self.config.distance_threshold_m
            count = int(np.count_nonzero(mask))
            if count > best_count:
                best, best_count = mask, count
        return best

    def _fit_segment(
        self, points: np.ndarray, index: int, scene: PreprocessedScene
    ) -> VisibleSegment | None:
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center, full_matrices=False)
        axis = vh[0]
        if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
            axis = -axis
        projection = (points - center) @ axis
        start = center + axis * float(np.min(projection))
        end = center + axis * float(np.max(projection))
        length = float(np.linalg.norm(end - start))
        if length < self.config.min_visible_length_m:
            return None

        radial = np.linalg.norm((points - center) - np.outer(projection, axis), axis=1)
        width = 2.0 * float(np.quantile(radial, 0.90))
        completeness = min(1.0, length / self.config.expected_length_m)
        # Z clearance is governed by table/pre-grasp checks.  Here clearance
        # means horizontal room to the calibrated workspace boundary.
        start_edge = np.minimum(
            start[:2] - scene.workspace_min[:2], scene.workspace_max[:2] - start[:2]
        )
        end_edge = np.minimum(end[:2] - scene.workspace_min[:2], scene.workspace_max[:2] - end[:2])
        boundary_clearance = max(0.0, float(np.min(np.concatenate((start_edge, end_edge)))))
        near_edge = boundary_clearance < self.config.workspace_edge_margin_m
        return VisibleSegment(
            segment_id=f"segment-{index:03d}",
            axis_unit=axis,
            endpoint_start=start,
            endpoint_end=end,
            width_m=width,
            mean_height_m=float(np.mean(points[:, 2])),
            completeness=completeness,
            clearance_m=boundary_clearance,
            point_count=len(points),
            near_workspace_edge=near_edge,
        )

    def _intersections(self, segments: list[VisibleSegment]) -> tuple[IntersectionHypothesis, ...]:
        output: list[IntersectionHypothesis] = []
        for first_index, first in enumerate(segments):
            for second in segments[first_index + 1 :]:
                point_a, point_b = _closest_points(first, second)
                separation = float(np.linalg.norm(point_a - point_b))
                if separation > self.config.intersection_distance_m:
                    continue
                cosine = abs(float(np.dot(first.axis_unit, second.axis_unit)))
                if cosine > 0.95:
                    relation = SegmentRelation.NEAR_PARALLEL
                else:
                    relation = SegmentRelation.CROSSING
                output.append(
                    IntersectionHypothesis(
                        first_segment_id=first.segment_id,
                        second_segment_id=second.segment_id,
                        position_base=(point_a + point_b) / 2.0,
                        relation=relation,
                        confidence=max(0.0, 1.0 - separation / self.config.intersection_distance_m),
                        separation_m=separation,
                    )
                )
        return tuple(output)


def _closest_points(first: VisibleSegment, second: VisibleSegment) -> tuple[np.ndarray, np.ndarray]:
    """Return closest points on two finite 3-D segments."""
    p1, q1 = first.endpoint_start, first.endpoint_end
    p2, q2 = second.endpoint_start, second.endpoint_end
    d1, d2, offset = q1 - p1, q2 - p2, p1 - p2
    a, e = float(np.dot(d1, d1)), float(np.dot(d2, d2))
    b, c, f = float(np.dot(d1, d2)), float(np.dot(d1, offset)), float(np.dot(d2, offset))
    denominator = a * e - b * b
    s = np.clip((b * f - c * e) / denominator, 0.0, 1.0) if denominator > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0.0, 1.0)
    s = np.clip((b * t - c) / a, 0.0, 1.0)
    return p1 + s * d1, p2 + t * d2
