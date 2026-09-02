"""Hard-filter-first grasp candidate generation and transparent scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .interfaces import MotionValidator
from .types import CandidateStatus, GraspCandidate, IntersectionHypothesis, PreprocessedScene, VisibleSegment


@dataclass(frozen=True)
class GraspPlannerConfig:
    min_graspable_length_m: float
    min_width_m: float
    max_width_m: float
    min_completeness: float
    min_clearance_m: float
    intersection_exclusion_radius_m: float
    pregrasp_distance_m: float
    grasp_height_offset_m: float = 0.0
    sample_fractions: tuple[float, ...] = (0.35, 0.50, 0.65)

    def __post_init__(self) -> None:
        if self.min_graspable_length_m <= 0 or self.max_width_m <= self.min_width_m:
            raise ValueError("invalid grasp length or width limits")
        if not 0.0 <= self.min_completeness <= 1.0:
            raise ValueError("min_completeness must be in [0, 1]")
        if any(value < 0 for value in (self.min_clearance_m, self.grasp_height_offset_m)):
            raise ValueError("clearance and grasp height offset must be non-negative")
        if self.intersection_exclusion_radius_m <= 0 or self.pregrasp_distance_m <= 0:
            raise ValueError("exclusion radius and pregrasp distance must be positive")
        if not self.sample_fractions or any(not 0.0 < value < 1.0 for value in self.sample_fractions):
            raise ValueError("sample_fractions must contain values strictly inside (0, 1)")


class GeometricGraspPlanner:
    """Produce task frames; a validator must approve them before execution."""

    def __init__(self, config: GraspPlannerConfig, validator: MotionValidator | None = None) -> None:
        self.config = config
        self.validator = validator

    def plan(
        self,
        scene: PreprocessedScene,
        segments: tuple[VisibleSegment, ...],
        intersections: tuple[IntersectionHypothesis, ...],
    ) -> tuple[GraspCandidate, ...]:
        candidates: list[GraspCandidate] = []
        for segment in segments:
            for fraction in self.config.sample_fractions:
                candidate = self._candidate(scene, segment, intersections, fraction)
                if candidate.status != CandidateStatus.REJECTED and self.validator is not None:
                    reasons = tuple(self.validator.rejection_reasons(candidate))
                    constraints = dict(candidate.hard_constraint_results)
                    constraints["motion_validation"] = not reasons
                    candidate = replace(
                        candidate,
                        status=(CandidateStatus.REJECTED if reasons else CandidateStatus.EXECUTION_VALIDATED),
                        hard_constraint_results=constraints,
                        rejection_reasons=reasons,
                    )
                candidates.append(candidate)
        return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))

    def _candidate(
        self,
        scene: PreprocessedScene,
        segment: VisibleSegment,
        intersections: tuple[IntersectionHypothesis, ...],
        fraction: float,
    ) -> GraspCandidate:
        position = segment.endpoint_start + fraction * (segment.endpoint_end - segment.endpoint_start)
        position = position.copy()
        position[2] += self.config.grasp_height_offset_m
        intersection_distance = self._nearest_intersection(position, segment.segment_id, intersections)
        constraints = {
            "base_calibrated": scene.coordinate_frame == "base",
            "point_cloud_reliable": not scene.quality.warnings,
            "visible_length": segment.length_m >= self.config.min_graspable_length_m,
            "width": self.config.min_width_m <= segment.width_m <= self.config.max_width_m,
            "completeness": segment.completeness >= self.config.min_completeness,
            "boundary_clearance": segment.clearance_m >= self.config.min_clearance_m,
            "workspace_boundary": not segment.near_workspace_edge,
            "intersection_clearance": intersection_distance >= self.config.intersection_exclusion_radius_m,
        }
        reasons: list[str] = []
        if not constraints["base_calibrated"]:
            reasons.append("missing_base_calibration")
        if not constraints["point_cloud_reliable"]:
            reasons.extend(f"point_cloud:{warning}" for warning in scene.quality.warnings)
        if not constraints["visible_length"]:
            reasons.append("visible_length_too_short")
        if not constraints["width"]:
            reasons.append("width_out_of_range")
        if not constraints["completeness"]:
            reasons.append("low_segment_completeness")
        if not constraints["boundary_clearance"]:
            reasons.append("insufficient_clearance")
        if not constraints["workspace_boundary"]:
            reasons.append("near_workspace_boundary")
        if not constraints["intersection_clearance"]:
            reasons.append("near_intersection")

        grasp_frame = self._task_frame(position, segment.axis_unit)
        approach = np.array([0.0, 0.0, -1.0])
        pregrasp = grasp_frame.copy()
        pregrasp[:3, 3] -= approach * self.config.pregrasp_distance_m
        score_components = {
            "visible_length": min(1.0, segment.length_m / self.config.min_graspable_length_m),
            "completeness": segment.completeness,
            "center_preference": 1.0 - 2.0 * abs(fraction - 0.5),
            "intersection_clearance": min(
                1.0, intersection_distance / self.config.intersection_exclusion_radius_m
            ),
            "hard_rejection_penalty": -10.0 * len(reasons),
        }
        return GraspCandidate(
            candidate_id=f"{segment.segment_id}-f{fraction:.2f}",
            segment_id=segment.segment_id,
            grasp_frame_base=grasp_frame,
            pregrasp_frame_base=pregrasp,
            approach_direction_base=approach,
            lift_direction_base=np.array([0.0, 0.0, 1.0]),
            status=CandidateStatus.REJECTED if reasons else CandidateStatus.GEOMETRY_VALID,
            score_components=score_components,
            hard_constraint_results=constraints,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _task_frame(position: np.ndarray, segment_axis: np.ndarray) -> np.ndarray:
        x_axis = segment_axis.copy()
        x_axis[2] = 0.0
        if np.linalg.norm(x_axis) < 1e-9:
            x_axis = np.array([1.0, 0.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.array([0.0, 0.0, -1.0])
        y_axis = np.cross(z_axis, x_axis)
        frame = np.eye(4)
        frame[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        frame[:3, 3] = position
        return frame

    @staticmethod
    def _nearest_intersection(
        position: np.ndarray,
        segment_id: str,
        intersections: tuple[IntersectionHypothesis, ...],
    ) -> float:
        distances = [
            float(np.linalg.norm(position - item.position_base))
            for item in intersections
            if segment_id in (item.first_segment_id, item.second_segment_id)
        ]
        return min(distances, default=float("inf"))
