"""Typed, serializable domain objects for the bamboo-sorting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]


def _float_array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _unit_vector(value: Any, name: str) -> FloatArray:
    vector = _float_array(value, (3,), name)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite non-zero vector")
    return vector / norm


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")

    @property
    def matrix(self) -> FloatArray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SceneFrame:
    """One synchronized RGB-D/point-cloud capture.

    ``points_xyz`` is expressed in ``frame_id``.  ``T_base_camera`` maps that
    coordinate frame into the AUBO base frame; it may be absent for uncalibrated
    offline captures, but such captures cannot become execution candidates.
    """

    points_xyz: FloatArray
    timestamp_s: float
    frame_id: str
    color: NDArray[Any] | None = None
    depth_m: FloatArray | None = None
    points_rgb: FloatArray | None = None
    intrinsics: CameraIntrinsics | None = None
    T_base_camera: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}")
        object.__setattr__(self, "points_xyz", points)
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")

        if self.points_rgb is not None:
            colors = np.asarray(self.points_rgb, dtype=np.float64)
            if colors.shape != points.shape:
                raise ValueError("points_rgb must have the same shape as points_xyz")
            object.__setattr__(self, "points_rgb", colors)

        if self.depth_m is not None:
            depth = np.asarray(self.depth_m, dtype=np.float64)
            if depth.ndim != 2:
                raise ValueError("depth_m must be a two-dimensional array")
            object.__setattr__(self, "depth_m", depth)

        if self.T_base_camera is not None:
            transform = _float_array(self.T_base_camera, (4, 4), "T_base_camera")
            if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
                raise ValueError("T_base_camera must be a homogeneous transform")
            object.__setattr__(self, "T_base_camera", transform)

    def points_in_base(self) -> FloatArray:
        if self.frame_id == "base":
            return self.points_xyz.copy()
        if self.T_base_camera is None:
            raise ValueError("T_base_camera is required to transform this frame into base coordinates")
        homogeneous = np.column_stack((self.points_xyz, np.ones(len(self.points_xyz))))
        return (self.T_base_camera @ homogeneous.T).T[:, :3]


@dataclass(frozen=True)
class PointCloudQualityReport:
    raw_points: int
    finite_points: int
    workspace_points: int
    retained_points: int
    finite_ratio: float
    retained_ratio: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreprocessedScene:
    source: SceneFrame
    points_xyz: FloatArray
    points_rgb: FloatArray | None
    coordinate_frame: str
    quality: PointCloudQualityReport
    workspace_min: FloatArray
    workspace_max: FloatArray


class SegmentRelation(str, Enum):
    CROSSING = "crossing"
    NEAR_PARALLEL = "near_parallel"
    OVERLAP = "overlap"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VisibleSegment:
    segment_id: str
    axis_unit: FloatArray
    endpoint_start: FloatArray
    endpoint_end: FloatArray
    width_m: float
    mean_height_m: float
    completeness: float
    clearance_m: float
    point_count: int
    near_intersection: bool = False
    near_workspace_edge: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_unit", _unit_vector(self.axis_unit, "axis_unit"))
        object.__setattr__(self, "endpoint_start", _float_array(self.endpoint_start, (3,), "endpoint_start"))
        object.__setattr__(self, "endpoint_end", _float_array(self.endpoint_end, (3,), "endpoint_end"))
        if self.length_m <= 0:
            raise ValueError("visible segment endpoints must be distinct")
        if self.width_m < 0 or self.clearance_m < 0:
            raise ValueError("width_m and clearance_m must be non-negative")
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("completeness must be in [0, 1]")
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.endpoint_end - self.endpoint_start))

    @property
    def midpoint(self) -> FloatArray:
        return (self.endpoint_start + self.endpoint_end) / 2.0


@dataclass(frozen=True)
class IntersectionHypothesis:
    first_segment_id: str
    second_segment_id: str
    position_base: FloatArray
    relation: SegmentRelation
    confidence: float
    separation_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_base", _float_array(self.position_base, (3,), "position_base"))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.separation_m < 0:
            raise ValueError("separation_m must be non-negative")


class CandidateStatus(str, Enum):
    REJECTED = "rejected"
    GEOMETRY_VALID = "geometry_valid"
    EXECUTION_VALIDATED = "execution_validated"


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    segment_id: str
    grasp_frame_base: FloatArray
    pregrasp_frame_base: FloatArray
    approach_direction_base: FloatArray
    lift_direction_base: FloatArray
    status: CandidateStatus
    score_components: dict[str, float]
    hard_constraint_results: dict[str, bool]
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grasp_frame_base",
            _float_array(self.grasp_frame_base, (4, 4), "grasp_frame_base"),
        )
        object.__setattr__(
            self,
            "pregrasp_frame_base",
            _float_array(self.pregrasp_frame_base, (4, 4), "pregrasp_frame_base"),
        )
        object.__setattr__(
            self,
            "approach_direction_base",
            _unit_vector(self.approach_direction_base, "approach_direction_base"),
        )
        object.__setattr__(
            self,
            "lift_direction_base",
            _unit_vector(self.lift_direction_base, "lift_direction_base"),
        )
        if self.status == CandidateStatus.REJECTED and not self.rejection_reasons:
            raise ValueError("rejected candidates must include at least one reason")
        if self.status != CandidateStatus.REJECTED and self.rejection_reasons:
            raise ValueError("valid candidates cannot include rejection reasons")
        if not self.hard_constraint_results:
            raise ValueError("candidate must record its hard constraint results")
        if self.status == CandidateStatus.REJECTED and all(self.hard_constraint_results.values()):
            raise ValueError("a rejected candidate must have at least one failed hard constraint")
        if self.status != CandidateStatus.REJECTED and not all(self.hard_constraint_results.values()):
            raise ValueError("a valid candidate cannot have a failed hard constraint")

    @property
    def score(self) -> float:
        return float(sum(self.score_components.values()))


class GraspOutcomeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class GraspOutcome:
    status: GraspOutcomeStatus
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    human_intervention: bool = False
    safety_stop: bool = False

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("grasp outcome reason must not be empty")
        if self.status == GraspOutcomeStatus.SUCCESS and not self.evidence:
            raise ValueError("successful grasp outcomes require independent verification evidence")
