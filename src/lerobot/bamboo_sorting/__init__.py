"""Hardware-neutral building blocks for AUBO bamboo-strip sorting.

The package deliberately contains no code that connects to a real robot or
camera.  Hardware adapters must be supplied explicitly by an application.
"""

from .gripper import CommandedGripperState, DigitalOutputPneumaticGripper
from .interfaces import GraspVerifier, MotionExecutor, MotionValidator, SceneSensor
from .metrics import ExperimentAllocation, ProportionEstimate, wilson_interval
from .pipeline import GeometricPlanningPipeline, PlanningResult
from .planner import GeometricGraspPlanner, GraspPlannerConfig
from .preprocessing import PointCloudPreprocessor, PreprocessConfig
from .replay import NpzSceneSensor
from .segments import RansacSegmentExtractor, SegmentExtractionConfig
from .state_machine import SortingCycleResult, SortingState, SortingTaskRunner
from .types import (
    CameraIntrinsics,
    CandidateStatus,
    GraspCandidate,
    GraspOutcome,
    GraspOutcomeStatus,
    IntersectionHypothesis,
    PointCloudQualityReport,
    PreprocessedScene,
    SceneFrame,
    SegmentRelation,
    VisibleSegment,
)

__all__ = [
    "CameraIntrinsics",
    "CandidateStatus",
    "CommandedGripperState",
    "DigitalOutputPneumaticGripper",
    "ExperimentAllocation",
    "GeometricGraspPlanner",
    "GeometricPlanningPipeline",
    "GraspCandidate",
    "GraspOutcome",
    "GraspOutcomeStatus",
    "GraspPlannerConfig",
    "GraspVerifier",
    "IntersectionHypothesis",
    "MotionExecutor",
    "MotionValidator",
    "NpzSceneSensor",
    "PlanningResult",
    "PointCloudPreprocessor",
    "PointCloudQualityReport",
    "PreprocessConfig",
    "PreprocessedScene",
    "ProportionEstimate",
    "RansacSegmentExtractor",
    "SceneFrame",
    "SceneSensor",
    "SegmentExtractionConfig",
    "SegmentRelation",
    "SortingCycleResult",
    "SortingState",
    "SortingTaskRunner",
    "VisibleSegment",
    "wilson_interval",
]
