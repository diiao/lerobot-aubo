"""Composition of capture-independent preprocessing, extraction and planning."""

from __future__ import annotations

from dataclasses import dataclass

from .planner import GeometricGraspPlanner
from .preprocessing import PointCloudPreprocessor
from .segments import RansacSegmentExtractor
from .types import GraspCandidate, IntersectionHypothesis, PreprocessedScene, SceneFrame, VisibleSegment


@dataclass(frozen=True)
class PlanningResult:
    scene: PreprocessedScene
    segments: tuple[VisibleSegment, ...]
    intersections: tuple[IntersectionHypothesis, ...]
    candidates: tuple[GraspCandidate, ...]

    @property
    def executable_candidates(self) -> tuple[GraspCandidate, ...]:
        from .types import CandidateStatus

        return tuple(item for item in self.candidates if item.status == CandidateStatus.EXECUTION_VALIDATED)


class GeometricPlanningPipeline:
    def __init__(
        self,
        preprocessor: PointCloudPreprocessor,
        extractor: RansacSegmentExtractor,
        planner: GeometricGraspPlanner,
    ) -> None:
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.planner = planner

    def plan(self, frame: SceneFrame) -> PlanningResult:
        scene = self.preprocessor.process(frame)
        segments, intersections = self.extractor.extract(scene)
        candidates = self.planner.plan(scene, segments, intersections)
        return PlanningResult(scene, segments, intersections, candidates)
