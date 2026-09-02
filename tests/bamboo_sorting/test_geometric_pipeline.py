from __future__ import annotations

import numpy as np

from lerobot.bamboo_sorting import (
    CandidateStatus,
    GeometricGraspPlanner,
    GeometricPlanningPipeline,
    GraspPlannerConfig,
    IntersectionHypothesis,
    PointCloudPreprocessor,
    PreprocessConfig,
    RansacSegmentExtractor,
    SceneFrame,
    SegmentRelation,
    SegmentExtractionConfig,
    VisibleSegment,
)


def _strip_points(
    start: tuple[float, float, float], end: tuple[float, float, float], seed: int = 4
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 180)
    centerline = np.asarray(start) + t[:, None] * (np.asarray(end) - np.asarray(start))
    noise = np.column_stack(
        (
            rng.normal(0.0, 0.0005, len(t)),
            rng.uniform(-0.004, 0.004, len(t)),
            rng.normal(0.0, 0.0003, len(t)),
        )
    )
    return centerline + noise


def _pipeline(validator=None) -> GeometricPlanningPipeline:
    preprocessor = PointCloudPreprocessor(
        PreprocessConfig(
            workspace_min=(0.0, -0.2, 0.0),
            workspace_max=(0.6, 0.2, 0.4),
            table_height_m=0.1,
            voxel_size_m=0.0,
            min_retained_points=50,
        )
    )
    extractor = RansacSegmentExtractor(
        SegmentExtractionConfig(
            distance_threshold_m=0.006,
            min_inlier_points=50,
            min_visible_length_m=0.12,
            expected_length_m=0.30,
            ransac_iterations=100,
        )
    )
    planner = GeometricGraspPlanner(
        GraspPlannerConfig(
            min_graspable_length_m=0.12,
            min_width_m=0.002,
            max_width_m=0.020,
            min_completeness=0.3,
            min_clearance_m=0.0,
            intersection_exclusion_radius_m=0.02,
            pregrasp_distance_m=0.05,
        ),
        validator,
    )
    return GeometricPlanningPipeline(preprocessor, extractor, planner)


class ApproveAll:
    def rejection_reasons(self, candidate):
        return ()


class RejectAll:
    def rejection_reasons(self, candidate):
        return ("synthetic_collision",)


def test_pipeline_requires_execution_validator_before_candidate_can_run():
    points = _strip_points((0.15, 0.0, 0.12), (0.45, 0.0, 0.12))
    frame = SceneFrame(points, timestamp_s=1.0, frame_id="base")

    geometry_only = _pipeline().plan(frame)
    assert geometry_only.segments
    assert any(item.status == CandidateStatus.GEOMETRY_VALID for item in geometry_only.candidates)
    assert geometry_only.executable_candidates == ()

    validated = _pipeline(ApproveAll()).plan(frame)
    assert validated.executable_candidates
    candidate = validated.executable_candidates[0]
    assert candidate.pregrasp_frame_base[2, 3] > candidate.grasp_frame_base[2, 3]
    assert set(candidate.score_components) == {
        "visible_length",
        "completeness",
        "center_preference",
        "intersection_clearance",
        "hard_rejection_penalty",
    }
    assert candidate.hard_constraint_results["motion_validation"]


def test_validator_rejection_reason_is_persisted():
    points = _strip_points((0.15, 0.0, 0.12), (0.45, 0.0, 0.12))
    result = _pipeline(RejectAll()).plan(SceneFrame(points, 1.0, "base"))
    assert result.executable_candidates == ()
    assert any("synthetic_collision" in item.rejection_reasons for item in result.candidates)
    assert any(not item.hard_constraint_results["motion_validation"] for item in result.candidates)


def test_missing_extrinsic_is_reported_and_rejected():
    points = _strip_points((0.15, 0.0, 0.12), (0.45, 0.0, 0.12))
    result = _pipeline(ApproveAll()).plan(SceneFrame(points, 1.0, "mech_eye"))
    assert "missing_base_calibration" in result.scene.quality.warnings
    assert all(item.status == CandidateStatus.REJECTED for item in result.candidates)
    assert all("missing_base_calibration" in item.rejection_reasons for item in result.candidates)


def test_preprocessing_reports_nonfinite_and_table_filtering():
    points = np.array(
        [
            [0.2, 0.0, 0.12],
            [0.2, 0.0, 0.10],
            [np.nan, 0.0, 0.12],
            [0.8, 0.0, 0.12],
        ]
    )
    processor = PointCloudPreprocessor(
        PreprocessConfig(
            (0.0, -0.2, 0.0),
            (0.6, 0.2, 0.4),
            table_height_m=0.1,
            table_clearance_m=0.002,
            voxel_size_m=0.0,
            min_retained_points=1,
        )
    )
    scene = processor.process(SceneFrame(points, 1.0, "base"))
    assert scene.quality.raw_points == 4
    assert scene.quality.finite_points == 3
    assert scene.quality.workspace_points == 2
    assert scene.quality.retained_points == 1


def test_radius_outlier_filter_removes_isolated_point():
    cluster = np.array([[0.2, 0.0, 0.12], [0.201, 0.0, 0.12], [0.2, 0.001, 0.12]])
    isolated = np.array([[0.4, 0.1, 0.12]])
    processor = PointCloudPreprocessor(
        PreprocessConfig(
            (0.0, -0.2, 0.0),
            (0.6, 0.2, 0.4),
            table_height_m=0.1,
            voxel_size_m=0.0,
            outlier_radius_m=0.005,
            outlier_min_neighbors=2,
            min_retained_points=1,
        )
    )
    scene = processor.process(SceneFrame(np.vstack((cluster, isolated)), 1.0, "base"))
    assert len(scene.points_xyz) == 3


def test_only_candidates_inside_intersection_zone_are_rejected():
    pipeline = _pipeline()
    frame = SceneFrame(_strip_points((0.15, 0.0, 0.12), (0.45, 0.0, 0.12)), 1.0, "base")
    scene = pipeline.preprocessor.process(frame)
    segment = VisibleSegment(
        "segment-a",
        np.array([1.0, 0.0, 0.0]),
        np.array([0.15, 0.0, 0.12]),
        np.array([0.45, 0.0, 0.12]),
        width_m=0.008,
        mean_height_m=0.12,
        completeness=1.0,
        clearance_m=0.01,
        point_count=100,
        near_intersection=True,
    )
    intersection = IntersectionHypothesis(
        "segment-a",
        "segment-b",
        np.array([0.30, 0.0, 0.12]),
        SegmentRelation.CROSSING,
        confidence=1.0,
        separation_m=0.0,
    )
    candidates = pipeline.planner.plan(scene, (segment,), (intersection,))
    center = next(item for item in candidates if item.candidate_id.endswith("f0.50"))
    outer = [item for item in candidates if not item.candidate_id.endswith("f0.50")]
    assert "near_intersection" in center.rejection_reasons
    assert all(item.status == CandidateStatus.GEOMETRY_VALID for item in outer)
