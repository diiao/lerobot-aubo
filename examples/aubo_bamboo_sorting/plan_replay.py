#!/usr/bin/env python
"""Plan one saved RGB-D/point-cloud NPZ frame; never connect to hardware."""

from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.bamboo_sorting import (
    GeometricGraspPlanner,
    GeometricPlanningPipeline,
    GraspPlannerConfig,
    NpzSceneSensor,
    PointCloudPreprocessor,
    PreprocessConfig,
    RansacSegmentExtractor,
    SegmentExtractionConfig,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _summary(result) -> dict[str, Any]:
    return {
        "safety_notice": (
            "offline task frames only; no candidate is executable without an external "
            "reachability/collision validator"
        ),
        "frame_id": result.scene.source.frame_id,
        "coordinate_frame": result.scene.coordinate_frame,
        "quality": {key: _json_value(value) for key, value in vars(result.scene.quality).items()},
        "segments": [
            {
                "segment_id": item.segment_id,
                "axis_unit": item.axis_unit.tolist(),
                "endpoint_start": item.endpoint_start.tolist(),
                "endpoint_end": item.endpoint_end.tolist(),
                "width_m": item.width_m,
                "visible_length_m": item.length_m,
                "completeness": item.completeness,
                "near_intersection": item.near_intersection,
                "near_workspace_edge": item.near_workspace_edge,
            }
            for item in result.segments
        ],
        "intersections": [
            {
                "segments": [item.first_segment_id, item.second_segment_id],
                "position_base": item.position_base.tolist(),
                "relation": item.relation.value,
                "confidence": item.confidence,
                "separation_m": item.separation_m,
            }
            for item in result.intersections
        ],
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "segment_id": item.segment_id,
                "status": item.status.value,
                "score": _json_value(item.score),
                "score_components": {key: _json_value(value) for key, value in item.score_components.items()},
                "hard_constraint_results": item.hard_constraint_results,
                "rejection_reasons": list(item.rejection_reasons),
                "grasp_task_frame_base": item.grasp_frame_base.tolist(),
                "pregrasp_task_frame_base": item.pregrasp_frame_base.tolist(),
            }
            for item in result.candidates
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="saved SceneFrame NPZ")
    parser.add_argument("config", type=Path, help="measured/synthetic JSON configuration")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    pipeline = GeometricPlanningPipeline(
        PointCloudPreprocessor(PreprocessConfig(**config["preprocess"])),
        RansacSegmentExtractor(SegmentExtractionConfig(**config["segments"])),
        # Deliberately omit MotionValidator: this CLI can never approve execution.
        GeometricGraspPlanner(GraspPlannerConfig(**config["planner"])),
    )
    result = pipeline.plan(NpzSceneSensor([args.scene]).capture())
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
