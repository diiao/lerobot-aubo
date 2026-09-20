# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.bamboo_sorting.c0_capture_contract import (
    C0_CAPTURE_PROFILE_ID,
    C0_DATASET_FPS,
    C0_EPISODE_MANIFEST_SCHEMA_VERSION,
    C0EpisodeManifestV1,
)
from lerobot.bamboo_sorting.c0_gripper_dataset_auditor import (
    C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION,
    C0_GRIPPER_SLICE_DIGEST_SCHEMA_VERSION,
    C0GripperDatasetAuditReportV1,
    C0GripperDatasetSliceDigestV1,
    audit_c0_gripper_dataset_episode,
)
from lerobot.bamboo_sorting.contracts import (
    ACTION_FIELD_NAMES,
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
)
from lerobot.bamboo_sorting.lerobot_bridge import (
    CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
    build_camera_set_v2_lerobot_features,
)
from lerobot.bamboo_sorting.observation_contract import OBSERVATION_STATE_FIELD_NAMES
from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_V2_SCHEMA_VERSION,
    FROZEN_CAMERA_SET_V2_SHA256,
)

EPISODE_ID = "c0-gripper-dataset-episode-00"
EVIDENCE_SHA = "34" * 32
FRAME_COUNT = 100
ON_FRAME = 10
OFF_FRAME = 50
FPS = int(C0_DATASET_FPS)

ACTION_GRIPPER_DIM = ACTION_FIELD_NAMES.index("ee.gripper_pos")
STATE_GRIPPER_DIM = OBSERVATION_STATE_FIELD_NAMES.index("gripper_pos")

FIXED_BOUNDARY_FIELDS = {
    "complete_gripper_audit_ready": False,
    "controller_event_evidence_bound": False,
    "controller_output_verified": False,
    "dataset_episode_sha256_match": None,
    "manifest_dataset_episode_sha256_verified": False,
    "video_content_verified": False,
    "physical_gripper_feedback_available": False,
    "physical_grasp_success_proven": False,
    "training_authorized": False,
    "policy_execution_authorized": False,
    "serialized_record_grants_live_authorization": False,
    "hardware_access_performed_by_audit": False,
    "dataset_files_modified_by_audit": False,
}


def _episode_manifest(frame_count: int = FRAME_COUNT, human_outcome: str = "success"):
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    return C0EpisodeManifestV1(
        schema_version=C0_EPISODE_MANIFEST_SCHEMA_VERSION,
        episode_id=EPISODE_ID,
        scene_id="c0-scene-00",
        session_id="c0-session-01",
        split_name="train",
        camera_set_schema_version=CAMERA_SET_V2_SCHEMA_VERSION,
        camera_set_sha256=FROZEN_CAMERA_SET_V2_SHA256,
        bridge_version=CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        instruction_schema_version=INSTRUCTION_SCHEMA_VERSION,
        instruction_id=instruction.instruction_id,
        instruction_text=instruction.text,
        instruction_language=INSTRUCTION_LANGUAGE,
        instruction_text_sha256=instruction.text_sha256,
        calibration_version="calibration-c0-v1",
        tool_version="tool-c0-v1",
        capture_profile_id=C0_CAPTURE_PROFILE_ID,
        fps=C0_DATASET_FPS,
        frame_count=frame_count,
        dataset_episode_ref="datasets/c0/episode-00",
        dataset_episode_sha256=EVIDENCE_SHA,
        timestamp_evidence_ref="evidence/timestamps-00.json",
        timestamp_evidence_sha256=EVIDENCE_SHA,
        authorization_evidence_ref="evidence/authorization-00.json",
        authorization_evidence_sha256=EVIDENCE_SHA,
        vlm_shadow_evidence_ref="evidence/vlm-shadow-00.json",
        vlm_shadow_evidence_sha256=EVIDENCE_SHA,
        human_outcome=human_outcome,
        human_outcome_evidence_ref="evidence/human-outcome-00.json",
        human_outcome_evidence_sha256=EVIDENCE_SHA,
        raw_data_immutable=True,
        finalized=True,
    )


def _gripper_action(frame_count: int, on_frame: int | None, off_frame: int | None) -> list[float]:
    values = [0.0] * frame_count
    if on_frame is not None:
        end = off_frame if off_frame is not None else frame_count
        for index in range(on_frame, end):
            values[index] = 100.0
    return values


def _latched_observation(action: list[float]) -> list[float]:
    return [0.0, *action[:-1]]


def _episode_rows(
    episode_index: int = 0,
    frame_count: int = FRAME_COUNT,
    on_frame: int | None = ON_FRAME,
    off_frame: int | None = OFF_FRAME,
    fps: float = C0_DATASET_FPS,
    index_offset: int = 0,
) -> dict[str, list]:
    """Build one structurally valid episode's rows; mutate copies for defects."""

    gripper_action = _gripper_action(frame_count, on_frame, off_frame)
    gripper_observation = _latched_observation(gripper_action)
    action = np.full((frame_count, len(ACTION_FIELD_NAMES)), 0.01, dtype=np.float64)
    state = np.full((frame_count, len(OBSERVATION_STATE_FIELD_NAMES)), 0.02, dtype=np.float64)
    action[:, ACTION_GRIPPER_DIM] = gripper_action
    state[:, STATE_GRIPPER_DIM] = gripper_observation
    return {
        "action": action.tolist(),
        "observation.state": state.tolist(),
        "episode_index": [episode_index] * frame_count,
        "frame_index": list(range(frame_count)),
        "timestamp": [frame / fps for frame in range(frame_count)],
        "index": [index_offset + frame for frame in range(frame_count)],
    }


def _rows_to_table(rows_list: list[dict[str, list]], shuffle: bool = False) -> pa.Table:
    merged: dict[str, list] = {}
    for rows in rows_list:
        for key, values in rows.items():
            merged.setdefault(key, []).extend(values)
    length = len(merged["episode_index"])
    order = np.arange(length)
    if shuffle:
        np.random.default_rng(20260920).shuffle(order)
    return pa.table(
        {
            "action": pa.array(
                [merged["action"][i] for i in order], type=pa.list_(pa.float64())
            ),
            "observation.state": pa.array(
                [merged["observation.state"][i] for i in order], type=pa.list_(pa.float64())
            ),
            "episode_index": pa.array(
                [merged["episode_index"][i] for i in order], type=pa.int64()
            ),
            "frame_index": pa.array([merged["frame_index"][i] for i in order], type=pa.int64()),
            "timestamp": pa.array([merged["timestamp"][i] for i in order], type=pa.float64()),
            "index": pa.array([merged["index"][i] for i in order], type=pa.int64()),
        }
    )


def _write_table(root: Path, table: pa.Table) -> None:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_dir / "episode_000000.parquet")


def _write_parquet(root: Path, rows_list: list[dict[str, list]], shuffle: bool = False) -> None:
    _write_table(root, _rows_to_table(rows_list, shuffle=shuffle))


def _replace_column(table: pa.Table, name: str, values: list, arrow_type: pa.DataType) -> pa.Table:
    return table.set_column(
        table.schema.get_field_index(name), name, pa.array(values, type=arrow_type)
    )


def _write_info(
    root: Path,
    *,
    fps: float = C0_DATASET_FPS,
    features: dict | None = None,
    raw_content: str | None = None,
) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if raw_content is not None:
        (meta_dir / "info.json").write_text(raw_content, encoding="utf-8")
        return
    if features is None:
        features = build_camera_set_v2_lerobot_features()
    # Normalize tuples to lists exactly as a JSON round trip would store them.
    features = json.loads(json.dumps(features))
    info = {
        "codebase_version": "v2.1",
        "fps": fps,
        "features": features,
        "total_episodes": 1,
        "total_frames": FRAME_COUNT,
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _valid_features() -> dict:
    return json.loads(json.dumps(build_camera_set_v2_lerobot_features()))


def _write_valid_dataset(root: Path, **row_overrides) -> None:
    rows = _episode_rows(**row_overrides)
    _write_info(root)
    _write_parquet(root, [rows])


def _audit(root: Path, episode_index: int = 0, manifest: C0EpisodeManifestV1 | None = None):
    return audit_c0_gripper_dataset_episode(
        dataset_root=root,
        episode_index=episode_index,
        episode_manifest=manifest or _episode_manifest(),
    )


def _dataset_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_valid_single_strip_episode_is_structurally_valid_only(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    report = _audit(tmp_path)

    assert report.errors == ()
    assert report.blockers == ()
    assert report.dataset_slice_structurally_valid is True
    assert report.complete_gripper_audit_ready is False
    assert report.frame_count == FRAME_COUNT
    assert report.fps == C0_DATASET_FPS
    assert report.derived_command_on_frames == (ON_FRAME,)
    assert report.derived_command_off_frames == (OFF_FRAME,)
    assert report.gripper_slice_sha256 is not None
    assert len(report.gripper_slice_sha256) == 64

    record = report.to_manifest_record()
    json.dumps(record, sort_keys=True, allow_nan=False)
    for key, expected in FIXED_BOUNDARY_FIELDS.items():
        assert record[key] == expected, key
    assert "data_ready_for_training" not in record
    assert record["human_outcome"] == "success"
    assert record["episode_index"] == 0
    assert record["episode_id"] == EPISODE_ID


def test_missing_dataset_root_reports_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    report = _audit(missing)
    assert "dataset_root_missing" in report.errors
    assert report.dataset_slice_structurally_valid is False
    assert report.gripper_slice_sha256 is None
    assert report.frame_count is None


def test_missing_info_json_reports_error(tmp_path: Path) -> None:
    _write_parquet(tmp_path, [_episode_rows()])
    report = _audit(tmp_path)
    assert "info_json_missing" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_corrupt_info_json_reports_error(tmp_path: Path) -> None:
    _write_parquet(tmp_path, [_episode_rows()])
    _write_info(tmp_path, raw_content="{ not valid json")
    report = _audit(tmp_path)
    assert "info_json_invalid" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_info_json_without_object_content_reports_error(tmp_path: Path) -> None:
    _write_parquet(tmp_path, [_episode_rows()])
    _write_info(tmp_path, raw_content="[1, 2, 3]")
    report = _audit(tmp_path)
    assert "info_json_invalid" in report.errors


def test_missing_data_parquet_reports_error(tmp_path: Path) -> None:
    _write_info(tmp_path)
    report = _audit(tmp_path)
    assert "data_parquet_missing" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_episode_index_not_found_reports_error(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    report = _audit(tmp_path, episode_index=7)
    assert "episode_index_not_found" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_second_episode_is_selected_by_explicit_index(tmp_path: Path) -> None:
    first = _episode_rows(episode_index=0, on_frame=10, off_frame=50)
    second = _episode_rows(episode_index=1, on_frame=20, off_frame=70, index_offset=FRAME_COUNT)
    _write_info(tmp_path)
    _write_parquet(tmp_path, [first, second])
    report = _audit(tmp_path, episode_index=1)
    assert report.errors == ()
    assert report.derived_command_on_frames == (20,)
    assert report.derived_command_off_frames == (70,)


def test_frame_count_mismatch_with_manifest_reports_error(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    report = _audit(tmp_path, manifest=_episode_manifest(frame_count=FRAME_COUNT - 1))
    assert "frame_count_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_duplicate_frame_index_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    rows["frame_index"][5] = 4
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "duplicate_frame_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_missing_frame_index_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    keep = [i for i in range(FRAME_COUNT) if i != 5]
    for key in rows:
        rows[key] = [rows[key][i] for i in keep]
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path, manifest=_episode_manifest(frame_count=FRAME_COUNT - 1))
    assert "missing_frame_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_nonzero_frame_index_start_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    rows["frame_index"] = [frame + 1 for frame in rows["frame_index"]]
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "frame_index_nonzero_start" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_shuffled_parquet_rows_are_sorted_deterministically(tmp_path: Path) -> None:
    rows = _episode_rows()
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows], shuffle=True)
    shuffled_report = _audit(tmp_path)
    assert shuffled_report.errors == ()
    assert shuffled_report.dataset_slice_structurally_valid is True

    ordered_root = tmp_path / "ordered"
    _write_valid_dataset(ordered_root)
    ordered_report = _audit(ordered_root)
    assert shuffled_report.gripper_slice_sha256 == ordered_report.gripper_slice_sha256


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_timestamp_reports_error(tmp_path: Path, bad_value: float) -> None:
    rows = _episode_rows()
    rows["timestamp"][10] = bad_value
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "timestamp_non_finite" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_negative_timestamp_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    rows["timestamp"][0] = -0.01
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "timestamp_negative" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_non_monotonic_timestamp_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    rows["timestamp"][20] = rows["timestamp"][19]
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "timestamp_not_strictly_increasing" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_timestamp_frame_mismatch_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    rows["timestamp"] = [value + 0.1 for value in rows["timestamp"]]
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "timestamp_frame_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_fps_mismatch_reports_error(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, fps=30)
    report = _audit(tmp_path)
    assert "fps_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_action_shape_mismatch_reports_error(tmp_path: Path) -> None:
    features = _valid_features()
    features["action"]["shape"] = [7]
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert "action_shape_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_state_shape_mismatch_reports_error(tmp_path: Path) -> None:
    features = _valid_features()
    features["observation.state"]["shape"] = [12]
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert "state_shape_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_missing_action_names_report_error(tmp_path: Path) -> None:
    features = _valid_features()
    del features["action"]["names"]
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert "action_names_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_reordered_state_names_report_error(tmp_path: Path) -> None:
    features = _valid_features()
    names = features["observation.state"]["names"]
    names[0], names[1] = names[1], names[0]
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert "state_names_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_missing_global_rgb_feature_reports_error(tmp_path: Path) -> None:
    features = _valid_features()
    del features["observation.images.global_rgb"]
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert "missing_global_rgb_feature" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_missing_grasp_rgb_feature_reports_error(tmp_path: Path) -> None:
    features = _valid_features()
    del features["observation.images.grasp_rgb"]
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert "missing_grasp_rgb_feature" in report.errors
    assert report.dataset_slice_structurally_valid is False


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "observation.images.handeye",
        "observation.images.fixed",
        "observation.images.wrist_rgb",
        "observation.images.wrist_depth_m",
        "observation.images.grasp_depth",
    ],
)
def test_forbidden_features_report_error(tmp_path: Path, forbidden_key: str) -> None:
    features = _valid_features()
    features[forbidden_key] = {"dtype": "video", "shape": [480, 640, 3]}
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert f"forbidden_feature_present:{forbidden_key}" in report.errors
    assert report.dataset_slice_structurally_valid is False


@pytest.mark.parametrize("bad_value", [50.0, 0.5, 1.0, 25.0])
def test_non_binary_gripper_values_report_error(tmp_path: Path, bad_value: float) -> None:
    rows = _episode_rows()
    rows["action"][30][ACTION_GRIPPER_DIM] = bad_value
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "gripper_value_invalid" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_observation_start_not_released_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    rows["observation.state"][0][STATE_GRIPPER_DIM] = 100.0
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "observation_start_not_released" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_observation_action_latch_mismatch_reports_error(tmp_path: Path) -> None:
    rows = _episode_rows()
    # Break the one-frame latch: observation[5] no longer equals action[4].
    rows["observation.state"][5][STATE_GRIPPER_DIM] = 100.0
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "observation_action_latch_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_missing_command_on_reports_error(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path, on_frame=None, off_frame=None)
    report = _audit(tmp_path)
    assert "missing_command_on" in report.errors
    assert "missing_command_off" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_missing_command_off_reports_error(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path, off_frame=None)
    report = _audit(tmp_path)
    assert "missing_command_off" in report.errors
    assert "action_not_end_released" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_single_frame_pulse_reports_chatter(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path, on_frame=ON_FRAME, off_frame=ON_FRAME + 1)
    report = _audit(tmp_path)
    assert "gripper_chatter" in report.errors
    assert "multiple_command_on" not in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_multiple_on_off_cycles_report_errors(tmp_path: Path) -> None:
    gripper = [0.0] * FRAME_COUNT
    for index in list(range(10, 30)) + list(range(60, 80)):
        gripper[index] = 100.0
    rows = _episode_rows()
    observation = _latched_observation(gripper)
    for index in range(FRAME_COUNT):
        rows["action"][index][ACTION_GRIPPER_DIM] = gripper[index]
        rows["observation.state"][index][STATE_GRIPPER_DIM] = observation[index]
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "multiple_command_on" in report.errors
    assert "multiple_command_off" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_action_not_starting_released_reports_error(tmp_path: Path) -> None:
    gripper = _gripper_action(FRAME_COUNT, ON_FRAME, OFF_FRAME)
    gripper[0] = 100.0
    observation = _latched_observation(gripper)
    rows = _episode_rows()
    for index in range(FRAME_COUNT):
        rows["action"][index][ACTION_GRIPPER_DIM] = gripper[index]
        rows["observation.state"][index][STATE_GRIPPER_DIM] = observation[index]
    _write_info(tmp_path)
    _write_parquet(tmp_path, [rows])
    report = _audit(tmp_path)
    assert "action_not_start_released" in report.errors
    assert report.dataset_slice_structurally_valid is False


@pytest.mark.parametrize("human_outcome", ["failure", "uncertain"])
def test_non_success_human_outcome_is_preserved_and_blocked(
    tmp_path: Path, human_outcome: str
) -> None:
    _write_valid_dataset(tmp_path)
    report = _audit(tmp_path, manifest=_episode_manifest(human_outcome=human_outcome))
    assert report.dataset_slice_structurally_valid is True
    assert report.blockers == ("human_outcome_not_success",)
    assert report.complete_gripper_audit_ready is False

    record = report.to_manifest_record()
    assert record["human_outcome"] == human_outcome
    assert record["blockers"] == ["human_outcome_not_success"]
    assert record["training_authorized"] is False
    assert record["complete_gripper_audit_ready"] is False


def test_digest_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    first = _audit(tmp_path)
    second = _audit(tmp_path)
    assert first.gripper_slice_sha256 is not None
    assert first.gripper_slice_sha256 == second.gripper_slice_sha256


def test_digest_changes_with_action_gripper_value(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    baseline = _audit(tmp_path)

    changed = tmp_path / "changed"
    _write_valid_dataset(changed, on_frame=ON_FRAME + 1, off_frame=OFF_FRAME + 1)
    changed_report = _audit(changed)

    assert changed_report.errors == ()
    assert baseline.gripper_slice_sha256 != changed_report.gripper_slice_sha256


def test_digest_changes_with_timestamp(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    baseline = _audit(tmp_path)

    shifted = tmp_path / "shifted"
    rows = _episode_rows()
    rows["timestamp"] = [value + 0.001 for value in rows["timestamp"]]
    _write_info(shifted)
    _write_parquet(shifted, [rows])
    shifted_report = _audit(shifted)

    assert shifted_report.errors == ()
    assert baseline.gripper_slice_sha256 != shifted_report.gripper_slice_sha256


def test_digest_changes_with_frame_index_sequence(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    baseline = _audit(tmp_path)

    shorter = tmp_path / "shorter"
    _write_valid_dataset(shorter, frame_count=FRAME_COUNT - 1)
    shorter_report = _audit(shorter, manifest=_episode_manifest(frame_count=FRAME_COUNT - 1))

    assert shorter_report.errors == ()
    assert baseline.gripper_slice_sha256 != shorter_report.gripper_slice_sha256


def test_slice_digest_dataclass_matches_report_digest(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    report = _audit(tmp_path)
    gripper_action = _gripper_action(FRAME_COUNT, ON_FRAME, OFF_FRAME)
    digest = C0GripperDatasetSliceDigestV1(
        schema_version=C0_GRIPPER_SLICE_DIGEST_SCHEMA_VERSION,
        episode_index=0,
        fps=C0_DATASET_FPS,
        frame_count=FRAME_COUNT,
        action_feature_names=ACTION_FIELD_NAMES,
        observation_state_feature_names=OBSERVATION_STATE_FIELD_NAMES,
        frame_index=tuple(range(FRAME_COUNT)),
        timestamps=tuple(frame / FPS for frame in range(FRAME_COUNT)),
        action_gripper_values=tuple(gripper_action),
        observation_gripper_values=tuple(_latched_observation(gripper_action)),
    )
    assert digest.sha256 == report.gripper_slice_sha256
    record = digest.to_manifest_record()
    assert record["digest_scope"] == "gripper_slice_only"
    assert record["covers_full_dataset_episode"] is False
    assert record["covers_video_content"] is False
    assert record["verifies_manifest_dataset_episode_sha256"] is False
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_report_record_json_serialization_is_stable(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    report = _audit(tmp_path)
    first = json.dumps(report.to_manifest_record(), sort_keys=True, allow_nan=False)
    second = json.dumps(report.to_manifest_record(), sort_keys=True, allow_nan=False)
    assert first == second


def test_report_record_is_deepcopy_isolated(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    report = _audit(tmp_path)
    pristine = report.to_manifest_record()
    mutated = report.to_manifest_record()
    mutated["errors"].append("forged_error")
    mutated["derived_command_on_frames"].append(999)
    mutated["dataset_slice_structurally_valid"] = False
    mutated["training_authorized"] = True
    assert report.to_manifest_record() == pristine
    assert report.errors == ()
    assert report.dataset_slice_structurally_valid is True


def test_audit_does_not_modify_dataset_files(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    before = _dataset_snapshot(tmp_path)
    report = _audit(tmp_path)
    assert report.dataset_slice_structurally_valid is True
    after = _dataset_snapshot(tmp_path)
    assert before == after


def test_report_results_cannot_be_forged_by_caller(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    constructor_kwargs = {
        "schema_version": C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION,
        "dataset_root": str(tmp_path),
        "episode_index": 0,
        "episode_manifest": _episode_manifest(),
    }
    for forged_field in (
        "errors",
        "blockers",
        "dataset_slice_structurally_valid",
        "complete_gripper_audit_ready",
        "training_authorized",
        "policy_execution_authorized",
        "gripper_slice_sha256",
    ):
        with pytest.raises(TypeError):
            C0GripperDatasetAuditReportV1(**constructor_kwargs, **{forged_field: []})

    report = C0GripperDatasetAuditReportV1(**constructor_kwargs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.episode_index = 1
    with pytest.raises(AttributeError):
        report.dataset_slice_structurally_valid = False
    with pytest.raises(AttributeError):
        report.complete_gripper_audit_ready = True


def test_entry_point_validates_explicit_episode_index(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    manifest = _episode_manifest()
    with pytest.raises(TypeError):
        audit_c0_gripper_dataset_episode(tmp_path, True, manifest)
    with pytest.raises(TypeError):
        audit_c0_gripper_dataset_episode(tmp_path, "0", manifest)
    with pytest.raises(ValueError, match="non-negative"):
        audit_c0_gripper_dataset_episode(tmp_path, -1, manifest)
    with pytest.raises(TypeError):
        audit_c0_gripper_dataset_episode(tmp_path, 0, manifest.to_manifest_record())


@pytest.mark.parametrize("bad_fps", [float("nan"), float("inf"), float("-inf"), 0, -25])
def test_invalid_fps_reports_error_without_crashing(tmp_path: Path, bad_fps: float) -> None:
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, fps=bad_fps)
    report = _audit(tmp_path)
    assert "fps_mismatch" in report.errors
    assert report.fps is None
    assert report.gripper_slice_sha256 is None
    assert report.dataset_slice_structurally_valid is False
    json.dumps(report.to_manifest_record(), sort_keys=True, allow_nan=False)


def test_float_episode_index_is_rejected_not_truncated(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(
        table, "episode_index", [0.5] * FRAME_COUNT, pa.float64()
    )
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:episode_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_string_episode_index_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(table, "episode_index", ["0"] * FRAME_COUNT, pa.utf8())
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:episode_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_null_episode_index_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(
        table, "episode_index", [0] * (FRAME_COUNT - 1) + [None], pa.int64()
    )
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_value:episode_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_float_frame_index_is_rejected_not_truncated(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(
        table,
        "frame_index",
        [frame + 0.25 for frame in range(FRAME_COUNT)],
        pa.float64(),
    )
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:frame_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_string_frame_index_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(
        table, "frame_index", [str(frame) for frame in range(FRAME_COUNT)], pa.utf8()
    )
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:frame_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_null_frame_index_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(
        table,
        "frame_index",
        list(range(FRAME_COUNT - 1)) + [None],
        pa.int64(),
    )
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_value:frame_index" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_string_timestamp_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(table, "timestamp", ["0.0"] * FRAME_COUNT, pa.utf8())
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:timestamp" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_null_timestamp_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    rows = _episode_rows()
    values = rows["timestamp"][:-1] + [None]
    table = _replace_column(table, "timestamp", values, pa.float64())
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_value:timestamp" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_scalar_action_column_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(table, "action", [0.0] * FRAME_COUNT, pa.float64())
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:action" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_string_list_action_column_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(
        table,
        "action",
        [["0.0"] * len(ACTION_FIELD_NAMES)] * FRAME_COUNT,
        pa.list_(pa.utf8()),
    )
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:action" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_string_list_state_column_is_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    table = _replace_column(
        table,
        "observation.state",
        [["0.0"] * len(OBSERVATION_STATE_FIELD_NAMES)] * FRAME_COUNT,
        pa.list_(pa.utf8()),
    )
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert "invalid_column_type:observation.state" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_ragged_action_rows_are_rejected(tmp_path: Path) -> None:
    table = _rows_to_table([_episode_rows()])
    rows = _episode_rows()
    ragged = [row[:7] for row in rows["action"]]
    table = _replace_column(table, "action", ragged, pa.list_(pa.float64()))
    _write_info(tmp_path)
    _write_table(tmp_path, table)
    report = _audit(tmp_path)
    assert report.dataset_slice_structurally_valid is False
    assert report.errors != ()


def test_third_unknown_rgb_feature_is_rejected(tmp_path: Path) -> None:
    features = _valid_features()
    features["observation.images.third_rgb"] = features["observation.images.global_rgb"]
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert "unexpected_image_feature:observation.images.third_rgb" in report.errors
    assert report.dataset_slice_structurally_valid is False


@pytest.mark.parametrize("camera_key", ["global_rgb", "grasp_rgb"])
@pytest.mark.parametrize(
    "mutation",
    [
        {"dtype": "string"},
        {"shape": [1]},
        {"names": ["not", "the", "axes"]},
    ],
)
def test_image_feature_schema_mismatch_is_rejected(
    tmp_path: Path, camera_key: str, mutation: dict
) -> None:
    features = _valid_features()
    features[f"observation.images.{camera_key}"].update(mutation)
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    assert f"{camera_key}_feature_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


@pytest.mark.parametrize("feature_key", ["action", "observation.state"])
def test_feature_dtype_mismatch_is_rejected(tmp_path: Path, feature_key: str) -> None:
    features = _valid_features()
    features[feature_key]["dtype"] = "video"
    _write_valid_dataset(tmp_path)
    _write_info(tmp_path, features=features)
    report = _audit(tmp_path)
    short = "action" if feature_key == "action" else "state"
    assert f"{short}_dtype_mismatch" in report.errors
    assert report.dataset_slice_structurally_valid is False


def test_reports_with_identical_inputs_compare_equal(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    assert _audit(tmp_path) == _audit(tmp_path)


def test_reports_differ_when_dataset_content_changes(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    first = _audit(tmp_path)
    _write_valid_dataset(tmp_path, on_frame=None, off_frame=None)
    second = _audit(tmp_path)
    assert first.dataset_slice_structurally_valid is True
    assert second.dataset_slice_structurally_valid is False
    assert first.gripper_slice_sha256 != second.gripper_slice_sha256
    assert first != second


_IMPORT_PROBE = r"""
import importlib.util
import json
import os
import sys
import threading
import types

# Stub the parent packages so importing the audited module does NOT execute
# the pre-existing lerobot.bamboo_sorting __init__ (which pulls in heavy
# modules such as torch via smolvla_adapter).
def _stub_package(name):
    spec = importlib.util.find_spec(name)
    module = types.ModuleType(name)
    module.__path__ = list(spec.submodule_search_locations)
    sys.modules[name] = module

_stub_package("lerobot")
_stub_package("lerobot.bamboo_sorting")

baseline_modules = set(sys.modules)

import lerobot.bamboo_sorting.c0_gripper_dataset_auditor  # noqa: F401

forbidden_roots = {"cv2", "pyaubo_sdk", "torch"}
new_modules = sorted(set(sys.modules) - baseline_modules)
forbidden_modules = [name for name in new_modules if name.split(".")[0] in forbidden_roots]
extra_threads = [
    thread.name for thread in threading.enumerate() if thread is not threading.main_thread()
]
sockets = []
for fd in os.listdir("/proc/self/fd"):
    path = f"/proc/self/fd/{fd}"
    if os.path.islink(path) and "socket" in os.readlink(path):
        sockets.append(fd)
print(
    json.dumps(
        {
            "forbidden_modules": forbidden_modules,
            "extra_threads": extra_threads,
            "sockets": sockets,
            "package_init_executed": "lerobot.bamboo_sorting.smolvla_adapter" in sys.modules,
            "module_loaded": "lerobot.bamboo_sorting.c0_gripper_dataset_auditor" in sys.modules,
        }
    )
)
"""


def test_importing_dataset_auditor_touches_no_hardware_network_or_threads() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    assert state["module_loaded"] is True
    assert state["package_init_executed"] is False
    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []
