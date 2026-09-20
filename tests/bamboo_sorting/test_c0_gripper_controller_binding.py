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
from lerobot.bamboo_sorting.c0_gripper_controller_binding import (
    C0_GRIPPER_CONTROLLER_BINDING_REPORT_SCHEMA_VERSION,
    C0GripperControllerBindingReportV1,
    audit_c0_gripper_controller_binding,
)
from lerobot.bamboo_sorting.c0_gripper_controller_sidecar import (
    C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
    C0_GRIPPER_CONTROLLER_SIDECAR_SCHEMA_VERSION,
    C0GripperControllerAttemptV1,
    C0GripperControllerSidecarV1,
    build_c0_gripper_event_from_aubo_trace,
    verify_c0_gripper_controller_sidecar_integrity,
)
from lerobot.bamboo_sorting.c0_gripper_dataset_auditor import (
    C0GripperDatasetAuditReportV1,
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
    CAMERASET_V1_LEROBOT_BRIDGE_VERSION,
    build_cameras_set_v1_lerobot_features,
)
from lerobot.bamboo_sorting.observation_contract import OBSERVATION_STATE_FIELD_NAMES
from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_SCHEMA_VERSION,
    FROZEN_CAMERA_SET_V1_SHA256,
)

EPISODE_ID = "c0-controller-binding-episode-00"
EVIDENCE_SHA = "56" * 32
FRAME_COUNT = 100
ON_FRAME = 10
OFF_FRAME = 50
FPS = C0_DATASET_FPS

ACTION_GRIPPER_DIM = ACTION_FIELD_NAMES.index("ee.gripper_pos")
STATE_GRIPPER_DIM = OBSERVATION_STATE_FIELD_NAMES.index("gripper_pos")


def _episode_manifest(
    frame_count: int = FRAME_COUNT, human_outcome: str = "success"
) -> C0EpisodeManifestV1:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    return C0EpisodeManifestV1(
        schema_version=C0_EPISODE_MANIFEST_SCHEMA_VERSION,
        episode_id=EPISODE_ID,
        scene_id="c0-scene-00",
        session_id="c0-session-01",
        split_name="train",
        camera_set_schema_version=CAMERA_SET_SCHEMA_VERSION,
        camera_set_sha256=FROZEN_CAMERA_SET_V1_SHA256,
        bridge_version=CAMERASET_V1_LEROBOT_BRIDGE_VERSION,
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


def _write_dataset(
    root: Path,
    *,
    frame_count: int = FRAME_COUNT,
    on_frame: int | None = ON_FRAME,
    off_frame: int | None = OFF_FRAME,
    fps: float = FPS,
) -> None:
    gripper_action = [0.0] * frame_count
    if on_frame is not None:
        end = off_frame if off_frame is not None else frame_count
        for index in range(on_frame, end):
            gripper_action[index] = 100.0
    gripper_observation = [0.0, *gripper_action[:-1]]

    action = np.full((frame_count, len(ACTION_FIELD_NAMES)), 0.01, dtype=np.float64)
    state = np.full((frame_count, len(OBSERVATION_STATE_FIELD_NAMES)), 0.02, dtype=np.float64)
    action[:, ACTION_GRIPPER_DIM] = gripper_action
    state[:, STATE_GRIPPER_DIM] = gripper_observation

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": pa.array(action.tolist(), type=pa.list_(pa.float64())),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float64())),
            "episode_index": pa.array([0] * frame_count, type=pa.int64()),
            "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
            "timestamp": pa.array(
                [frame / fps for frame in range(frame_count)], type=pa.float64()
            ),
            "index": pa.array(list(range(frame_count)), type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    features = json.loads(json.dumps(build_cameras_set_v1_lerobot_features()))
    info = {
        "codebase_version": "v2.1",
        "fps": fps,
        "features": features,
        "total_episodes": 1,
        "total_frames": frame_count,
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _dataset_report(
    root: Path, manifest: C0EpisodeManifestV1 | None = None, **dataset_overrides
) -> C0GripperDatasetAuditReportV1:
    _write_dataset(root, **dataset_overrides)
    return audit_c0_gripper_dataset_episode(
        dataset_root=root,
        episode_index=0,
        episode_manifest=manifest or _episode_manifest(),
    )


def _success_trace(target_on: bool, *, readback_supported: bool = True) -> dict:
    if target_on:
        requested_do = {"2": True, "3": False}
        do_writes = [
            {"pin": 3, "value": False, "return_code": 0},
            {"pin": 2, "value": True, "return_code": 0},
        ]
        before, after = False, True
    else:
        requested_do = {"2": False, "3": True}
        do_writes = [
            {"pin": 2, "value": False, "return_code": 0},
            {"pin": 3, "value": True, "return_code": 0},
        ]
        before, after = True, False
    trace = {
        "requested_transition": True,
        "target_suction_on": target_on,
        "requested_do": requested_do,
        "do_writes": do_writes,
        "do_write_attempt_count": 2,
        "do_api_success": True,
        "do_readback_supported": readback_supported,
        "do_readback": dict(requested_do) if readback_supported else None,
        "do_readback_after_failure": None,
        "do_readback_error": None,
        "controller_output_state_known": readback_supported,
        "controller_output_matches_requested": True if readback_supported else None,
        "partial_write_possible": False,
        "commanded_state_before": before,
        "commanded_state_after": after,
        "error": None,
    }
    if readback_supported:
        trace["do_readback_attempts"] = 1
    return trace


def _write_exception_failure_trace() -> dict:
    """First SDK write raised: attempt_count == len(do_writes) + 1."""

    return {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [],
        "do_write_attempt_count": 1,
        "do_api_success": False,
        "do_readback_supported": True,
        "do_readback": None,
        "do_readback_after_failure": {"2": False, "3": False},
        "do_readback_error": None,
        "controller_output_state_known": True,
        "controller_output_matches_requested": False,
        "partial_write_possible": True,
        "commanded_state_before": False,
        "commanded_state_after": False,
        "error": "sdk connection lost",
    }


def _readback_error_failure_trace() -> dict:
    """Writes succeeded but the readback API raised: output state unknown."""

    return {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [
            {"pin": 3, "value": False, "return_code": 0},
            {"pin": 2, "value": True, "return_code": 0},
        ],
        "do_write_attempt_count": 2,
        "do_api_success": False,
        "do_readback_supported": True,
        "do_readback": None,
        "do_readback_after_failure": None,
        "do_readback_error": "getStandardDigitalOutput raised: io offline",
        "controller_output_state_known": False,
        "controller_output_matches_requested": None,
        "partial_write_possible": True,
        "commanded_state_before": False,
        "commanded_state_after": False,
        "error": "getStandardDigitalOutput raised: io offline",
    }


def _readback_mismatch_failure_trace() -> dict:
    """Writes succeeded but the readback never matched the request."""

    return {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [
            {"pin": 3, "value": False, "return_code": 0},
            {"pin": 2, "value": True, "return_code": 0},
        ],
        "do_write_attempt_count": 2,
        "do_api_success": False,
        "do_readback_supported": True,
        "do_readback": {"2": False, "3": False},
        "do_readback_after_failure": {"2": False, "3": False},
        "do_readback_error": None,
        "controller_output_state_known": True,
        "controller_output_matches_requested": False,
        "partial_write_possible": True,
        "commanded_state_before": False,
        "commanded_state_after": False,
        "error": "controller DO readback mismatch",
        "do_readback_attempts": 8,
    }


def _attempt(
    trace: dict,
    *,
    attempt_index: int,
    frame_index: int,
    committed: bool,
    episode_id: str = EPISODE_ID,
    dataset_timestamp_s: float | None = None,
) -> C0GripperControllerAttemptV1:
    if dataset_timestamp_s is None:
        dataset_timestamp_s = frame_index / FPS
    event = build_c0_gripper_event_from_aubo_trace(
        episode_id=episode_id,
        event_index=attempt_index,
        candidate_frame_index=frame_index,
        dataset_timestamp_s=dataset_timestamp_s,
        action_gripper_pos=100.0 if trace["target_suction_on"] else 0.0,
        source_trace=trace,
    )
    return C0GripperControllerAttemptV1(
        schema_version=C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
        attempt_index=attempt_index,
        candidate_frame_index=frame_index,
        dataset_frame_committed=committed,
        event=event,
        source_trace=trace,
    )


def _success_attempts(**trace_kwargs) -> tuple[C0GripperControllerAttemptV1, ...]:
    return (
        _attempt(
            _success_trace(True, **trace_kwargs),
            attempt_index=0,
            frame_index=ON_FRAME,
            committed=True,
        ),
        _attempt(
            _success_trace(False, **trace_kwargs),
            attempt_index=1,
            frame_index=OFF_FRAME,
            committed=True,
        ),
    )


def _sidecar(
    report: C0GripperDatasetAuditReportV1,
    attempts,
    *,
    finalized: bool = True,
    episode_id: str = EPISODE_ID,
    episode_index: int = 0,
    fps: float = FPS,
    frame_count: int = FRAME_COUNT,
    digest: str | None = None,
) -> C0GripperControllerSidecarV1:
    return C0GripperControllerSidecarV1(
        schema_version=C0_GRIPPER_CONTROLLER_SIDECAR_SCHEMA_VERSION,
        episode_id=episode_id,
        episode_index=episode_index,
        fps=fps,
        frame_count=frame_count,
        dataset_gripper_slice_sha256=(
            digest if digest is not None else report.gripper_slice_sha256
        ),
        attempts=tuple(attempts),
        finalized=finalized,
    )


def _audit(
    report: C0GripperDatasetAuditReportV1, sidecar: C0GripperControllerSidecarV1
) -> C0GripperControllerBindingReportV1:
    return audit_c0_gripper_controller_binding(report, sidecar)


# --- C. dataset <-> controller sidecar binding ------------------------------


def test_valid_binding_structurally_valid_and_output_evidence_complete(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    assert report.dataset_slice_structurally_valid is True
    sidecar = _sidecar(report, _success_attempts())

    binding = _audit(report, sidecar)

    assert binding.errors == ()
    assert binding.blockers == ()
    assert binding.committed_on_frames == (ON_FRAME,)
    assert binding.committed_off_frames == (OFF_FRAME,)
    assert binding.failed_or_uncommitted_attempt_count == 0
    assert binding.controller_event_binding_structurally_valid is True
    assert binding.controller_output_evidence_complete is True
    # Fixed in this schema version: sidecar is not wired into record_loop.
    assert binding.complete_gripper_audit_ready is False


def test_timestamp_within_half_cycle_tolerance_passes(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        _attempt(
            _success_trace(True),
            attempt_index=0,
            frame_index=ON_FRAME,
            committed=True,
            dataset_timestamp_s=ON_FRAME / FPS + 0.01,  # < 0.5 / FPS
        ),
        _attempt(
            _success_trace(False), attempt_index=1, frame_index=OFF_FRAME, committed=True
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert binding.errors == ()
    assert binding.controller_event_binding_structurally_valid is True


def test_episode_id_mismatch_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = _success_attempts()
    rebuilt = tuple(
        _attempt(
            attempt.source_trace,
            attempt_index=attempt.attempt_index,
            frame_index=attempt.candidate_frame_index,
            committed=True,
            episode_id="c0-other-episode",
        )
        for attempt in attempts
    )
    binding = _audit(report, _sidecar(report, rebuilt, episode_id="c0-other-episode"))
    assert "episode_id_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_episode_index_mismatch_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts(), episode_index=1))
    assert "episode_index_mismatch" in binding.errors


def test_frame_count_mismatch_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts(), frame_count=101))
    assert "frame_count_mismatch" in binding.errors


def test_fps_mismatch_blocks_and_nan_cannot_bypass(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts(), fps=30.0))
    assert "fps_mismatch" in binding.errors
    # A NaN sidecar fps cannot even be constructed to bypass the comparison.
    with pytest.raises(ValueError, match="fps"):
        _sidecar(report, _success_attempts(), fps=float("nan"))


def test_slice_digest_mismatch_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts(), digest="00" * 32))
    assert "dataset_gripper_slice_sha256_mismatch" in binding.errors


def test_missing_command_on_event_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        _attempt(
            _success_trace(False), attempt_index=0, frame_index=OFF_FRAME, committed=True
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "missing_committed_controller_event" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_missing_command_off_event_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        _attempt(
            _success_trace(True), attempt_index=0, frame_index=ON_FRAME, committed=True
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "missing_committed_controller_event" in binding.errors


def test_unexpected_extra_committed_event_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        *_success_attempts(),
        _attempt(_success_trace(True), attempt_index=2, frame_index=70, committed=True),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "unexpected_committed_controller_event" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_swapped_direction_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        _attempt(
            _success_trace(False), attempt_index=0, frame_index=ON_FRAME, committed=True
        ),
        _attempt(
            _success_trace(True), attempt_index=1, frame_index=OFF_FRAME, committed=True
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "controller_event_direction_mismatch" in binding.errors


def test_frame_off_by_one_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        _attempt(
            _success_trace(True), attempt_index=0, frame_index=ON_FRAME + 1, committed=True
        ),
        _attempt(
            _success_trace(False), attempt_index=1, frame_index=OFF_FRAME, committed=True
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "controller_event_frame_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_timestamp_beyond_half_cycle_tolerance_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        _attempt(
            _success_trace(True),
            attempt_index=0,
            frame_index=ON_FRAME,
            committed=True,
            dataset_timestamp_s=ON_FRAME / FPS + 0.5 / FPS + 0.001,
        ),
        _attempt(
            _success_trace(False), attempt_index=1, frame_index=OFF_FRAME, committed=True
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "controller_event_timestamp_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_failed_event_blocks_binding(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        *_success_attempts(),
        _attempt(
            _write_exception_failure_trace(),
            attempt_index=2,
            frame_index=FRAME_COUNT,
            committed=False,
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "controller_event_failure" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_uncommitted_failure_attempt_preserved_and_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        *_success_attempts(),
        _attempt(
            _write_exception_failure_trace(),
            attempt_index=2,
            frame_index=FRAME_COUNT,
            committed=False,
        ),
    )
    sidecar = _sidecar(report, attempts)
    binding = _audit(report, sidecar)

    assert "uncommitted_controller_attempt" in binding.errors
    assert binding.failed_or_uncommitted_attempt_count == 1
    assert binding.controller_event_binding_structurally_valid is False
    # The failure evidence survives untouched and is never dressed up as a
    # committed dataset frame.
    stored = sidecar.attempts[2]
    assert stored.dataset_frame_committed is False
    assert stored.candidate_frame_index == FRAME_COUNT
    assert stored.event.error == "sdk connection lost"
    assert stored.event.partial_write_possible is True
    assert stored.event.commanded_state_after == stored.event.commanded_state_before


def test_readback_not_supported_blocks_output_evidence(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts(readback_supported=False))
    binding = _audit(report, sidecar)

    assert binding.blockers == ("controller_readback_not_supported",)
    assert binding.controller_output_evidence_complete is False
    assert binding.controller_event_binding_structurally_valid is False
    assert binding.complete_gripper_audit_ready is False


def test_controller_output_state_unknown_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        *_success_attempts(),
        _attempt(
            _readback_error_failure_trace(),
            attempt_index=2,
            frame_index=FRAME_COUNT,
            committed=False,
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "controller_output_state_unknown" in binding.errors


def test_controller_output_mismatch_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        *_success_attempts(),
        _attempt(
            _readback_mismatch_failure_trace(),
            attempt_index=2,
            frame_index=FRAME_COUNT,
            committed=False,
        ),
    )
    binding = _audit(report, _sidecar(report, attempts))
    assert "controller_output_mismatch" in binding.errors


def test_structurally_invalid_dataset_report_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path, off_frame=None)
    assert report.dataset_slice_structurally_valid is False
    attempts = _success_attempts()
    binding = _audit(report, _sidecar(report, attempts))
    assert "dataset_report_structurally_invalid" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_dataset_human_outcome_blocker_grants_no_training_readiness(tmp_path: Path) -> None:
    manifest = _episode_manifest(human_outcome="failure")
    report = _dataset_report(tmp_path, manifest=manifest)
    assert "human_outcome_not_success" in report.blockers

    binding = _audit(report, _sidecar(report, _success_attempts()))
    record = binding.to_manifest_record()

    assert "human_outcome_not_success" in record["dataset_report_blockers"]
    assert record["complete_gripper_audit_ready"] is False
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert "data_ready_for_training" not in record


def test_unfinalized_sidecar_blocks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts(), finalized=False))
    assert "sidecar_not_finalized" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_report_constructor_rejects_injected_derived_fields(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts())
    for forged in (
        {"errors": ()},
        {"blockers": ()},
        {"committed_on_frames": (ON_FRAME,)},
        {"controller_event_binding_structurally_valid": True},
        {"controller_output_evidence_complete": True},
        {"complete_gripper_audit_ready": True},
    ):
        with pytest.raises(TypeError):
            C0GripperControllerBindingReportV1(
                schema_version=C0_GRIPPER_CONTROLLER_BINDING_REPORT_SCHEMA_VERSION,
                dataset_report=report,
                controller_sidecar=sidecar,
                **forged,
            )


def test_audit_entrypoint_type_checks(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts())
    with pytest.raises(TypeError, match="dataset_report"):
        audit_c0_gripper_controller_binding("not-a-report", sidecar)
    with pytest.raises(TypeError, match="controller_sidecar"):
        audit_c0_gripper_controller_binding(report, "not-a-sidecar")


def test_equal_inputs_equal_reports_changed_evidence_unequal(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    first = _audit(report, _sidecar(report, _success_attempts()))
    second = _audit(report, _sidecar(report, _success_attempts()))
    assert first == second

    perturbed_attempts = (
        _attempt(
            _success_trace(True),
            attempt_index=0,
            frame_index=ON_FRAME,
            committed=True,
            dataset_timestamp_s=ON_FRAME / FPS + 0.01,
        ),
        _attempt(
            _success_trace(False), attempt_index=1, frame_index=OFF_FRAME, committed=True
        ),
    )
    third = _audit(report, _sidecar(report, perturbed_attempts))
    assert first != third

    # A change in the dataset content changes the digest and the comparison.
    other = _dataset_report(tmp_path, on_frame=ON_FRAME + 2)
    fourth = _audit(other, _sidecar(report, _success_attempts()))
    assert first != fourth


def test_errors_and_blockers_sorted_deduped_deterministic(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    attempts = (
        _attempt(
            _success_trace(False), attempt_index=0, frame_index=ON_FRAME, committed=True
        ),
        _attempt(
            _write_exception_failure_trace(),
            attempt_index=1,
            frame_index=FRAME_COUNT,
            committed=False,
        ),
    )
    sidecar = _sidecar(report, attempts, finalized=False, digest="00" * 32)
    binding = _audit(report, sidecar)

    assert binding.errors == tuple(sorted(set(binding.errors)))
    assert binding.blockers == tuple(sorted(set(binding.blockers)))
    repeat = _audit(report, sidecar)
    assert binding.errors == repeat.errors
    assert binding.blockers == repeat.blockers
    assert {
        "controller_event_direction_mismatch",
        "controller_event_failure",
        "dataset_gripper_slice_sha256_mismatch",
        "missing_committed_controller_event",
        "sidecar_not_finalized",
        "uncommitted_controller_attempt",
    } <= set(binding.errors)


def test_manifest_record_fixed_boundary_fields_verbatim(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts()))
    record = binding.to_manifest_record()

    json.dumps(record, sort_keys=True, allow_nan=False)
    fixed = {
        "controller_event_origin_authenticated": False,
        "live_capture_integration_verified": False,
        "complete_gripper_audit_ready": False,
        "physical_gripper_feedback_available": False,
        "physical_grasp_success_proven": False,
        "training_authorized": False,
        "policy_execution_authorized": False,
        "serialized_record_grants_live_authorization": False,
        "hardware_access_performed_by_audit": False,
        "dataset_files_modified_by_audit": False,
    }
    for key, value in fixed.items():
        assert record[key] is value, key
    assert record["controller_event_binding_structurally_valid"] is True
    assert record["controller_output_evidence_complete"] is True
    assert "data_ready_for_training" not in record

    # Mutating the returned record must not reach back into the report.
    record["errors"].append("mutated")
    assert binding.errors == ()


# --- Rework: ordering and post-construction integrity (A-H) -----------------


def test_cross_direction_reversed_order_rejected_at_construction_and_binding(
    tmp_path: Path,
) -> None:
    report = _dataset_report(tmp_path)
    # Codex counterexample: attempt 0 = command_off@50, attempt 1 =
    # command_on@10 with sequential attempt_index.
    off_first = _attempt(
        _success_trace(False), attempt_index=0, frame_index=OFF_FRAME, committed=True
    )
    on_second = _attempt(
        _success_trace(True), attempt_index=1, frame_index=ON_FRAME, committed=True
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _sidecar(report, (off_first, on_second))

    # Controlled simulation of an object corrupted around the constructor:
    # the binding layer must reject it too, never valid=True.
    sidecar = _sidecar(report, _success_attempts())
    object.__setattr__(sidecar, "attempts", (off_first, on_second))
    binding = _audit(report, sidecar)
    assert "controller_event_order_mismatch" in binding.errors
    assert "sidecar_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False
    assert binding.controller_output_evidence_complete is False


def test_binding_rejects_trace_target_suction_on_tampering(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts())
    sidecar.attempts[0].source_trace["target_suction_on"] = False

    binding = _audit(report, sidecar)
    assert "controller_attempt_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False
    assert binding.controller_output_evidence_complete is False


def test_binding_rejects_trace_error_tampering(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts())
    sidecar.attempts[0].source_trace["error"] = "tampered after validation"

    binding = _audit(report, sidecar)
    assert "controller_attempt_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_binding_rejects_event_requested_do_tampering(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts())
    sidecar.attempts[0].event.requested_do["2"] = False

    binding = _audit(report, sidecar)
    assert "controller_attempt_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_binding_rejects_event_readback_and_writes_tampering(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)

    sidecar = _sidecar(report, _success_attempts())
    sidecar.attempts[1].event.do_readback["3"] = False
    binding = _audit(report, sidecar)
    assert "controller_attempt_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False

    sidecar = _sidecar(report, _success_attempts())
    sidecar.attempts[0].event.do_writes[1]["value"] = False
    binding = _audit(report, sidecar)
    assert "controller_attempt_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False


def test_binding_rejects_post_construction_attempt_tampering_and_serialization(
    tmp_path: Path,
) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts())

    # Untouched baseline passes.
    assert _audit(report, sidecar).controller_event_binding_structurally_valid is True

    # Tamper after full construction: binding rejects and serialization fails
    # closed instead of emitting a stale-digest record.
    sidecar.attempts[0].source_trace["target_suction_on"] = False
    sidecar.attempts[0].source_trace["error"] = "tampered after validation"
    binding = _audit(report, sidecar)
    assert "controller_attempt_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False
    with pytest.raises(ValueError, match="integrity_mismatch"):
        sidecar.to_manifest_record()


def test_binding_untampered_audit_remains_deterministic(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    first = _audit(report, _sidecar(report, _success_attempts()))
    second = _audit(report, _sidecar(report, _success_attempts()))
    assert first == second
    assert first.controller_event_binding_structurally_valid is True
    assert first.controller_output_evidence_complete is True
    assert first.complete_gripper_audit_ready is False
    assert json.dumps(first.to_manifest_record(), sort_keys=True, allow_nan=False) == (
        json.dumps(second.to_manifest_record(), sort_keys=True, allow_nan=False)
    )


# --- Rework: binding report lifecycle integrity (audit snapshot) -------------


def test_report_is_audit_snapshot_detached_from_caller_sidecar(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    sidecar = _sidecar(report, _success_attempts())
    binding = _audit(report, sidecar)
    assert binding.controller_event_binding_structurally_valid is True

    # Codex counterexample: tamper the caller's ORIGINAL sidecar after the
    # binding report was constructed. The historical audit snapshot must not
    # change.
    original_digest = sidecar.sidecar_sha256
    sidecar.attempts[0].source_trace["target_suction_on"] = False
    sidecar.attempts[0].source_trace["error"] = "tampered after binding report"
    assert verify_c0_gripper_controller_sidecar_integrity(sidecar) != ()

    assert binding.controller_sidecar is not sidecar
    assert binding.errors == ()
    assert binding.controller_event_binding_structurally_valid is True
    assert binding.controller_output_evidence_complete is True
    assert binding.complete_gripper_audit_ready is False

    # Every serialized field comes from the detached snapshot: the pinned
    # digest is the construction-time one and no tampered content leaks in.
    record = binding.to_manifest_record()
    assert record["sidecar_sha256"] == original_digest
    assert record["errors"] == []
    assert record["controller_event_binding_structurally_valid"] is True
    assert (
        binding.controller_sidecar.attempts[0].source_trace["target_suction_on"] is True
    )
    assert binding.controller_sidecar.attempts[0].source_trace["error"] is None


def test_report_own_snapshot_source_trace_tampering_fails_closed(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts()))

    binding.controller_sidecar.attempts[0].source_trace["target_suction_on"] = False
    binding.controller_sidecar.attempts[0].source_trace["error"] = "tampered"

    assert "binding_input_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False
    assert binding.controller_output_evidence_complete is False
    assert binding.complete_gripper_audit_ready is False
    with pytest.raises(ValueError, match="binding_input_integrity_mismatch"):
        binding.to_manifest_record()


def test_report_own_snapshot_event_requested_do_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts()))

    binding.controller_sidecar.attempts[0].event.requested_do["2"] = False

    assert "binding_input_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False
    with pytest.raises(ValueError, match="binding_input_integrity_mismatch"):
        binding.to_manifest_record()


def test_report_own_snapshot_attempt_order_swap_fails_closed(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    binding = _audit(report, _sidecar(report, _success_attempts()))

    swapped = (
        binding.controller_sidecar.attempts[1],
        binding.controller_sidecar.attempts[0],
    )
    object.__setattr__(binding.controller_sidecar, "attempts", swapped)

    assert "binding_input_integrity_mismatch" in binding.errors
    assert binding.controller_event_binding_structurally_valid is False
    assert binding.controller_output_evidence_complete is False
    with pytest.raises(ValueError, match="binding_input_integrity_mismatch"):
        binding.to_manifest_record()


def test_untampered_report_remains_deterministic(tmp_path: Path) -> None:
    report = _dataset_report(tmp_path)
    first = _audit(report, _sidecar(report, _success_attempts()))
    second = _audit(report, _sidecar(report, _success_attempts()))

    assert first == second
    assert first.errors == ()
    assert first.blockers == ()
    assert first.controller_event_binding_structurally_valid is True
    assert first.controller_output_evidence_complete is True
    assert first.complete_gripper_audit_ready is False
    encoded_first = json.dumps(
        first.to_manifest_record(), sort_keys=True, allow_nan=False
    )
    encoded_second = json.dumps(
        second.to_manifest_record(), sort_keys=True, allow_nan=False
    )
    assert encoded_first == encoded_second
    # Repeated access is stable and side-effect free.
    assert first.errors == ()
    assert first.controller_event_binding_structurally_valid is True


# --- D. import side-effect probe --------------------------------------------

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

import lerobot.bamboo_sorting.c0_gripper_controller_binding  # noqa: F401

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
            "module_loaded": "lerobot.bamboo_sorting.c0_gripper_controller_binding" in sys.modules,
        }
    )
)
"""


def test_importing_binding_module_touches_no_hardware_network_threads_or_files(
    tmp_path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
        cwd=tmp_path,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    assert state["module_loaded"] is True
    assert state["package_init_executed"] is False
    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []
    assert list(tmp_path.iterdir()) == []
