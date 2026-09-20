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

"""Offline tests for the C0-R4B two-phase post-save evidence finalizer.

All datasets are synthetic snapshots under ``tmp_path``; the lifecycle test
exercises a real, still-open ``pyarrow.ParquetWriter`` to prove the audit
phase only becomes valid after the writer is safely closed. Robots and
datasets are in-memory fakes. No hardware, camera, DO/IO, motion, real
dataset, network, training, or inference is involved.
"""

import copy
import dataclasses
import json
import os
import types
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
from lerobot.bamboo_sorting.c0_gripper_capture_journal import (
    C0GripperPendingCaptureJournalV1,
)
from lerobot.bamboo_sorting.c0_gripper_save_finalizer import (
    C0_GRIPPER_SAVE_FINALIZATION_RESULT_SCHEMA_VERSION,
    C0GripperEpisodeSavePendingAuditV1,
    C0GripperSaveFinalizationError,
    C0GripperSaveFinalizationResultV1,
    begin_c0_gripper_episode_finalization,
    complete_c0_gripper_episode_finalization,
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

EPISODE_ID = "c0-r4b-finalizer-episode-00"
EVIDENCE_SHA = "34" * 32
FPS = float(C0_DATASET_FPS)

# Small default episode: transitions on at frame 2, off at frame 4.
FRAME_COUNT = 6
ON_FRAME = 2
OFF_FRAME = 4

ACTION_GRIPPER_DIM = ACTION_FIELD_NAMES.index("ee.gripper_pos")
STATE_GRIPPER_DIM = OBSERVATION_STATE_FIELD_NAMES.index("gripper_pos")

AUTHORIZATION_FIELDS = {
    "training_authorized",
    "policy_execution_authorized",
    "serialized_record_grants_live_authorization",
    "hardware_access_performed_by_serialization",
    "hardware_access_performed_by_audit",
    "physical_gripper_feedback_available",
    "physical_grasp_success_proven",
}


def _episode_manifest(frame_count: int = FRAME_COUNT) -> C0EpisodeManifestV1:
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
        human_outcome="success",
        human_outcome_evidence_ref="evidence/human-outcome-00.json",
        human_outcome_evidence_sha256=EVIDENCE_SHA,
        raw_data_immutable=True,
        finalized=True,
    )


def _gripper_action(
    frame_count: int, on_frame: int | None, off_frame: int | None
) -> list[float]:
    values = [0.0] * frame_count
    if on_frame is not None:
        end = off_frame if off_frame is not None else frame_count
        for index in range(on_frame, end):
            values[index] = 100.0
    return values


def _episode_table(
    *,
    frame_count: int,
    on_frame: int | None,
    off_frame: int | None,
    fps: float = FPS,
) -> pa.Table:
    action_values = _gripper_action(frame_count, on_frame, off_frame)
    observation = [0.0, *action_values[:-1]]
    action = np.full((frame_count, len(ACTION_FIELD_NAMES)), 0.01, dtype=np.float64)
    state = np.full((frame_count, len(OBSERVATION_STATE_FIELD_NAMES)), 0.02, dtype=np.float64)
    action[:, ACTION_GRIPPER_DIM] = action_values
    state[:, STATE_GRIPPER_DIM] = observation
    return pa.table(
        {
            "action": pa.array(action.tolist(), type=pa.list_(pa.float64())),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float64())),
            "episode_index": pa.array([0] * frame_count, type=pa.int64()),
            "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
            "timestamp": pa.array([frame / fps for frame in range(frame_count)], type=pa.float64()),
            "index": pa.array(list(range(frame_count)), type=pa.int64()),
        }
    )


def _write_info(root: Path, *, frame_count: int, fps: float = FPS) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "fps": fps,
        "features": json.loads(json.dumps(build_cameras_set_v1_lerobot_features())),
        "total_episodes": 1,
        "total_frames": frame_count,
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _write_dataset(
    root: Path,
    *,
    frame_count: int = FRAME_COUNT,
    on_frame: int | None = ON_FRAME,
    off_frame: int | None = OFF_FRAME,
    fps: float = FPS,
) -> None:
    """Write one closed synthetic saved-episode snapshot."""
    table = _episode_table(frame_count=frame_count, on_frame=on_frame, off_frame=off_frame, fps=fps)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_dir / "episode_000000.parquet")
    _write_info(root, frame_count=frame_count, fps=fps)


def _success_trace(target_on: bool, *, readback_supported: bool = True) -> dict:
    if target_on:
        trace = {
            "requested_transition": True,
            "target_suction_on": True,
            "requested_do": {"2": True, "3": False},
            "do_writes": [
                {"pin": 3, "value": False, "return_code": 0},
                {"pin": 2, "value": True, "return_code": 0},
            ],
            "do_write_attempt_count": 2,
            "do_api_success": True,
            "do_readback_supported": readback_supported,
            "do_readback": {"2": True, "3": False} if readback_supported else None,
            "do_readback_after_failure": None,
            "do_readback_error": None,
            "controller_output_state_known": readback_supported,
            "controller_output_matches_requested": True if readback_supported else None,
            "partial_write_possible": False,
            "commanded_state_before": False,
            "commanded_state_after": True,
            "error": None,
        }
    else:
        trace = {
            "requested_transition": True,
            "target_suction_on": False,
            "requested_do": {"2": False, "3": True},
            "do_writes": [
                {"pin": 2, "value": False, "return_code": 0},
                {"pin": 3, "value": True, "return_code": 0},
            ],
            "do_write_attempt_count": 2,
            "do_api_success": True,
            "do_readback_supported": readback_supported,
            "do_readback": {"2": False, "3": True} if readback_supported else None,
            "do_readback_after_failure": None,
            "do_readback_error": None,
            "controller_output_state_known": readback_supported,
            "controller_output_matches_requested": True if readback_supported else None,
            "partial_write_possible": False,
            "commanded_state_before": True,
            "commanded_state_after": False,
            "error": None,
        }
    if readback_supported:
        trace["do_readback_attempts"] = 1
    return trace


def _hold_trace(on: bool) -> dict:
    return {
        "requested_transition": False,
        "target_suction_on": on,
        "requested_do": None,
        "do_writes": [],
        "do_write_attempt_count": 0,
        "do_api_success": None,
        "do_readback_supported": True,
        "do_readback": None,
        "do_readback_after_failure": None,
        "do_readback_error": None,
        "controller_output_state_known": False,
        "controller_output_matches_requested": None,
        "partial_write_possible": False,
        "commanded_state_before": on,
        "commanded_state_after": on,
        "error": None,
    }


class FakeRobot:
    def __init__(self, trace):
        self.last_gripper_command_trace = trace


class FakeDataset:
    """Duck-typed dataset mimicking the real save lifecycle.

    ``episode_buffer`` is open before the save (size and episode_index of the
    pending episode) and cleared once ``save_episode`` returns, like
    ``LeRobotDataset``.
    """

    def __init__(
        self,
        root: Path,
        *,
        frame_count: int,
        episode_index: int = 0,
        fail: Exception | None = None,
        with_meta: bool = False,
    ):
        self.root = root
        self.fail = fail
        self.save_calls = 0
        self.episode_buffer: dict | None = {
            "size": frame_count,
            "episode_index": episode_index,
        }
        self.meta = (
            types.SimpleNamespace(total_episodes=episode_index) if with_meta else None
        )

    def save_episode(self) -> None:
        self.save_calls += 1
        if self.fail is not None:
            raise self.fail
        # The real dataset clears its in-memory buffer once the episode is
        # handed to the (still-open) ParquetWriter.
        self.episode_buffer = None


def _journal() -> C0GripperPendingCaptureJournalV1:
    return C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )


def _drive_cycle(journal, *, frame_index: int, trace: dict, gripper: float) -> None:
    action = {"ee.gripper_pos": gripper}
    robot = FakeRobot(trace)
    journal.before_send(
        robot=robot,
        dataset=None,
        action=action,
        robot_action_to_send=action,
        candidate_frame_index=frame_index,
    )
    journal.on_send_success(
        robot=robot,
        dataset=None,
        action=action,
        robot_action_to_send=action,
        sent_action=action,
        candidate_frame_index=frame_index,
    )
    journal.on_add_frame_success(
        robot=robot,
        dataset=None,
        action=action,
        robot_action_to_send=action,
        candidate_frame_index=frame_index,
    )


def _drive_full_journal(
    journal,
    *,
    frame_count: int = FRAME_COUNT,
    on_frame: int | None = ON_FRAME,
    off_frame: int | None = OFF_FRAME,
) -> None:
    """Drive one cycle per frame with traces matching the dataset trajectory."""

    action_values = _gripper_action(frame_count, on_frame, off_frame)
    for frame in range(frame_count):
        gripper = action_values[frame]
        if on_frame is not None and frame == on_frame:
            trace = _success_trace(target_on=True)
        elif off_frame is not None and frame == off_frame:
            trace = _success_trace(target_on=False)
        else:
            trace = _hold_trace(on=gripper == 100.0)
        _drive_cycle(journal, frame_index=frame, trace=trace, gripper=gripper)


def _begin(journal, dataset, *, frame_count: int = FRAME_COUNT) -> C0GripperEpisodeSavePendingAuditV1:
    return begin_c0_gripper_episode_finalization(
        journal=journal,
        dataset=dataset,
        episode_index=0,
        episode_manifest=_episode_manifest(frame_count=frame_count),
    )


def _begin_complete(
    journal, dataset, *, frame_count: int = FRAME_COUNT
) -> C0GripperSaveFinalizationResultV1:
    pending = _begin(journal, dataset, frame_count=frame_count)
    return complete_c0_gripper_episode_finalization(
        journal=journal,
        dataset=dataset,
        pending=pending,
        episode_manifest=_episode_manifest(frame_count=frame_count),
    )


def _walk_false_authorization_fields(node) -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in AUTHORIZATION_FIELDS and value is not False:
                bad.append(key)
            bad.extend(_walk_false_authorization_fields(value))
    elif isinstance(node, list):
        for item in node:
            bad.extend(_walk_false_authorization_fields(item))
    return bad


def _manual_token(
    root: Path,
    *,
    frame_count: int = FRAME_COUNT,
    fps: float = FPS,
    episode_index: int = 0,
) -> C0GripperEpisodeSavePendingAuditV1:
    return C0GripperEpisodeSavePendingAuditV1(
        schema_version="C0GripperEpisodeSavePendingAuditV1",
        episode_id=EPISODE_ID,
        episode_index=episode_index,
        fps=fps,
        frame_count=frame_count,
        dataset_root=str(Path(root).resolve(strict=True)),
        candidate_frame_indices=tuple(range(frame_count)),
    )


# ------------------------------------------------------------- lifecycle layer


def test_audit_before_writer_close_fails_then_succeeds_after_close(tmp_path) -> None:
    """Real ParquetWriter lifecycle: save returns while the writer is open."""
    _write_info(tmp_path, frame_count=FRAME_COUNT)
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    table = _episode_table(frame_count=FRAME_COUNT, on_frame=ON_FRAME, off_frame=OFF_FRAME)
    writer = pq.ParquetWriter(data_dir / "episode_000000.parquet", table.schema)
    writer.write_table(table)
    try:
        journal = _journal()
        _drive_full_journal(journal)
        dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
        pending = _begin(journal, dataset)

        # The episode is saved but NOT re-readable: the writer is still open.
        with pytest.raises(C0GripperSaveFinalizationError, match="structurally invalid"):
            complete_c0_gripper_episode_finalization(
                journal=journal,
                dataset=dataset,
                pending=pending,
                episode_manifest=_episode_manifest(),
            )
        assert dataset.save_calls == 1
    finally:
        writer.close()

    # After the public close, the audit-only retry succeeds without re-saving.
    result = complete_c0_gripper_episode_finalization(
        journal=journal,
        dataset=dataset,
        pending=pending,
        episode_manifest=_episode_manifest(),
    )
    assert dataset.save_calls == 1
    assert result.structurally_finalized is True


def test_complete_before_begin_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    token = _manual_token(tmp_path)

    with pytest.raises(C0GripperSaveFinalizationError, match="saved_pending_audit"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=token,
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 0


def test_save_failure_enters_save_uncertain_and_never_resaves(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT, fail=RuntimeError("flush crashed"))

    with pytest.raises(RuntimeError, match="flush crashed"):
        _begin(journal, dataset)
    assert dataset.save_calls == 1

    # Save-uncertain is terminal for this coordinator: no automatic re-save,
    # and the audit phase is unreachable without manual recovery.
    with pytest.raises(C0GripperSaveFinalizationError, match="save_uncertain"):
        _begin(journal, dataset)
    assert dataset.save_calls == 1
    with pytest.raises(C0GripperSaveFinalizationError, match="save_uncertain"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=_manual_token(tmp_path),
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 1


def test_keyboard_interrupt_enters_save_uncertain_and_never_resaves(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(
        tmp_path, frame_count=FRAME_COUNT, fail=KeyboardInterrupt("operator interrupt")
    )

    with pytest.raises(KeyboardInterrupt):
        _begin(journal, dataset)
    assert dataset.save_calls == 1
    assert journal._c0_r4b_state == "save_uncertain"

    with pytest.raises(C0GripperSaveFinalizationError, match="save_uncertain"):
        _begin(journal, dataset)
    assert dataset.save_calls == 1


def test_unknown_save_state_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    journal._c0_r4b_state = "bogus-garbage"
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    with pytest.raises(C0GripperSaveFinalizationError, match="unknown save state"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_post_save_audit_failure_retries_are_audit_only(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    # Sabotage the saved snapshot: audit fails after the save.
    parquet = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.unlink()
    with pytest.raises(C0GripperSaveFinalizationError, match="structurally invalid"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 1

    # Audit-only retry after restoring the snapshot: no second save.
    _write_dataset(tmp_path)
    result = complete_c0_gripper_episode_finalization(
        journal=journal,
        dataset=dataset,
        pending=pending,
        episode_manifest=_episode_manifest(),
    )
    assert dataset.save_calls == 1
    assert result.structurally_finalized is True


# -------------------------------------------------------- pre-save validation


def test_identity_mismatch_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    with pytest.raises(C0GripperSaveFinalizationError, match="episode_index"):
        begin_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            episode_index=1,
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 0


def test_manifest_episode_id_mismatch_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    other_manifest = dataclasses.replace(
        _episode_manifest(), episode_id="c0-r4b-other-episode"
    )

    with pytest.raises(C0GripperSaveFinalizationError, match="episode_id"):
        begin_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            episode_index=0,
            episode_manifest=other_manifest,
        )
    assert dataset.save_calls == 0


def test_partial_journal_rejected_before_save(tmp_path) -> None:
    """Two journal cycles can never match a saved episode: coverage is exact."""
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_cycle(journal, frame_index=ON_FRAME, trace=_success_trace(True), gripper=100.0)
    _drive_cycle(journal, frame_index=OFF_FRAME, trace=_success_trace(False), gripper=0.0)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_duplicate_or_out_of_range_candidates_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    duplicate = _journal()
    _drive_full_journal(duplicate)
    # Re-drive frame 1 with a duplicate candidate (append extra hold cycle).
    _drive_cycle(duplicate, frame_index=1, trace=_hold_trace(on=False), gripper=0.0)
    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        _begin(duplicate, dataset)

    out_of_range = _journal()
    _drive_full_journal(out_of_range)
    _drive_cycle(
        out_of_range,
        frame_index=FRAME_COUNT,
        trace=_success_trace(True),
        gripper=100.0,
    )
    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        _begin(out_of_range, dataset)
    assert dataset.save_calls == 0


def test_buffer_size_mismatch_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT + 1)

    with pytest.raises(C0GripperSaveFinalizationError, match="episode_buffer size"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_closed_buffer_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    dataset.episode_buffer = None

    with pytest.raises(C0GripperSaveFinalizationError, match="not open"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_buffer_episode_index_mismatch_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT, episode_index=999)

    with pytest.raises(C0GripperSaveFinalizationError, match="episode_index 999"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


@pytest.mark.parametrize("bad_index", [True, 1.5, "0", None])
def test_buffer_episode_index_non_integer_rejected_before_save(tmp_path, bad_index) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    if bad_index is None:
        del dataset.episode_buffer["episode_index"]
    else:
        dataset.episode_buffer["episode_index"] = bad_index

    with pytest.raises(C0GripperSaveFinalizationError, match="episode_index"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_meta_total_episodes_mismatch_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(
        tmp_path, frame_count=FRAME_COUNT, episode_index=0, with_meta=True
    )
    dataset.meta.total_episodes = 7

    with pytest.raises(C0GripperSaveFinalizationError, match="total_episodes"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_meta_total_episodes_non_integer_rejected_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(
        tmp_path, frame_count=FRAME_COUNT, episode_index=0, with_meta=True
    )
    dataset.meta.total_episodes = True

    with pytest.raises(C0GripperSaveFinalizationError, match="total_episodes"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


# --------------------------------------------------------- root binding layer


def test_audited_root_must_be_the_saved_datasets_root(tmp_path) -> None:
    """Save A / audit B is structurally impossible: the root is derived."""
    saved_root = tmp_path / "dataset-A"
    audited_root = tmp_path / "dataset-B"
    saved_root.mkdir()
    audited_root.mkdir()
    _write_dataset(saved_root)  # the episode actually saved
    journal = _journal()
    _drive_full_journal(journal)
    # The fake dataset claims a root whose files do not exist.
    dataset = FakeDataset(audited_root, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    with pytest.raises(C0GripperSaveFinalizationError, match="structurally invalid"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 1


def test_root_changed_between_save_and_audit_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    _write_dataset(other_root)
    dataset.root = other_root  # the dataset object now points elsewhere

    with pytest.raises(C0GripperSaveFinalizationError, match="no longer matches"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )


def test_symlink_retarget_between_begin_and_complete_fails_closed(tmp_path) -> None:
    real_a = tmp_path / "real-a"
    real_b = tmp_path / "real-b"
    link = tmp_path / "current"
    _write_dataset(real_a)
    _write_dataset(real_b, frame_count=8, on_frame=2, off_frame=6)
    os.symlink(real_a, link)

    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(link, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    # Retarget the symlink at a different dataset between save and audit.
    os.remove(link)
    os.symlink(real_b, link)

    with pytest.raises(C0GripperSaveFinalizationError, match="no longer matches"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )


# --------------------------------------------------------- snapshot integrity


def test_coherent_journal_change_after_save_rejected_by_snapshot_digest(
    tmp_path,
) -> None:
    """Mutating hold-cycle evidence to another VALID combo still changes the
    save-time snapshot and must be rejected."""
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    cycle = journal._cycles[1]
    cycle["dataset_action_gripper_pos"] = 100.0
    cycle["sent_action_gripper_command"] = 100.0
    cycle["trace"] = _hold_trace(on=True)

    with pytest.raises(C0GripperSaveFinalizationError, match="snapshot digest mismatch"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 1


def test_pending_token_fps_tampering_rejected(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    forged = dataclasses.replace(pending, fps=999.0)
    with pytest.raises(C0GripperSaveFinalizationError, match="not the one produced"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=forged,
            episode_manifest=_episode_manifest(),
        )


def test_pending_token_from_another_journal_rejected(tmp_path) -> None:
    _write_dataset(tmp_path)
    other_root = tmp_path / "other"
    _write_dataset(other_root)
    journal_a = _journal()
    _drive_full_journal(journal_a)
    journal_b = _journal()
    _drive_full_journal(journal_b)

    pending_a = _begin(journal_a, FakeDataset(tmp_path, frame_count=FRAME_COUNT))
    pending_b = _begin(journal_b, FakeDataset(other_root, frame_count=FRAME_COUNT))

    with pytest.raises(C0GripperSaveFinalizationError, match="not the one produced"):
        complete_c0_gripper_episode_finalization(
            journal=journal_a,
            dataset=FakeDataset(tmp_path, frame_count=FRAME_COUNT),
            pending=pending_b,
            episode_manifest=_episode_manifest(),
        )
    # journal_a's own token still works after the foreign token was refused.
    result = complete_c0_gripper_episode_finalization(
        journal=journal_a,
        dataset=FakeDataset(tmp_path, frame_count=FRAME_COUNT),
        pending=pending_a,
        episode_manifest=_episode_manifest(),
    )
    assert result.structurally_finalized is True


@pytest.mark.parametrize(
    "candidates",
    [
        tuple(range(FRAME_COUNT - 1)),  # too short
        tuple(range(FRAME_COUNT)) + (0,),  # duplicate
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),  # floats
        ("0", "1", "2", "3", "4", "5"),  # strings
        (True, False, 2, 3, 4, 5),  # bools
        (1, 0, 2, 3, 4, 5),  # wrong order
    ],
)
def test_pending_token_rejects_bool_float_and_string_candidates(candidates) -> None:
    with pytest.raises(ValueError):
        C0GripperEpisodeSavePendingAuditV1(
            schema_version="C0GripperEpisodeSavePendingAuditV1",
            episode_id=EPISODE_ID,
            episode_index=0,
            fps=FPS,
            frame_count=FRAME_COUNT,
            dataset_root="/tmp/whatever",
            candidate_frame_indices=candidates,
        )


# ------------------------------------------------------------- journal health


def test_incomplete_journal_cannot_begin(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    journal.mark_incomplete(reason="operator stopped recording")
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    with pytest.raises(C0GripperSaveFinalizationError, match="recording"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_abandoned_rerecord_journal_cannot_begin(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    journal.mark_abandoned(reason="operator requested rerecord")
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    with pytest.raises(C0GripperSaveFinalizationError, match="recording"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_corrupted_pending_attempt_fails_before_save(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    journal._attempts[0].source_trace["do_api_success"] = False
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    with pytest.raises(C0GripperSaveFinalizationError, match="tampered|corrupted"):
        _begin(journal, dataset)
    assert dataset.save_calls == 0


def test_journal_mutated_between_save_and_audit_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    # After the save, the journal evidence must be immutable.
    journal._cycles[0]["trace"]["do_api_success"] = False
    with pytest.raises(C0GripperSaveFinalizationError, match="tampered"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 1


# ------------------------------------------------------- transition correspondence


def test_missing_controller_event_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    # Journal holds the dataset trajectory but the on-transition trace is a hold.
    action_values = _gripper_action(FRAME_COUNT, ON_FRAME, OFF_FRAME)
    for frame in range(FRAME_COUNT):
        gripper = action_values[frame]
        trace = (
            _success_trace(target_on=False)
            if frame == OFF_FRAME
            else _hold_trace(on=gripper == 100.0)
        )
        _drive_cycle(journal, frame_index=frame, trace=trace, gripper=gripper)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    with pytest.raises(C0GripperSaveFinalizationError, match="one-to-one"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )


def test_extra_controller_event_fails_closed(tmp_path) -> None:
    # Dataset has no transitions; the journal fabricates one.
    _write_dataset(tmp_path, on_frame=None, off_frame=None)
    journal = _journal()
    _drive_full_journal(journal, on_frame=None, off_frame=None)
    _drive_cycle(journal, frame_index=3, trace=_success_trace(True), gripper=100.0)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        _begin(journal, dataset)


def test_controller_event_direction_mismatch_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    action_values = _gripper_action(FRAME_COUNT, ON_FRAME, OFF_FRAME)
    for frame in range(FRAME_COUNT):
        gripper = action_values[frame]
        if frame == ON_FRAME:
            # Internally consistent command_off where the dataset derives
            # command_on: direction mismatch must surface as a 1:1 failure.
            trace = _success_trace(target_on=False)
            gripper = 0.0
        elif frame == OFF_FRAME:
            trace = _success_trace(target_on=False)
        else:
            trace = _hold_trace(on=gripper == 100.0)
        _drive_cycle(journal, frame_index=frame, trace=trace, gripper=gripper)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    with pytest.raises(C0GripperSaveFinalizationError, match="one-to-one"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )


# ------------------------------------------------------------- R2 report layer


def test_r2_structurally_invalid_report_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    _write_info(tmp_path, frame_count=FRAME_COUNT, fps=30.0)  # info fps mismatch
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    with pytest.raises(C0GripperSaveFinalizationError, match="structurally invalid"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )
    assert dataset.save_calls == 1


def test_r2_missing_digest_fails_closed(tmp_path) -> None:
    # info.json exists but the parquet was never written.
    _write_info(tmp_path, frame_count=FRAME_COUNT)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    pending = _begin(journal, dataset)

    with pytest.raises(C0GripperSaveFinalizationError, match="structurally invalid"):
        complete_c0_gripper_episode_finalization(
            journal=journal,
            dataset=dataset,
            pending=pending,
            episode_manifest=_episode_manifest(),
        )


# ---------------------------------------------------------------- success path


def test_successful_two_phase_finalization_promotes_evidence(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)

    result = _begin_complete(journal, dataset)

    assert isinstance(result, C0GripperSaveFinalizationResultV1)
    assert result.schema_version == C0_GRIPPER_SAVE_FINALIZATION_RESULT_SCHEMA_VERSION
    assert result.structurally_finalized is True
    assert result.binding_report.errors == ()
    assert result.binding_report.controller_event_binding_structurally_valid is True

    assert len(result.sidecar.attempts) == 2
    assert all(
        attempt.dataset_frame_committed is True for attempt in result.sidecar.attempts
    )
    assert [attempt.attempt_index for attempt in result.sidecar.attempts] == [0, 1]
    assert result.sidecar.finalized is True
    assert result.sidecar.frame_count == FRAME_COUNT
    assert result.sidecar.dataset_gripper_slice_sha256 == (
        result.dataset_report.gripper_slice_sha256
    )
    assert result.dataset_gripper_slice_sha256 == result.dataset_report.gripper_slice_sha256
    assert result.sidecar.committed_attempts[0].event.event_type == "command_on"
    assert result.sidecar.committed_attempts[0].candidate_frame_index == ON_FRAME
    assert result.sidecar.committed_attempts[1].event.event_type == "command_off"
    assert result.sidecar.committed_attempts[1].candidate_frame_index == OFF_FRAME


def test_pending_journal_and_attempts_are_never_modified(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    pending_before = copy.deepcopy(journal.to_manifest_record())

    result = _begin_complete(journal, FakeDataset(tmp_path, frame_count=FRAME_COUNT))

    assert journal.to_manifest_record() == pending_before
    assert all(
        attempt.dataset_frame_committed is False for attempt in journal.controller_attempts
    )
    new_attempt = result.sidecar.attempts[0]
    old_attempt = journal.controller_attempts[0]
    assert new_attempt is not old_attempt
    assert new_attempt.dataset_frame_committed is True
    assert new_attempt.candidate_frame_index == old_attempt.candidate_frame_index
    assert new_attempt.event == old_attempt.event
    assert new_attempt.source_trace == old_attempt.source_trace


def test_result_record_is_deterministic_pure_json_and_detached(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    result = _begin_complete(journal, FakeDataset(tmp_path, frame_count=FRAME_COUNT))

    first = result.to_manifest_record()
    second = result.to_manifest_record()
    assert first == second
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == first
    assert _walk_false_authorization_fields(first) == []

    first["sidecar"]["attempts"][0]["source_trace"]["do_api_success"] = False
    first["training_authorized"] = True
    third = result.to_manifest_record()
    assert third["sidecar"]["attempts"][0]["source_trace"]["do_api_success"] is True
    assert third["training_authorized"] is False
    assert _walk_false_authorization_fields(third) == []


def test_identical_setups_produce_identical_records(tmp_path) -> None:
    _write_dataset(tmp_path)

    journal_a = _journal()
    _drive_full_journal(journal_a)
    journal_b = _journal()
    _drive_full_journal(journal_b)
    result_a = _begin_complete(journal_a, FakeDataset(tmp_path, frame_count=FRAME_COUNT))
    result_b = _begin_complete(journal_b, FakeDataset(tmp_path, frame_count=FRAME_COUNT))
    assert result_a.to_manifest_record() == result_b.to_manifest_record()


def test_repeated_finalization_fails_closed(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    _begin_complete(journal, dataset)

    with pytest.raises(C0GripperSaveFinalizationError, match="already"):
        _begin(journal, dataset)
    assert dataset.save_calls == 1


def test_rerecord_fork_does_not_leak_old_attempts(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    journal.mark_abandoned(reason="operator requested rerecord")
    nxt = journal.fork_for_rerecord(episode_id=EPISODE_ID, episode_index=0)

    assert nxt.controller_attempts == ()
    assert nxt.pending_events == ()
    dataset = FakeDataset(tmp_path, frame_count=FRAME_COUNT)
    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        _begin(nxt, dataset)
    assert dataset.save_calls == 0
    assert all(
        attempt.dataset_frame_committed is False for attempt in journal.controller_attempts
    )


# ------------------------------------------------------ result integrity layer


def _successful_result(tmp_path) -> C0GripperSaveFinalizationResultV1:
    _write_dataset(tmp_path)
    journal = _journal()
    _drive_full_journal(journal)
    return _begin_complete(journal, FakeDataset(tmp_path, frame_count=FRAME_COUNT))


def test_result_rejects_contradictory_top_level_identity(tmp_path) -> None:
    result = _successful_result(tmp_path)
    with pytest.raises(ValueError, match="episode_id contradicts"):
        C0GripperSaveFinalizationResultV1(
            schema_version=result.schema_version,
            episode_id="forged-episode",
            episode_index=result.episode_index,
            fps=result.fps,
            frame_count=result.frame_count,
            dataset_gripper_slice_sha256=result.dataset_gripper_slice_sha256,
            sidecar=result.sidecar,
            dataset_report=result.dataset_report,
            binding_report=result.binding_report,
        )


def test_result_rejects_contradictory_frame_count(tmp_path) -> None:
    result = _successful_result(tmp_path)
    with pytest.raises(ValueError, match="frame_count contradicts"):
        C0GripperSaveFinalizationResultV1(
            schema_version=result.schema_version,
            episode_id=result.episode_id,
            episode_index=result.episode_index,
            fps=result.fps,
            frame_count=1,
            dataset_gripper_slice_sha256=result.dataset_gripper_slice_sha256,
            sidecar=result.sidecar,
            dataset_report=result.dataset_report,
            binding_report=result.binding_report,
        )


def test_result_rejects_binding_report_from_another_sidecar(tmp_path) -> None:
    # A different episode (different frame count) produces a different
    # sidecar digest, so the binding report cannot be replayed here.
    other_root = tmp_path / "other"
    _write_dataset(other_root, frame_count=8, on_frame=2, off_frame=6)
    other_journal = _journal()
    _drive_full_journal(other_journal, frame_count=8, on_frame=2, off_frame=6)
    other_result = _begin_complete(
        other_journal, FakeDataset(other_root, frame_count=8), frame_count=8
    )

    result = _successful_result(tmp_path)
    with pytest.raises(ValueError, match="not produced over this sidecar"):
        C0GripperSaveFinalizationResultV1(
            schema_version=result.schema_version,
            episode_id=result.episode_id,
            episode_index=result.episode_index,
            fps=result.fps,
            frame_count=result.frame_count,
            dataset_gripper_slice_sha256=result.dataset_gripper_slice_sha256,
            sidecar=result.sidecar,
            dataset_report=result.dataset_report,
            binding_report=other_result.binding_report,
        )


def test_readback_not_supported_blocker_aligns_structurally_finalized(tmp_path) -> None:
    _write_dataset(tmp_path)
    journal = _journal()
    action_values = _gripper_action(FRAME_COUNT, ON_FRAME, OFF_FRAME)
    for frame in range(FRAME_COUNT):
        gripper = action_values[frame]
        if frame == ON_FRAME:
            trace = _success_trace(target_on=True, readback_supported=False)
        elif frame == OFF_FRAME:
            trace = _success_trace(target_on=False, readback_supported=False)
        else:
            trace = _hold_trace(on=gripper == 100.0)
        _drive_cycle(journal, frame_index=frame, trace=trace, gripper=gripper)

    result = _begin_complete(journal, FakeDataset(tmp_path, frame_count=FRAME_COUNT))

    # No structural error, so the promotion completes; the blocker is neither
    # deleted nor forged, and structurally_finalized aligns with R3 semantics
    # (no contradictory "finalized" claim while a blocker stands).
    assert result.binding_report.errors == ()
    assert "controller_readback_not_supported" in result.binding_report.blockers
    assert result.structurally_finalized is False
    assert (
        result.structurally_finalized
        == result.binding_report.controller_event_binding_structurally_valid
    )
    record = result.to_manifest_record()
    assert record["structurally_finalized"] is False
    assert "controller_readback_not_supported" in record["binding_blockers"]
