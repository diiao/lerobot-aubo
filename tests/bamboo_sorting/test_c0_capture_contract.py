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
from dataclasses import replace

import pytest

from lerobot.bamboo_sorting.c0_capture_contract import (
    C0_BATCH_MANIFEST_SCHEMA_VERSION,
    C0_CAPTURE_PROFILE_ID,
    C0_DATASET_FPS,
    C0_EPISODE_MANIFEST_SCHEMA_VERSION,
    C0_EXPECTED_EPISODE_COUNT,
    C0_SENSOR_TIMESTAMP_SCHEMA_VERSION,
    C0_SPLIT_RULE_VERSION,
    GRIPPER_STATE_SEMANTICS,
    C0BatchManifestV1,
    C0EpisodeManifestV1,
    C0SensorTimestampV1,
)
from lerobot.bamboo_sorting.contracts import (
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
)
from lerobot.bamboo_sorting.lerobot_bridge import CAMERASET_V2_LEROBOT_BRIDGE_VERSION
from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_V2_SCHEMA_VERSION,
    FROZEN_CAMERA_SET_V2_SHA256,
)

DATASET_SHA = "12" * 32
TIMESTAMP_SHA = "23" * 32
AUTHORIZATION_SHA = "34" * 32
VLM_SHA = "45" * 32
HUMAN_SHA = "56" * 32


def _episode(index: int = 0, **changes: object) -> C0EpisodeManifestV1:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    values = {
        "schema_version": C0_EPISODE_MANIFEST_SCHEMA_VERSION,
        "episode_id": f"c0-episode-{index:02d}",
        "scene_id": f"c0-scene-{index:02d}",
        "session_id": "c0-session-01",
        "split_name": "train" if index < 8 else "validation",
        "camera_set_schema_version": CAMERA_SET_V2_SCHEMA_VERSION,
        "camera_set_sha256": FROZEN_CAMERA_SET_V2_SHA256,
        "bridge_version": CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": instruction.instruction_id,
        "instruction_text": instruction.text,
        "instruction_language": INSTRUCTION_LANGUAGE,
        "instruction_text_sha256": instruction.text_sha256,
        "calibration_version": "calibration-c0-v1",
        "tool_version": "tool-c0-v1",
        "capture_profile_id": C0_CAPTURE_PROFILE_ID,
        "fps": C0_DATASET_FPS,
        "frame_count": 250 + index,
        "dataset_episode_ref": f"datasets/c0/episode-{index:02d}",
        "dataset_episode_sha256": DATASET_SHA,
        "timestamp_evidence_ref": f"evidence/timestamps-{index:02d}.json",
        "timestamp_evidence_sha256": TIMESTAMP_SHA,
        "authorization_evidence_ref": f"evidence/authorization-{index:02d}.json",
        "authorization_evidence_sha256": AUTHORIZATION_SHA,
        "vlm_shadow_evidence_ref": f"evidence/vlm-shadow-{index:02d}.json",
        "vlm_shadow_evidence_sha256": VLM_SHA,
        "human_outcome": "success",
        "human_outcome_evidence_ref": f"evidence/human-outcome-{index:02d}.json",
        "human_outcome_evidence_sha256": HUMAN_SHA,
        "raw_data_immutable": True,
        "finalized": True,
    }
    values.update(changes)
    return C0EpisodeManifestV1(**values)


def _batch(**changes: object) -> C0BatchManifestV1:
    values = {
        "schema_version": C0_BATCH_MANIFEST_SCHEMA_VERSION,
        "batch_id": "c0-batch-01",
        "episodes": tuple(_episode(index) for index in range(C0_EXPECTED_EPISODE_COUNT)),
        "split_rule_version": C0_SPLIT_RULE_VERSION,
        "finalized_at_utc": "2026-09-19T12:00:00Z",
        "frozen": True,
    }
    values.update(changes)
    return C0BatchManifestV1(**values)


def test_valid_episode_manifest_is_offline_only_and_json_serializable() -> None:
    episode = _episode()
    record = episode.to_manifest_record()

    assert record["sensor_streams"] == ["global_rgb", "grasp_rgb", "robot_state"]
    assert record["gripper_state_semantics"] == GRIPPER_STATE_SEMANTICS
    assert record["gripper_physical_mapping_verified"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    assert record["policy_execution_authorized"] is False
    assert record["training_authorized"] is False
    assert record["hardware_access_performed_by_serialization"] is False
    json.dumps(record, allow_nan=False)


def test_valid_sensor_timestamp_modes() -> None:
    host = C0SensorTimestampV1(
        schema_version=C0_SENSOR_TIMESTAMP_SCHEMA_VERSION,
        stream_name="global_rgb",
        sync_timestamp_s=1.25,
        host_receive_monotonic_s=1.25,
        sync_method="host_receive",
    )
    calibrated = C0SensorTimestampV1(
        stream_name="robot_state",
        sync_timestamp_s=2.0,
        host_receive_monotonic_s=2.01,
        sync_method="device_to_host_calibrated",
        device_timestamp_s=20.0,
        device_clock_id="aubo-rtde-clock",
    )

    assert host.to_manifest_record()["schema_version"] == C0_SENSOR_TIMESTAMP_SCHEMA_VERSION
    assert calibrated.device_timestamp_s == 20.0


@pytest.mark.parametrize("stream_name", ["wrist_rgb", "wrist_depth_m", "depth", "handeye", "fixed"])
def test_timestamp_rejects_legacy_or_depth_streams(stream_name: str) -> None:
    with pytest.raises(ValueError, match="stream_name must be one of"):
        C0SensorTimestampV1(
            stream_name=stream_name,
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.0,
            sync_method="host_receive",
        )


@pytest.mark.parametrize("field", ["sync_timestamp_s", "host_receive_monotonic_s"])
@pytest.mark.parametrize("bad_value", [True, -0.1, float("nan"), float("inf")])
def test_timestamp_rejects_invalid_common_times(field: str, bad_value: object) -> None:
    values = {
        "stream_name": "grasp_rgb",
        "sync_timestamp_s": 1.0,
        "host_receive_monotonic_s": 1.0,
        "sync_method": "host_receive",
    }
    values[field] = bad_value
    with pytest.raises(ValueError, match=field):
        C0SensorTimestampV1(**values)


@pytest.mark.parametrize("bad_value", [True, -0.1, float("nan"), float("inf")])
def test_timestamp_rejects_invalid_device_time(bad_value: object) -> None:
    with pytest.raises(ValueError, match="device_timestamp_s"):
        C0SensorTimestampV1(
            stream_name="robot_state",
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.1,
            sync_method="device_to_host_calibrated",
            device_timestamp_s=bad_value,
            device_clock_id="clock-1",
        )


def test_host_receive_requires_equal_timestamp() -> None:
    with pytest.raises(ValueError, match="must equal"):
        C0SensorTimestampV1(
            stream_name="global_rgb",
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.1,
            sync_method="host_receive",
        )


@pytest.mark.parametrize(
    ("device_timestamp_s", "device_clock_id"),
    [(1.0, None), (None, "clock-1"), (1.0, "")],
)
def test_device_timestamp_and_clock_id_are_paired(
    device_timestamp_s: float | None, device_clock_id: str | None
) -> None:
    with pytest.raises(ValueError, match="device_"):
        C0SensorTimestampV1(
            stream_name="robot_state",
            sync_timestamp_s=1.0,
            host_receive_monotonic_s=1.1,
            sync_method="device_to_host_calibrated",
            device_timestamp_s=device_timestamp_s,
            device_clock_id=device_clock_id,
        )


def test_episode_rejects_valid_but_non_frozen_camera_sha() -> None:
    with pytest.raises(ValueError, match="frozen CameraSetV2"):
        _episode(camera_set_sha256="ab" * 32)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_set_sha256", "A" * 64),
        ("dataset_episode_sha256", "1" * 63),
        ("timestamp_evidence_sha256", "g" * 64),
        ("authorization_evidence_sha256", "AB" * 32),
        ("vlm_shadow_evidence_sha256", "4" * 65),
        ("human_outcome_evidence_sha256", ""),
    ],
)
def test_episode_rejects_non_strict_sha256(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        _episode(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("instruction_text", "Pick one strip.", "does not match"),
        ("instruction_language", "zh", "Unsupported instruction language"),
        ("instruction_text_sha256", "0" * 64, "does not match"),
        ("instruction_schema_version", "InstructionSchemaV2", "Unsupported instruction schema"),
    ],
)
def test_episode_rejects_noncanonical_language(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _episode(**{field: value})


def test_episode_rejects_control_and_other_task_instructions() -> None:
    control = INSTRUCTION_SPECS["control_fixed"]
    with pytest.raises(ValueError, match="not valid for demonstrations"):
        _episode(
            instruction_id=control.instruction_id,
            instruction_text=control.text,
            instruction_text_sha256=control.text_sha256,
        )

    other = INSTRUCTION_SPECS["pick_long_zone_a"]
    with pytest.raises(ValueError, match="only accepts"):
        _episode(
            instruction_id=other.instruction_id,
            instruction_text=other.text,
            instruction_text_sha256=other.text_sha256,
        )


@pytest.mark.parametrize(
    ("ref_field", "sha_field"),
    [
        ("dataset_episode_ref", "dataset_episode_sha256"),
        ("timestamp_evidence_ref", "timestamp_evidence_sha256"),
        ("authorization_evidence_ref", "authorization_evidence_sha256"),
        ("vlm_shadow_evidence_ref", "vlm_shadow_evidence_sha256"),
        ("human_outcome_evidence_ref", "human_outcome_evidence_sha256"),
    ],
)
def test_episode_requires_each_evidence_ref_sha_pair(ref_field: str, sha_field: str) -> None:
    with pytest.raises(ValueError, match="provided together"):
        _episode(**{ref_field: None})
    with pytest.raises(ValueError, match="provided together"):
        _episode(**{sha_field: None})
    with pytest.raises(ValueError, match="non-empty string"):
        _episode(**{ref_field: ""})


@pytest.mark.parametrize("frame_count", [True, 0, -1, 1.0])
def test_episode_rejects_invalid_frame_count(frame_count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _episode(frame_count=frame_count)


def test_episode_requires_explicit_25hz_capture_profile() -> None:
    with pytest.raises(ValueError, match="capture_profile_id"):
        _episode(capture_profile_id="implicit-native-camera-fps")
    with pytest.raises(ValueError, match="fps must be 25"):
        _episode(fps=30.0)


@pytest.mark.parametrize("human_outcome", ["success", "failure", "uncertain"])
def test_human_outcome_frozen_enum_preserves_value(human_outcome: str) -> None:
    episode = _episode(human_outcome=human_outcome)
    assert episode.human_outcome == human_outcome
    assert episode.to_manifest_record()["human_outcome"] == human_outcome


def test_episode_rejects_invalid_outcome_and_physical_gripper_claim() -> None:
    with pytest.raises(ValueError, match="human_outcome must be one of"):
        _episode(human_outcome="passed")
    with pytest.raises(ValueError, match="must remain False"):
        _episode(gripper_physical_mapping_verified=True)
    with pytest.raises(ValueError, match="gripper_state_semantics"):
        _episode(gripper_state_semantics="vacuum_feedback")


def test_episode_manifest_record_is_detached_deep_copy() -> None:
    episode = _episode()
    record = episode.to_manifest_record()
    record["sensor_streams"].append("wrist_rgb")
    record["human_outcome"] = "failure"

    fresh = episode.to_manifest_record()
    assert fresh["sensor_streams"] == ["global_rgb", "grasp_rgb", "robot_state"]
    assert episode.human_outcome == "success"


def test_valid_batch_derives_counts_scenes_and_review_readiness() -> None:
    batch = _batch()
    record = batch.to_manifest_record()

    assert batch.episode_count == 10
    assert batch.total_frame_count == sum(250 + index for index in range(10))
    assert batch.scene_ids == tuple(f"c0-scene-{index:02d}" for index in range(10))
    assert batch.blockers == ()
    assert batch.data_ready_for_c0_review is True
    assert record["episode_count"] == 10
    assert record["policy_execution_authorized"] is False
    assert record["training_authorized"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("episode_count", [9, 11])
def test_batch_requires_exactly_ten_episodes(episode_count: int) -> None:
    episodes = tuple(_episode(index) for index in range(episode_count))
    with pytest.raises(ValueError, match="exactly 10 episodes"):
        _batch(episodes=episodes)


def test_batch_rejects_forged_expected_count_and_duplicate_episode_id() -> None:
    with pytest.raises(ValueError, match="must remain 10"):
        _batch(expected_episode_count=9)

    episodes = list(_batch().episodes)
    episodes[-1] = replace(episodes[-1], episode_id=episodes[0].episode_id)
    with pytest.raises(ValueError, match="episode_id must be globally unique"):
        _batch(episodes=tuple(episodes))


def test_batch_rejects_same_scene_across_splits() -> None:
    episodes = list(_batch().episodes)
    episodes[-1] = replace(
        episodes[-1], scene_id=episodes[0].scene_id, split_name="validation"
    )
    with pytest.raises(ValueError, match="scene_id cannot cross split"):
        _batch(episodes=tuple(episodes))


def test_batch_requires_new_scene_after_each_single_strip_episode() -> None:
    episodes = list(_batch().episodes)
    episodes[1] = replace(episodes[1], scene_id=episodes[0].scene_id, split_name="train")
    with pytest.raises(ValueError, match="new scene_id"):
        _batch(episodes=tuple(episodes))


@pytest.mark.parametrize(
    "field",
    [
        "camera_set_sha256",
        "bridge_version",
        "action_schema_version",
        "instruction_schema_version",
        "calibration_version",
        "tool_version",
        "capture_profile_id",
    ],
)
def test_batch_rejects_configuration_drift(field: str) -> None:
    episodes = list(_batch().episodes)
    if field == "camera_set_sha256":
        # Episode-level validation prevents constructing drift in frozen constants.
        with pytest.raises(ValueError, match="frozen CameraSetV2"):
            replace(episodes[-1], **{field: "ab" * 32})
        return
    if field == "bridge_version":
        with pytest.raises(ValueError, match="bridge_version"):
            replace(episodes[-1], **{field: "OtherBridgeV1"})
        return
    if field == "action_schema_version":
        with pytest.raises(ValueError, match="action_schema_version"):
            replace(episodes[-1], **{field: "ActionSchemaV2"})
        return
    if field == "instruction_schema_version":
        with pytest.raises(ValueError, match="Unsupported instruction schema"):
            replace(episodes[-1], **{field: "InstructionSchemaV2"})
        return
    if field == "capture_profile_id":
        with pytest.raises(ValueError, match="capture_profile_id"):
            replace(episodes[-1], **{field: "OtherProfile"})
        return

    episodes[-1] = replace(episodes[-1], **{field: f"different-{field}"})
    with pytest.raises(ValueError, match=f"configuration drift in {field}"):
        _batch(episodes=tuple(episodes))


def test_batch_blockers_are_stable_and_do_not_imply_authorization() -> None:
    first = _batch(frozen=False)
    second = _batch(frozen=False)

    assert first.blockers == ("batch_not_frozen",)
    assert first.blockers == second.blockers
    assert first.data_ready_for_c0_review is False
    record = first.to_manifest_record()
    assert record["blockers"] == ["batch_not_frozen"]
    assert record["policy_execution_authorized"] is False
    assert record["training_authorized"] is False


def test_batch_manifest_record_is_detached_deep_copy() -> None:
    batch = _batch()
    record = batch.to_manifest_record()
    record["episodes"][0]["human_outcome"] = "failure"
    record["scene_ids"].append("forged-scene")

    assert batch.episodes[0].human_outcome == "success"
    assert "forged-scene" not in batch.scene_ids


_IMPORT_PROBE = r"""
import json
import os
import sys
import threading

import lerobot.bamboo_sorting.c0_capture_contract  # noqa: F401

forbidden_modules = [name for name in sys.modules if name.split(".")[0] in {"cv2", "pyaubo_sdk"}]
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
        }
    )
)
"""


def test_importing_c0_contract_touches_no_hardware_network_or_threads() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []
