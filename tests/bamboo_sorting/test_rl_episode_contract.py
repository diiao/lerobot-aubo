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

from lerobot.bamboo_sorting.contracts import ACTION_SCHEMA_VERSION
from lerobot.bamboo_sorting.lerobot_bridge import CAMERASET_V1_LEROBOT_BRIDGE_VERSION
from lerobot.bamboo_sorting.rgb_gate import CAMERA_SET_SCHEMA_VERSION, FROZEN_CAMERA_SET_V1_SHA256
from lerobot.bamboo_sorting.rl_contract import (
    REWARD_SCHEMA_VERSION,
    RL_OBSERVATION_REF_SCHEMA_VERSION,
    RL_TRANSITION_SCHEMA_VERSION,
    RewardSchemaV1,
    RLObservationRefV1,
    RLTransitionV1,
)
from lerobot.bamboo_sorting.rl_episode_contract import (
    RESET_RECORD_SCHEMA_VERSION,
    RL_EPISODE_MANIFEST_SCHEMA_VERSION,
    RL_REPLAY_MANIFEST_SCHEMA_VERSION,
    ResetRecordV1,
    RLEpisodeManifestV1,
    RLReplayManifestV1,
)

ACTION = (0.0, 0.1, -0.5, 0.2, 0.0, 0.0, 0.0, 100.0)
EVIDENCE_SHA = "12" * 32
DATASET_SHA = "34" * 32


def _observation_ref(index: int, episode_id: str = "episode-0001") -> RLObservationRefV1:
    return RLObservationRefV1(
        schema_version=RL_OBSERVATION_REF_SCHEMA_VERSION,
        observation_id=f"observation-{episode_id}-{index:04d}",
        scene_id=f"scene-{episode_id}-{index:04d}",
        instruction_id="pick_any_collection",
        camera_set_schema_version=CAMERA_SET_SCHEMA_VERSION,
        camera_set_sha256=FROZEN_CAMERA_SET_V1_SHA256,
        bridge_version=CAMERASET_V1_LEROBOT_BRIDGE_VERSION,
        model_input_ref=f"model-inputs/{episode_id}/step-{index + 1:04d}.json",
        model_input_sha256="ef" * 32,
        sync_timestamp_s=10.25 + 0.04 * index,
        action_schema_version=ACTION_SCHEMA_VERSION,
    )


def _transition(
    index: int,
    *,
    episode_id: str = "episode-0001",
    done: bool = False,
    truncated: bool = False,
    task_success: int = 0,
    safety_cost: float = 0.0,
) -> RLTransitionV1:
    observation = _observation_ref(index, episode_id)
    next_observation = _observation_ref(index + 1, episode_id)
    return RLTransitionV1(
        schema_version=RL_TRANSITION_SCHEMA_VERSION,
        transition_id=f"transition-{episode_id}-{index:04d}",
        episode_id=episode_id,
        step_index=index,
        observation=observation,
        action=ACTION,
        next_observation=next_observation,
        reward=RewardSchemaV1(
            schema_version=REWARD_SCHEMA_VERSION,
            observation_id=next_observation.observation_id,
            scene_id=next_observation.scene_id,
            instruction_id="pick_any_collection",
            task_success=task_success,
            uncertain=False,
            safety_cost=safety_cost,
            intervention=False,
            reward_source="human",
            reward_model_ref=None,
            reward_model_sha256=None,
        ),
        done=done,
        truncated=truncated,
        control_source="policy",
        action_gate_passed=True,
        action_execution_confirmed=True,
        execution_evidence_ref=f"execution-logs/{episode_id}/step-{index:04d}.json",
    )


def _reset(**changes: object) -> ResetRecordV1:
    values = {
        "schema_version": RESET_RECORD_SCHEMA_VERSION,
        "reset_id": "reset-0001",
        "target_episode_id": "episode-0001",
        "reset_reason": "initial_setup",
        "requested_monotonic_s": 5.0,
        "completed_monotonic_s": 5.5,
        "reset_completed": True,
        "human_verified": True,
        "pre_reset_observation_id": None,
        "post_reset_observation": _observation_ref(0),
        "evidence_ref": "resets/reset-0001.json",
        "evidence_sha256": EVIDENCE_SHA,
        "automatic_reset_authorized": False,
    }
    values.update(changes)
    return ResetRecordV1(**values)


def _episode(
    n: int = 3,
    *,
    episode_id: str = "episode-0001",
    termination_reason: str = "task_failure",
    **changes: object,
) -> RLEpisodeManifestV1:
    terminal_done = termination_reason in ("task_success", "task_failure")
    transitions = []
    for index in range(n):
        is_last = index == n - 1
        transitions.append(
            _transition(
                index,
                episode_id=episode_id,
                done=is_last and terminal_done,
                truncated=is_last and not terminal_done,
                task_success=1 if (is_last and termination_reason == "task_success") else 0,
                safety_cost=0.5 if (is_last and termination_reason == "safety_abort") else 0.0,
            )
        )
    values = {
        "schema_version": RL_EPISODE_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "instruction_id": "pick_any_collection",
        "start_reset": _reset(
            reset_id=f"reset-{episode_id}",
            target_episode_id=episode_id,
            post_reset_observation=transitions[0].observation,
        ),
        "transitions": tuple(transitions),
        "termination_reason": termination_reason,
        "finalized": True,
    }
    values.update(changes)
    return RLEpisodeManifestV1(**values)


def _replay(**changes: object) -> RLReplayManifestV1:
    values = {
        "schema_version": RL_REPLAY_MANIFEST_SCHEMA_VERSION,
        "replay_id": "replay-0001",
        "source_dataset_ref": "datasets/bamboo_rl_v1",
        "source_dataset_sha256": DATASET_SHA,
        "split_name": "train",
        "episodes": (_episode(),),
        "frozen": True,
        "upstream_truncated_roundtrip_verified": False,
        "upstream_truncated_roundtrip_evidence_ref": None,
        "upstream_truncated_roundtrip_evidence_sha256": None,
    }
    values.update(changes)
    return RLReplayManifestV1(**values)


# ResetRecordV1


def test_reset_success_accepts_and_serializes() -> None:
    reset = _reset()

    record = reset.to_manifest_record()
    assert record["reset_completed"] is True
    assert record["post_reset_observation"]["observation_id"] == "observation-episode-0001-0000"
    assert record["serialized_record_grants_live_authorization"] is False
    assert record["hardware_access_performed_by_serialization"] is False
    json.dumps(record, allow_nan=False)


def test_reset_failure_accepts_with_empty_completion_fields() -> None:
    reset = _reset(
        completed_monotonic_s=None,
        reset_completed=False,
        human_verified=False,
        post_reset_observation=None,
    )
    assert reset.to_manifest_record()["reset_completed"] is False


def test_reset_non_initial_reason_requires_pre_reset_observation_id() -> None:
    for reason in ("after_done", "after_truncated", "operator_requested", "safety_abort_recovery"):
        with pytest.raises(ValueError, match="pre_reset_observation_id"):
            _reset(reset_reason=reason, pre_reset_observation_id=None)
        accepted = _reset(reset_reason=reason, pre_reset_observation_id="observation-episode-0001-0003")
        assert accepted.reset_reason == reason


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"automatic_reset_authorized": True}, "automatic_reset_authorized must be False"),
        ({"automatic_reset_authorized": "yes"}, "automatic_reset_authorized must be bool"),
        ({"completed_monotonic_s": 5.0}, "later than requested_monotonic_s"),
        ({"completed_monotonic_s": 4.9}, "later than requested_monotonic_s"),
        ({"reset_completed": False, "completed_monotonic_s": 5.5}, "failed reset"),
        ({"reset_completed": False, "post_reset_observation": _observation_ref(0)}, "failed reset"),
        (
            {"reset_completed": False, "human_verified": True, "completed_monotonic_s": None},
            "failed reset",
        ),
        ({"reset_completed": True, "human_verified": False}, "human_verified=True"),
        ({"reset_completed": True, "completed_monotonic_s": None}, "completed reset"),
        ({"reset_completed": True, "post_reset_observation": None}, "completed reset"),
        ({"reset_reason": ["initial_setup"]}, "reset_reason must be a string"),
        ({"reset_reason": "watchdog"}, "reset_reason must be one of"),
        ({"evidence_ref": ""}, "evidence_ref must be a non-empty string"),
        ({"evidence_sha256": "AB" * 32}, "lowercase SHA-256"),
        ({"requested_monotonic_s": True}, "finite non-negative"),
        ({"requested_monotonic_s": float("nan")}, "finite non-negative"),
        ({"completed_monotonic_s": float("inf")}, "finite non-negative"),
        ({"reset_id": ""}, "reset_id must be a non-empty string"),
        ({"schema_version": "ResetRecordV2"}, "Unsupported reset record schema"),
        ({"post_reset_observation": object()}, "post_reset_observation must be"),
    ],
)
def test_reset_rejects_invalid_records(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _reset(**changes)


def test_reset_rejects_post_observation_not_strictly_after_completion() -> None:
    # post_reset_observation defaults to sync_timestamp_s=10.25
    for completed in (10.25, 12.0):
        with pytest.raises(
            ValueError, match="post_reset_observation.sync_timestamp_s must be later"
        ):
            _reset(completed_monotonic_s=completed)


def test_reset_manifest_record_is_detached_deep_copy() -> None:
    reset = _reset()
    record = reset.to_manifest_record()

    record["reset_id"] = "tampered"
    record["post_reset_observation"]["observation_id"] = "tampered"

    assert reset.reset_id == "reset-0001"
    assert reset.post_reset_observation is not None
    assert reset.post_reset_observation.observation_id == "observation-episode-0001-0000"


# RLEpisodeManifestV1


@pytest.mark.parametrize(
    "termination_reason",
    ["task_success", "task_failure", "time_limit", "safety_abort", "operator_abort"],
)
def test_episode_accepts_all_termination_reasons(termination_reason: str) -> None:
    episode = _episode(termination_reason=termination_reason)

    assert episode.transition_count == 3
    record = episode.to_manifest_record()
    assert record["termination_reason"] == termination_reason
    assert record["transition_count"] == 3
    assert record["policy_execution_authorized"] is False
    assert record["hardware_access_performed"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    json.dumps(record, allow_nan=False)


def test_episode_allows_scene_change_mid_episode() -> None:
    first = _transition(0)
    moved = replace(first.next_observation, scene_id="scene-after-place-0001")
    first = replace(
        first,
        next_observation=moved,
        reward=replace(first.reward, scene_id=moved.scene_id),
    )
    second = replace(_transition(1), observation=moved)
    third = replace(_transition(2), done=True)
    episode = _episode(transitions=(first, second, third))
    assert episode.transitions[1].observation.scene_id == "scene-after-place-0001"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"finalized": False}, "finalized must be True"),
        ({"termination_reason": "rain_stop"}, "termination_reason must be one of"),
        ({"termination_reason": ["task_failure"]}, "termination_reason must be a string"),
        ({"instruction_id": "control_fixed"}, "Control instruction"),
        ({"transitions": ()}, "non-empty tuple"),
        ({"start_reset": object()}, "start_reset must be a ResetRecordV1"),
        ({"schema_version": "RLEpisodeManifestV2"}, "Unsupported episode manifest schema"),
    ],
)
def test_episode_rejects_invalid_fields(changes: dict[str, object], message: str) -> None:
    episode = _episode()
    with pytest.raises(ValueError, match=message):
        replace(episode, **changes)


def test_episode_rejects_foreign_start_reset_target() -> None:
    with pytest.raises(ValueError, match="target_episode_id"):
        _episode(start_reset=_reset(target_episode_id="episode-9999"))


def test_episode_rejects_unverified_start_reset() -> None:
    with pytest.raises(ValueError, match="completed and human verified"):
        _episode(
            start_reset=_reset(
                reset_completed=False,
                human_verified=False,
                completed_monotonic_s=None,
                post_reset_observation=None,
            )
        )


def test_episode_rejects_duplicate_transition_ids() -> None:
    duplicate = replace(_transition(1), transition_id=_transition(0).transition_id)
    with pytest.raises(ValueError, match="duplicate transition_id"):
        _episode(transitions=(_transition(0), duplicate))


def test_episode_rejects_step_index_skips() -> None:
    skipped = replace(_transition(1), step_index=2)
    with pytest.raises(ValueError, match=r"0\.\.N-1"):
        _episode(transitions=(_transition(0), skipped))


def test_episode_rejects_broken_chain() -> None:
    foreign = replace(_transition(1), observation=replace(_observation_ref(1), scene_id="scene-foreign"))
    with pytest.raises(ValueError, match="chain break"):
        _episode(transitions=(_transition(0), foreign))


def test_episode_rejects_early_termination() -> None:
    early = replace(_transition(0), done=True)
    with pytest.raises(ValueError, match="only the final transition"):
        _episode(transitions=(early, _transition(1)))


def test_episode_rejects_unterminated_final_transition() -> None:
    with pytest.raises(ValueError, match="must end the episode"):
        _episode(transitions=(_transition(0), _transition(1)))


@pytest.mark.parametrize(
    ("termination_reason", "changes", "message"),
    [
        ("task_success", {"task_success": 0}, "task_success requires"),
        ("task_failure", {"truncated": True, "done": False}, "task_failure requires"),
        ("time_limit", {"done": True, "truncated": False}, "time_limit requires"),
        ("operator_abort", {"done": True, "truncated": False}, "operator_abort requires"),
        ("safety_abort", {"safety_cost": 0.0}, "safety_abort requires"),
    ],
)
def test_episode_rejects_termination_mismatch(
    termination_reason: str, changes: dict[str, object], message: str
) -> None:
    transitions = list(_episode(termination_reason=termination_reason).transitions)
    last = transitions[-1]
    transitions[-1] = replace(
        last,
        done=changes.get("done", last.done),
        truncated=changes.get("truncated", last.truncated),
        reward=replace(
            last.reward,
            task_success=changes.get("task_success", last.reward.task_success),
            safety_cost=changes.get("safety_cost", last.reward.safety_cost),
        ),
    )
    with pytest.raises(ValueError, match=message):
        _episode(termination_reason=termination_reason, transitions=tuple(transitions))


def test_episode_rejects_foreign_instruction_transition() -> None:
    first = _transition(0)
    other = replace(first.next_observation, instruction_id="pick_long_zone_a")
    first = replace(
        first,
        observation=replace(first.observation, instruction_id="pick_long_zone_a"),
        next_observation=other,
        reward=replace(first.reward, instruction_id="pick_long_zone_a"),
    )
    second = _transition(1)
    second_next = replace(second.next_observation, instruction_id="pick_long_zone_a")
    second = replace(
        second,
        observation=other,
        next_observation=second_next,
        reward=replace(
            second.reward,
            observation_id=second_next.observation_id,
            instruction_id="pick_long_zone_a",
        ),
    )
    with pytest.raises(ValueError, match="canonical instruction"):
        _episode(
            transitions=(first, second),
            start_reset=_reset(post_reset_observation=first.observation),
        )


def test_episode_rejects_start_reset_observation_mismatch() -> None:
    with pytest.raises(
        ValueError, match=r"start_reset\.post_reset_observation must equal transitions\[0\]\.observation"
    ):
        _episode(start_reset=_reset(post_reset_observation=_observation_ref(5)))


def test_episode_rejects_foreign_episode_transition() -> None:
    intruder = _transition(1, episode_id="episode-9999")
    intruder = replace(intruder, observation=_transition(0).next_observation)
    with pytest.raises(ValueError, match="foreign episode_id"):
        _episode(transitions=(_transition(0), intruder))


# RLReplayManifestV1


def test_replay_counts_and_scenes_are_computed_from_content() -> None:
    replay = _replay(
        episodes=(
            _episode(n=3, episode_id="episode-0001"),
            _episode(n=2, episode_id="episode-0002"),
        )
    )

    assert replay.episode_count == 2
    assert replay.transition_count == 5
    assert replay.scene_ids == (
        "scene-episode-0001-0000",
        "scene-episode-0001-0001",
        "scene-episode-0001-0002",
        "scene-episode-0001-0003",
        "scene-episode-0002-0000",
        "scene-episode-0002-0001",
        "scene-episode-0002-0002",
    )


def test_replay_default_blocks_data_readiness_on_unverified_truncation_roundtrip() -> None:
    replay = _replay()

    assert replay.frozen
    assert not replay.data_ready_for_training
    assert replay.blockers == ("upstream_truncated_roundtrip_unverified",)
    record = replay.to_manifest_record()
    assert record["data_ready_for_training"] is False
    assert record["blockers"] == ["upstream_truncated_roundtrip_unverified"]
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert record["hardware_access_performed"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    json.dumps(record, allow_nan=False)


def test_replay_data_ready_only_with_evidenced_verified_truncation_roundtrip() -> None:
    replay = _replay(
        upstream_truncated_roundtrip_verified=True,
        upstream_truncated_roundtrip_evidence_ref="audits/truncated_roundtrip.json",
        upstream_truncated_roundtrip_evidence_sha256="56" * 32,
    )

    assert replay.data_ready_for_training
    assert replay.blockers == ()
    record = replay.to_manifest_record()
    assert record["data_ready_for_training"] is True
    assert record["blockers"] == []
    assert record["training_authorized"] is False
    assert record["upstream_truncated_roundtrip_evidence_ref"] == "audits/truncated_roundtrip.json"
    assert record["upstream_truncated_roundtrip_evidence_sha256"] == "56" * 32


def test_replay_verified_roundtrip_requires_both_evidence_fields() -> None:
    with pytest.raises(ValueError, match="requires both evidence ref and sha256"):
        _replay(upstream_truncated_roundtrip_verified=True)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "upstream_truncated_roundtrip_evidence_ref": "audits/truncated_roundtrip.json",
            "upstream_truncated_roundtrip_evidence_sha256": "56" * 32,
        },
        {
            "upstream_truncated_roundtrip_evidence_ref": "audits/other.json",
            "upstream_truncated_roundtrip_evidence_sha256": "78" * 32,
        },
    ],
)
def test_replay_unverified_roundtrip_rejects_any_evidence(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must leave evidence ref and sha256 empty"):
        _replay(**changes)


def test_replay_evidence_ref_and_sha_are_strictly_validated() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        _replay(
            upstream_truncated_roundtrip_evidence_ref="audits/truncated_roundtrip.json",
        )
    with pytest.raises(ValueError, match="upstream_truncated_roundtrip_evidence_ref must be"):
        _replay(
            upstream_truncated_roundtrip_verified=True,
            upstream_truncated_roundtrip_evidence_ref=" ",
            upstream_truncated_roundtrip_evidence_sha256="56" * 32,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _replay(
            upstream_truncated_roundtrip_verified=True,
            upstream_truncated_roundtrip_evidence_ref="audits/truncated_roundtrip.json",
            upstream_truncated_roundtrip_evidence_sha256="AB" * 32,
        )


def test_replay_rejects_duplicate_episode_ids() -> None:
    episode = _episode()
    with pytest.raises(ValueError, match="duplicate episode_id"):
        _replay(episodes=(episode, episode))


def test_replay_rejects_duplicate_transition_ids_across_episodes() -> None:
    episode_a = _episode(n=1, episode_id="episode-0001")
    episode_b = _episode(n=1, episode_id="episode-0002")
    clash = replace(
        episode_b.transitions[0], transition_id=episode_a.transitions[0].transition_id
    )
    episode_b_clash = replace(episode_b, transitions=(clash,))
    with pytest.raises(ValueError, match="duplicate transition_id"):
        _replay(episodes=(episode_a, episode_b_clash))


def test_replay_rejects_duplicate_observation_ids_across_episodes() -> None:
    episode_a = _episode(n=1, episode_id="episode-0001")
    episode_b = _episode(n=1, episode_id="episode-0002")
    clash = replace(episode_b.transitions[0], observation=episode_a.transitions[0].observation)
    episode_b_clash = replace(
        episode_b,
        transitions=(clash,),
        start_reset=replace(episode_b.start_reset, post_reset_observation=clash.observation),
    )
    with pytest.raises(ValueError, match="duplicate observation_id"):
        _replay(episodes=(episode_a, episode_b_clash))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frozen": False}, "frozen must be True"),
        ({"split_name": "holdout"}, "split_name must be one of"),
        ({"split_name": {"train": True}}, "split_name must be a string"),
        ({"source_dataset_sha256": "zz" * 32}, "lowercase SHA-256"),
        ({"source_dataset_ref": ""}, "source_dataset_ref must be a non-empty string"),
        ({"replay_id": ""}, "replay_id must be a non-empty string"),
        ({"episodes": ()}, "non-empty tuple"),
        ({"upstream_truncated_roundtrip_verified": "yes"}, "must be bool"),
        ({"schema_version": "RLReplayManifestV2"}, "Unsupported replay manifest schema"),
    ],
)
def test_replay_rejects_invalid_records(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _replay(**changes)


def test_replay_manifest_record_is_detached_deep_copy() -> None:
    replay = _replay()
    record = replay.to_manifest_record()

    record["episode_count"] = 99
    record["episodes"][0]["transitions"][0]["reward"]["task_success"] = 1

    assert replay.episode_count == 1
    assert replay.episodes[0].transitions[0].reward.task_success == 0


# Offline guarantee


_IMPORT_PROBE = r"""
import json
import os
import sys
import threading

import lerobot.bamboo_sorting.rl_episode_contract  # noqa: F401

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


def test_importing_rl_episode_contract_touches_no_hardware_network_or_threads() -> None:
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
