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
from types import MappingProxyType

import pytest

from lerobot.bamboo_sorting.contracts import ACTION_SCHEMA_VERSION
from lerobot.bamboo_sorting.lerobot_bridge import CAMERASET_V2_LEROBOT_BRIDGE_VERSION
from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_V2_SCHEMA_VERSION,
    FROZEN_CAMERA_SET_V2_SHA256,
)
from lerobot.bamboo_sorting.rl_contract import (
    MODE_REQUIRED_GATES,
    REWARD_SCHEMA_VERSION,
    RL_OBSERVATION_REF_SCHEMA_VERSION,
    RL_READINESS_SCHEMA_VERSION,
    RL_TRANSITION_SCHEMA_VERSION,
    RewardSchemaV1,
    RLObservationRefV1,
    RLReadinessReportV1,
    RLTransitionV1,
)

LEARNED_SHA = "ab" * 32
ACTION = (0.0, 0.1, -0.5, 0.2, 0.0, 0.0, 0.0, 100.0)


def _observation_ref(
    observation_id: str = "observation-0001",
    scene_id: str = "scene-0001",
    step_index: int = 1,
    sync_timestamp_s: float = 10.25,
) -> RLObservationRefV1:
    return RLObservationRefV1(
        schema_version=RL_OBSERVATION_REF_SCHEMA_VERSION,
        observation_id=observation_id,
        scene_id=scene_id,
        instruction_id="pick_any_collection",
        camera_set_schema_version=CAMERA_SET_V2_SCHEMA_VERSION,
        camera_set_sha256=FROZEN_CAMERA_SET_V2_SHA256,
        bridge_version=CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
        model_input_ref=f"model-inputs/episode-0001/step-{step_index:04d}.json",
        model_input_sha256="ef" * 32,
        sync_timestamp_s=sync_timestamp_s,
        action_schema_version=ACTION_SCHEMA_VERSION,
    )


def _reward(**changes: object) -> RewardSchemaV1:
    values = {
        "schema_version": REWARD_SCHEMA_VERSION,
        "observation_id": "observation-0002",
        "scene_id": "scene-0002",
        "instruction_id": "pick_any_collection",
        "task_success": 0,
        "uncertain": False,
        "safety_cost": 0.0,
        "intervention": False,
        "reward_source": "human",
        "reward_model_ref": None,
        "reward_model_sha256": None,
    }
    values.update(changes)
    return RewardSchemaV1(**values)


def _transition(**changes: object) -> RLTransitionV1:
    values = {
        "schema_version": RL_TRANSITION_SCHEMA_VERSION,
        "transition_id": "transition-0001",
        "episode_id": "episode-0001",
        "step_index": 0,
        "observation": _observation_ref(),
        "action": ACTION,
        "next_observation": _observation_ref("observation-0002", "scene-0002", 2, 10.29),
        "reward": _reward(),
        "done": False,
        "truncated": False,
        "control_source": "policy",
        "action_gate_passed": True,
        "action_execution_confirmed": True,
        "execution_evidence_ref": "execution-logs/episode-0001/step-0001.json",
    }
    values.update(changes)
    return RLTransitionV1(**values)


def _report(mode: str = "rabc", **changes: object) -> RLReadinessReportV1:
    values = {
        "schema_version": RL_READINESS_SCHEMA_VERSION,
        "mode": mode,
        "data_split_frozen": True,
        "reward_schema_frozen": True,
        "independent_reward_test_set": True,
        "bc_baseline_available": True,
        "aubo_env_adapter": True,
        "explicit_reset": True,
        "intervention_recording": True,
        "replay_buffer": True,
        "actor_learner": True,
        "thin_safety_gate_integration": True,
        "sim_or_null_actuator_passed": True,
        "chunk_critic": True,
        "behavior_or_kl_constraint": True,
        "offline_policy_evaluation": True,
        "language_counterfactual_tests": True,
        "safety_gate_isolation": True,
        "checkpoint_provenance": True,
        "reward_hacking_checks": True,
    }
    values.update(changes)
    return RLReadinessReportV1(**values)


# RLObservationRefV1


def test_observation_ref_binds_two_rgb_cameraset_identity() -> None:
    ref = _observation_ref()

    assert ref.sync_timestamp_s == 10.25
    record = ref.to_manifest_record()
    assert record["observation_id"] == "observation-0001"
    assert record["camera_set_sha256"] == FROZEN_CAMERA_SET_V2_SHA256
    assert record["model_input_sha256"] == "ef" * 32
    json.dumps(record, allow_nan=False)


def test_observation_ref_manifest_has_no_image_or_depth_payload() -> None:
    record = _observation_ref().to_manifest_record()
    serialized = json.dumps(record)

    assert "wrist_rgb" not in serialized
    assert "wrist_depth" not in serialized
    for key, value in record.items():
        if key != "serialized_record_grants_live_authorization":
            assert isinstance(value, (str, int, float)), key


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"observation_id": ""}, "observation_id must be a non-empty string"),
        ({"scene_id": "  "}, "scene_id must be a non-empty string"),
        ({"instruction_id": ["pick_any_collection"]}, "instruction_id must be a string"),
        ({"instruction_id": "made_up"}, "Unknown instruction_id"),
        ({"instruction_id": "control_fixed"}, "Control instructions"),
        ({"camera_set_schema_version": "ThreeRgbV1"}, "camera_set_schema_version must be"),
        ({"camera_set_sha256": "ab" * 32}, "must equal FROZEN_CAMERA_SET_V2_SHA256"),
        (
            {"camera_set_sha256": "1" + FROZEN_CAMERA_SET_V2_SHA256[1:]},
            "must equal FROZEN_CAMERA_SET_V2_SHA256",
        ),
        ({"bridge_version": "BridgeV0"}, "bridge_version must be"),
        ({"model_input_ref": ""}, "model_input_ref must be a non-empty string"),
        ({"model_input_sha256": "ef" * 31}, "lowercase SHA-256"),
        ({"sync_timestamp_s": float("nan")}, "finite non-negative"),
        ({"sync_timestamp_s": -1.0}, "finite non-negative"),
        ({"sync_timestamp_s": True}, "finite non-negative"),
        ({"action_schema_version": "ActionSchemaV2"}, "action_schema_version must be"),
        ({"schema_version": "RLObservationRefV2"}, "Unsupported observation ref schema"),
    ],
)
def test_observation_ref_rejects_invalid_records(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_observation_ref(), **changes)


def test_observation_ref_manifest_record_is_detached_deep_copy() -> None:
    ref = _observation_ref()
    record = ref.to_manifest_record()

    record["observation_id"] = "tampered"
    record["camera_set_sha256"] = "00" * 32

    assert ref.observation_id == "observation-0001"
    assert ref.camera_set_sha256 == FROZEN_CAMERA_SET_V2_SHA256


# RewardSchemaV1


def test_human_reward_accepts_and_serializes() -> None:
    reward = _reward(task_success=0, uncertain=True, safety_cost=0.5)

    assert reward.requires_human_review
    record = reward.to_manifest_record()
    assert record["task_success"] == 0
    assert record["safety_cost"] == 0.5
    assert record["serialized_record_grants_live_authorization"] is False
    json.dumps(record, allow_nan=False)


def test_learned_reward_accepts_with_model_provenance() -> None:
    reward = _reward(
        reward_source="learned",
        reward_model_ref="rewards/run01/best",
        reward_model_sha256=LEARNED_SHA,
    )
    record = reward.to_manifest_record()
    assert record["reward_model_ref"] == "rewards/run01/best"
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task_success": 1, "uncertain": True}, "uncertain=true forbids"),
        ({"reward_source": "learned"}, "both reward_model_ref and reward_model_sha256"),
        (
            {
                "reward_source": "learned",
                "reward_model_ref": "rewards/run01/best",
                "reward_model_sha256": "A" * 64,
            },
            "lowercase SHA-256",
        ),
        (
            {
                "reward_source": "learned",
                "reward_model_ref": "rewards/run01/best",
                "reward_model_sha256": "ab" * 31,
            },
            "lowercase SHA-256",
        ),
        (
            {"reward_model_ref": "rewards/run01/best", "reward_model_sha256": LEARNED_SHA},
            "human rewards must leave",
        ),
        ({"safety_cost": float("nan")}, "finite non-negative"),
        ({"safety_cost": float("inf")}, "finite non-negative"),
        ({"safety_cost": -0.1}, "finite non-negative"),
        ({"safety_cost": True}, "not bool"),
        ({"task_success": True}, "not bool"),
        ({"task_success": 1.0}, "integer 0 or 1"),
        ({"task_success": 2}, "must be 0 or 1"),
        ({"intervention": 1}, "intervention must be bool"),
        ({"uncertain": 0}, "uncertain must be bool"),
        ({"observation_id": ""}, "observation_id must be a non-empty string"),
        ({"instruction_id": "made_up"}, "Unknown instruction_id"),
        ({"instruction_id": ["pick_any_collection"]}, "instruction_id must be a string"),
        ({"instruction_id": "control_fixed"}, "Control instructions"),
        ({"reward_source": ["human"]}, "reward_source must be a string"),
        ({"reward_source": "simulator"}, "reward_source must be one of"),
        ({"schema_version": "RewardSchemaV2"}, "Unsupported reward schema"),
    ],
)
def test_reward_rejects_invalid_records(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_reward(), **changes)


def test_reward_manifest_record_is_detached_deep_copy() -> None:
    reward = _reward()
    record = reward.to_manifest_record()

    record["task_success"] = 1
    record["safety_cost"] = 99.0
    record["observation_id"] = "tampered"

    assert reward.task_success == 0
    assert reward.safety_cost == 0.0
    assert reward.observation_id == "observation-0002"


# RLTransitionV1


def test_policy_transition_accepts_and_serializes() -> None:
    transition = _transition(
        reward=_reward(task_success=1),
        done=True,
    )

    record = transition.to_manifest_record()
    assert record["control_source"] == "policy"
    assert record["action_gate_passed"] is True
    assert record["done"] is True and record["truncated"] is False
    assert record["step_index"] == 0
    assert record["execution_evidence_ref"].endswith("step-0001.json")
    assert record["observation"]["observation_id"] == "observation-0001"
    assert record["next_observation"]["observation_id"] == "observation-0002"
    assert record["next_observation"]["model_input_ref"].endswith("step-0002.json")
    assert record["next_observation"]["sync_timestamp_s"] > record["observation"]["sync_timestamp_s"]
    assert record["next_observation"]["model_input_sha256"] == "ef" * 32
    assert record["reward"]["task_success"] == 1
    json.dumps(record, allow_nan=False)


def test_transition_scene_may_change_but_instruction_may_not() -> None:
    moved = _transition(
        next_observation=_observation_ref("observation-0002", "scene-0003", 2, 10.29),
        reward=_reward(scene_id="scene-0003"),
    )
    assert moved.next_observation.scene_id == "scene-0003"

    changed_instruction = replace(
        _observation_ref("observation-0002", "scene-0002", 2, 10.29),
        instruction_id="pick_long_zone_a",
    )
    with pytest.raises(ValueError, match="share one instruction_id"):
        _transition(next_observation=changed_instruction)


def test_transition_rejects_repeated_or_regressed_observations() -> None:
    with pytest.raises(ValueError, match="distinct observation_ids"):
        _transition(
            next_observation=_observation_ref("observation-0001", "scene-0002", 2, 10.29)
        )
    with pytest.raises(ValueError, match="must be later than observation.sync_timestamp_s"):
        _transition(next_observation=_observation_ref("observation-0002", "scene-0002", 2, 10.25))
    with pytest.raises(ValueError, match="must be later than observation.sync_timestamp_s"):
        _transition(next_observation=_observation_ref("observation-0002", "scene-0002", 2, 10.20))


def test_reward_must_annotate_next_observation() -> None:
    with pytest.raises(ValueError, match="observation_id mismatch"):
        _transition(reward=_reward(observation_id="observation-9999"))
    with pytest.raises(ValueError, match="scene_id mismatch"):
        _transition(reward=_reward(scene_id="scene-9999"))
    with pytest.raises(ValueError, match="instruction_id mismatch"):
        _transition(reward=_reward(instruction_id="pick_long_zone_a"))


def test_intervention_and_demonstration_allow_none_gate() -> None:
    intervention = _transition(
        control_source="intervention",
        action_gate_passed=None,
        reward=_reward(intervention=True),
    )
    demonstration = _transition(control_source="demonstration", action_gate_passed=None)

    assert intervention.action_gate_passed is None
    assert intervention.reward.intervention is True
    assert demonstration.control_source == "demonstration"


@pytest.mark.parametrize("bad_gate", [0, 1, "yes", object()])
def test_transition_rejects_non_bool_gate_values(bad_gate: object) -> None:
    with pytest.raises(ValueError, match="action_gate_passed must be bool or None"):
        _transition(control_source="demonstration", action_gate_passed=bad_gate)


def test_transition_rejects_unconfirmed_execution_for_any_source() -> None:
    for source in ("policy", "intervention", "demonstration"):
        with pytest.raises(ValueError, match="action_execution_confirmed=True"):
            _transition(
                control_source=source,
                action_execution_confirmed=False,
                reward=_reward(intervention=(source == "intervention")),
            )


def test_success_reward_requires_done() -> None:
    with pytest.raises(ValueError, match="requires done=True"):
        _transition(reward=_reward(task_success=1), done=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"transition_id": ""}, "transition_id must be a non-empty string"),
        ({"episode_id": ""}, "episode_id must be a non-empty string"),
        ({"execution_evidence_ref": ""}, "execution_evidence_ref must be a non-empty string"),
        ({"step_index": True}, "step_index must be a non-negative integer"),
        ({"step_index": 1.5}, "step_index must be a non-negative integer"),
        ({"step_index": -1}, "step_index must be a non-negative integer"),
        ({"action_gate_passed": False}, "action_gate_passed=True"),
        ({"action_gate_passed": None}, "action_gate_passed=True"),
        ({"control_source": "teleop"}, "control_source must be one of"),
        ({"control_source": ["policy"]}, "control_source must be a string"),
        ({"done": True, "truncated": True}, "must not both be true"),
        ({"done": 1}, "done must be bool"),
        ({"truncated": "yes"}, "truncated must be bool"),
        ({"action": (0.0, 0.1)}, "ordered scalar values"),
        ({"action": (True,) * 8}, "not bool"),
        ({"action": (float("nan"),) * 8}, "finite"),
        (
            {"reward": _reward(intervention=True)},
            "must not claim a human intervention reward",
        ),
        (
            {
                "control_source": "intervention",
                "action_gate_passed": None,
                "reward": _reward(intervention=False),
            },
            "require reward.intervention=True",
        ),
        (
            {"control_source": "demonstration", "reward": _reward(intervention=True)},
            "demonstration transitions must not claim intervention",
        ),
        ({"observation": object()}, "observation must be an RLObservationRefV1"),
        ({"reward": object()}, "reward must be a RewardSchemaV1"),
        ({"schema_version": "RLTransitionV2"}, "Unsupported transition schema"),
    ],
)
def test_transition_rejects_invalid_records(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _transition(**changes)


def test_intervention_transition_with_false_gate_rejected() -> None:
    with pytest.raises(ValueError, match="gate-failed candidate action"):
        _transition(
            control_source="intervention",
            action_gate_passed=False,
            reward=_reward(intervention=True),
        )


def test_transition_manifest_record_is_detached_deep_copy() -> None:
    transition = _transition()
    record = transition.to_manifest_record()

    record["action"][0] = 9.9
    record["reward"]["task_success"] = 1
    record["observation"]["observation_id"] = "tampered"
    record["observation"]["model_input_sha256"] = "00" * 32

    assert transition.action[0] == ACTION[0]
    assert transition.reward.task_success == 0
    assert transition.observation.observation_id == "observation-0001"
    assert transition.observation.model_input_sha256 == "ef" * 32


# RLReadinessReportV1


@pytest.mark.parametrize("mode", ["rabc", "hil_serl_sac", "smolvla_rl"])
def test_readiness_minimal_passing_cases(mode: str) -> None:
    report = _report(mode)

    assert report.ready
    assert report.blockers == ()
    record = report.to_manifest_record()
    assert record["ready"] is True
    assert record["blockers"] == []
    assert record["policy_execution_authorized"] is False
    assert record["hardware_access_performed"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("mode", ["rabc", "hil_serl_sac", "smolvla_rl"])
def test_readiness_reports_every_missing_gate(mode: str) -> None:
    all_false = dict.fromkeys(MODE_REQUIRED_GATES[mode], False)
    report = _report(mode, **all_false)

    assert not report.ready
    assert report.blockers == MODE_REQUIRED_GATES[mode]


def test_smolvla_rl_requires_replay_and_safety_isolation() -> None:
    for missing in ("replay_buffer", "thin_safety_gate_integration", "safety_gate_isolation"):
        report = _report("smolvla_rl", **{missing: False})

        assert not report.ready
        assert report.blockers == (missing,)


def test_readiness_blocker_order_is_stable() -> None:
    first = _report("hil_serl_sac", actor_learner=False, replay_buffer=False)
    second = _report("hil_serl_sac", replay_buffer=False, actor_learner=False)

    assert first.blockers == ("replay_buffer", "actor_learner")
    assert first.blockers == second.blockers


def test_mode_required_gates_is_immutable() -> None:
    assert isinstance(MODE_REQUIRED_GATES, MappingProxyType)
    original = MODE_REQUIRED_GATES["rabc"]

    with pytest.raises(TypeError):
        MODE_REQUIRED_GATES["rabc"] = ()
    with pytest.raises(TypeError):
        MODE_REQUIRED_GATES["dreamerv3"] = ("data_split_frozen",)
    with pytest.raises((TypeError, AttributeError)):
        MODE_REQUIRED_GATES.clear()  # type: ignore[attr-defined]

    assert MODE_REQUIRED_GATES["rabc"] == original


def test_readiness_rejects_non_string_mode_with_value_error() -> None:
    with pytest.raises(ValueError, match="mode must be a string"):
        _report(mode=["rabc"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "dreamerv3"}, "mode must be one of"),
        ({"schema_version": "RLReadinessReportV2"}, "Unsupported readiness schema"),
        ({"data_split_frozen": 1}, "must be an explicit bool"),
        ({"chunk_critic": None}, "must be an explicit bool"),
    ],
)
def test_readiness_rejects_invalid_records(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _report(**changes)


# Offline guarantee


_IMPORT_PROBE = r"""
import json
import os
import sys
import threading

import lerobot.bamboo_sorting.rl_contract  # noqa: F401

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


def test_importing_rl_contract_touches_no_hardware_network_or_threads() -> None:
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
