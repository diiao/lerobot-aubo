"""Offline integration checks of suction labels, normalization and loss."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from lerobot.bamboo_sorting.joint_gripper_quality import (
    gripper_episode_quality, require_joint_normalization_stats,
)
from lerobot.bamboo_sorting.smolvla_joint_adapter import (
    SmolVLAJointOfflineAdapter, make_joint_smolvla_config, make_joint_smolvla_processors,
)
from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK, joint_contract_record
from lerobot.policies.smolvla.losses import masked_action_loss
from lerobot.policies.smolvla.action_padding import restore_action_padding
from tests.processor.test_smolvla_processor import MockTokenizerProcessorStep


def _sequence(targets):
    actions = np.zeros((len(targets), 7), np.float32)
    actions[:, 6] = targets
    states = np.zeros_like(actions)
    states[1:, 6] = actions[:-1, 6]
    return states, actions


def _stats():
    # Deliberately unbalanced suction: normalized zero means 80, not off.
    return {key: {"mean": [0, 0, 0, 0, 0, 0, 80], "std": [1, 1, 1, 1, 1, 1, 40],
                  "min": [-1, -1, -1, -1, -1, -1, 0], "max": [1, 1, 1, 1, 1, 1, 100]}
            for key in ("observation.state", "action")}


def test_constant_labels_are_consistent_but_not_usable_pick_release_demonstrations():
    result = gripper_episode_quality(*_sequence([0] * 100))
    assert result["copy_current_state_accuracy"] == 1
    assert result["quality_issues"] and not result["complete_pick_release_cycle"]


def test_state_copy_can_score_high_while_missing_every_switch():
    result = gripper_episode_quality(*_sequence([0] * 50 + [100] * 50 + [0] * 50))
    assert result["activate_frames"] == [50] and result["release_frames"] == [100]
    assert result["copy_current_state_accuracy"] > .98
    assert result["complete_pick_release_cycle"] and not result["quality_issues"]


def test_release_after_recording_is_not_a_recorded_release_label():
    result = gripper_episode_quality(*_sequence([0, 100, 100]))
    assert not result["complete_pick_release_cycle"]


@pytest.mark.parametrize("key", ["action", "observation.state"])
@pytest.mark.parametrize("bad", ["wrong_width", "constant", "nonfinite", "missing"])
def test_invalid_normalization_stats_are_rejected(key, bad):
    stats = _stats()
    if bad == "wrong_width":
        stats[key]["mean"].append(0)
    elif bad == "constant":
        stats[key]["std"][6] = 0
    elif bad == "nonfinite":
        stats[key]["std"][6] = float("nan")
    else:
        del stats[key]["mean"]
    with pytest.raises(ValueError):
        require_joint_normalization_stats(stats)


def test_actual_normalizers_preserve_suction_and_decode_in_physical_units(tmp_path):
    cfg = make_joint_smolvla_config()
    with patch("lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep):
        pre, post = make_joint_smolvla_processors(cfg, _stats())
    _, actions = _sequence([0, 100, 100, 0])
    normalized = pre({"observation.state": torch.zeros(7), "action": torch.from_numpy(actions), "task": JOINT_TASK})
    torch.testing.assert_close(normalized["action"][..., 6], torch.tensor([-2., .5, .5, -2.]))
    torch.testing.assert_close(post(normalized["action"]), torch.from_numpy(actions))
    # The real output processor must also survive checkpoint serialization.
    post.save_pretrained(tmp_path)
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    torch.testing.assert_close(loaded(normalized["action"]), torch.from_numpy(actions))
    policy = Mock(config=cfg)
    policy.predict_action_chunk.return_value = torch.zeros(1, 2, 7)
    adapter = SmolVLAJointOfflineAdapter(policy, pre, loaded, contract=joint_contract_record())
    frame = {"observation.state": np.zeros(7, np.float32), "task": JOINT_TASK,
             **{f"observation.images.{name}": np.zeros((480, 640, 3), np.uint8)
                for name in ("global_rgb", "grasp_rgb")}}
    assert adapter(frame)[:, 6].tolist() == [100, 100]
    assert adapter.raw_actions[:, 6].tolist() == [80, 80]


def test_suction_coordinate_receives_gradient_and_padding_does_not():
    predictions = torch.zeros(1, 3, 32, requires_grad=True)
    targets = torch.zeros_like(predictions)
    targets[..., 6] = 1
    padding = torch.tensor([[False, False, True]])
    loss = masked_action_loss((predictions - targets).square(), 7, padding).mean()
    loss.backward()
    assert torch.all(predictions.grad[0, :2, 6] != 0)
    assert torch.count_nonzero(predictions.grad[..., 7:]) == 0
    assert torch.count_nonzero(predictions.grad[:, 2]) == 0
    restored = restore_action_padding(predictions.detach(), torch.ones_like(predictions), .5, 7)
    torch.testing.assert_close(restored[..., :7], predictions.detach()[..., :7])
    torch.testing.assert_close(restored[..., 7:], torch.full_like(restored[..., 7:], .5))


def test_real_dataset_roundtrip_with_latched_suction_and_mocked_controller(tmp_path):
    """Real record loop, normalization stats and video files; fake hardware only."""
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.bamboo_sorting.aubo_joint_audit import audit_joint_dataset
    from lerobot.bamboo_sorting.aubo_joint_capture import write_joint_json
    from lerobot.bamboo_sorting.aubo_joint_contract import joint_dataset_features
    from lerobot.robots.aubo_i10.joint_capture import PhoneToJointAction
    from lerobot.robots.aubo_i10.robot_processor import AuboGripperVelocityToPosition
    from lerobot.processor import RobotProcessorPipeline
    from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
    from lerobot.scripts.lerobot_record import record_loop, finalize_recorded_dataset
    from tests.bamboo_sorting.test_aubo_joint_capture import _robot, _observation, _phone_target, _entry, _session
    from tests.scripts.test_lerobot_record_observer import FakeTeleop

    robot, session, entry = _robot(), _session(tmp_path), _entry()
    del robot._control_softpaws_based_on_gripper  # Exercise the actual driver suction implementation.
    robot.is_suction_on = False
    robot.suction_on_pin, robot.suction_off_pin = 2, 3
    robot.suction_do_readback_attempts, robot.suction_do_readback_interval_s = 1, 0
    output = {2: False, 3: True}
    writes = []
    def write(pin, value):
        writes.append((pin, value))
        output[pin] = value
        return 0
    robot.io_control = Mock()
    robot.io_control.setStandardDigitalOutput.side_effect = write
    robot.io_control.getStandardDigitalOutput.side_effect = lambda pin: output[pin]
    pipeline = RobotProcessorPipeline(steps=[AuboGripperVelocityToPosition(latch=True)],
        to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action)
    velocities = [0, 1, 0, 0, -1, 0]
    events, sent = {"exit_early": False}, []
    def observation():
        robot.last_c0_sensor_timestamps = dict.fromkeys(("robot_state", "global_rgb", "grasp_rgb"),
                                                        1.0 + len(sent) / 25)
        return {**_observation(), "gripper_pos": 100. if robot.is_suction_on else 0.}
    robot.get_observation = observation
    actual_send = robot.send_action
    def send(command):
        result = actual_send(command)
        sent.append(result)
        events["exit_early"] = len(sent) == len(velocities)
        return result
    robot.send_action = send
    dataset = LeRobotDataset.create(repo_id="local/joint-test", root=tmp_path / "data", fps=25,
        features=joint_dataset_features(), robot_type="aubo_i10", use_videos=True)
    write_joint_json(dataset.root / "meta/aubo_joint_contract.json", joint_contract_record())
    record_loop(robot=robot, events=events, fps=25, dataset=dataset,
        teleop=FakeTeleop([{**_phone_target(), "ee.gripper_vel": v} for v in velocities]),
        teleop_action_processor=PhoneToJointAction(pipeline, robot),
        robot_action_processor=entry.IdentityAction(), robot_observation_processor=entry.learning_observation,
        control_time_s=2, single_task=JOINT_TASK, display_data=False,
        observer=session.episode_frame_observer())
    session.save_episode(dataset=dataset, episode_index=0)
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=1)
    report = audit_joint_dataset(dataset.root, tmp_path / "evidence")
    assert report["audit_passed"] and report["gripper_quality_passed"]
    assert report["supervised_candidate_episodes"] == [0]
    assert [a["gripper_pos"] for a in sent] == [0, 100, 100, 100, 0, 0]
    assert writes == [(3, False), (2, True), (2, False), (3, True)]
    videos = list(dataset.root.glob("videos/*/chunk-*/*.mp4"))
    assert len(videos) == 2
    for path in videos:
        with av.open(str(path)) as video:
            assert sum(1 for _ in video.decode(video=0)) == len(velocities)
    loaded = LeRobotDataset(repo_id="local/joint-test", root=dataset.root,
        delta_timestamps={"action": [i / 25 for i in range(50)]}, video_backend="pyav")
    switch = loaded[1]
    assert switch["observation.state"][6] == 0
    assert switch["action"][0, 6] == 100
    assert switch["action"].shape == (50, 7)
    assert switch["task"] == JOINT_TASK
    assert switch["action"][0, 4] == 91  # J5 is recorded, not silently fixed or omitted.
    terminal = loaded[len(velocities) - 1]
    assert not terminal["action_is_pad"][0] and terminal["action_is_pad"][1:].all()
    import shutil
    moved = tmp_path / "relocated"
    shutil.copytree(dataset.root, moved)
    with pytest.raises(ValueError, match="matching finalized"):
        audit_joint_dataset(moved, tmp_path / "evidence")
    relocated = audit_joint_dataset(moved, tmp_path / "evidence", source_dataset_root=dataset.root)
    assert relocated["gripper_quality_passed"] and relocated["source_dataset_root"] == str(dataset.root)
    with pytest.raises(ValueError, match="matching finalized"):
        audit_joint_dataset(moved, tmp_path / "evidence", source_dataset_root=tmp_path / "foreign")
