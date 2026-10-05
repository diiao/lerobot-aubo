"""Offline tests: real record-loop/data boundary, mocked SDK, no devices."""

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.bamboo_sorting.aubo_joint_contract import (
    JOINT_FIELDS, JOINT_NAMES, JOINT_TASK, expand_joint_command,
    joint_command, joint_contract_record, joint_dataset_features, require_joint_dataset,
)
from lerobot.bamboo_sorting.aubo_joint_capture import JointCaptureSession, JointFrameObserver
from lerobot.bamboo_sorting.aubo_joint_audit import audit_joint_dataset
from lerobot.bamboo_sorting.smolvla_joint_adapter import SmolVLAJointOfflineAdapter, make_joint_smolvla_config
from lerobot.robots.aubo_i10.joint_capture import AuboI10JointCaptureRobot, PhoneToJointAction
from lerobot.scripts.lerobot_record import record_loop
from tests.scripts.test_lerobot_record_observer import FakeTeleop, _no_transition_trace


def _entry():
    path = Path(__file__).parents[2] / "examples/phone_to_auboi10/record_joint.py"
    spec = importlib.util.spec_from_file_location("record_joint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _robot():
    robot = object.__new__(AuboI10JointCaptureRobot)
    robot.robot_rpc_client = Mock()
    robot.robot_rpc_client.hasConnected.return_value = True
    robot.disable_servo_mode = Mock(return_value=True)
    robot.robot_interface = Mock()
    current = [0, 0, 0, 0, math.pi / 2, 0]
    robot.robot_interface.getRobotState().getJointPositions.return_value = current
    robot.robot_interface.getRobotAlgorithm().inverseKinematics.return_value = ([0.01, 0.02, 0, 0, math.radians(91), 0], 0)
    motion = robot.robot_interface.getMotionControl()
    motion.setSpeedFraction.return_value = 0
    motion.isServoModeEnabled.return_value = True
    motion.servoJoint.return_value = 0
    robot.is_servo_mode_enabled = True
    robot.joint_lower_deg = (-360,) * 6
    robot.joint_upper_deg = (360,) * 6
    robot.max_joint_tracking_deg = 2
    robot._joint_sequence = 0
    robot._previous_joint_target = None
    robot.last_joint_command_trace = None
    robot.fixed_axis5_deg = 90
    robot.servo_max_queue_retry = 1
    robot.servo_time, robot.servo_blend_radius = 0.04, 0
    robot.joint_acceleration, robot.joint_velocity = 1, 1
    robot._control_softpaws_based_on_gripper = lambda value: setattr(robot, "last_gripper_command_trace", _no_transition_trace(False))
    return robot


def _observation(image_shapes=None):
    shapes = image_shapes or {"global_rgb": (480, 640, 3), "grasp_rgb": (480, 640, 3)}
    return {**dict.fromkeys(JOINT_NAMES, 0.0), "J5": 90.0, "gripper_pos": 0.0,
            **dict.fromkeys(("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"), 0.0),
            **{name: np.zeros(shape, np.uint8) for name, shape in shapes.items()}}


def _phone_target():
    return {**dict.fromkeys(("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"), 0.0),
            "ee_mode": "abs_j6yaw", "ee.j6_target": 0.015, "ee.gripper_pos": 0.0}


class Dataset:
    fps = 25
    num_episodes = 0
    image_writer = None

    def __init__(self, root, image_shapes=None):
        self.root = root
        self.features = joint_dataset_features(image_shapes)
        self.episode_buffer = {"size": 0, "action": []}
        self.frames = []
        self.save_calls = 0

    def add_frame(self, frame):
        self.frames.append(frame)
        self.episode_buffer["action"].append(frame["action"])
        self.episode_buffer["size"] += 1

    def save_episode(self):
        self.save_calls += 1
        self.num_episodes += 1
        self.clear_episode_buffer()

    def clear_episode_buffer(self):
        self.episode_buffer = {"size": 0, "action": []}


def _record_one(tmp_path, observer=None, action_processor=None, image_shapes=None):
    robot, dataset, entry = _robot(), Dataset(tmp_path / "data", image_shapes), _entry()
    robot.cameras = {name: SimpleNamespace(height=dataset.features[f"observation.images.{name}"]["shape"][0],
                                           width=dataset.features[f"observation.images.{name}"]["shape"][1])
                    for name in ("global_rgb", "grasp_rgb")}
    events = {"exit_early": False}
    def observation():
        robot.last_c0_sensor_timestamps = dict.fromkeys(("robot_state", "global_rgb", "grasp_rgb"), 1.0)
        return _observation(image_shapes)
    robot.get_observation = observation
    actual_send = robot.send_action
    def send(command):
        result = actual_send(command)
        events["exit_early"] = True
        return result
    robot.send_action = send
    record_loop(robot=robot, events=events, fps=25, dataset=dataset, teleop=FakeTeleop([_phone_target()]),
                teleop_action_processor=PhoneToJointAction(lambda pair: pair[0], robot),
                robot_action_processor=action_processor or entry.IdentityAction(),
                robot_observation_processor=entry.learning_observation, control_time_s=1,
                single_task=JOINT_TASK, display_data=False, observer=observer or JointFrameObserver(0))
    return robot, dataset


def test_real_record_loop_saves_accepted_joint_target_not_measured_state(tmp_path):
    observer = JointFrameObserver(0)
    robot, dataset = _record_one(tmp_path, observer)
    observer.validate_buffer(dataset)
    frame = dataset.frames[0]
    assert frame["action"].shape == frame["observation.state"].shape == (7,)
    assert frame["action"][0] == pytest.approx(math.degrees(0.01))
    assert frame["observation.state"][0] == 0
    assert "J5" in dataset.features["action"]["names"]
    assert not any("ee." in key for key in dataset.features["action"]["names"])
    sdk_q = robot.robot_interface.getMotionControl().servoJoint.call_args.args[0]
    assert sdk_q[4] == pytest.approx(math.radians(91))
    assert sdk_q[5] == pytest.approx(0.015)
    assert observer.frames[0]["joint_command_trace"]["sdk_target_rad"] == sdk_q


def test_record_refuses_label_command_mismatch_before_send(tmp_path):
    def change(pair):
        return {**pair[0], "J1": 1.9}
    with pytest.raises(ValueError, match="label differs"):
        _record_one(tmp_path, action_processor=change)


@pytest.mark.parametrize("changes", [{"J7": 90}, {"ee.x": 0}, {"J1": float("nan")}, {"J2": True}, {"gripper_pos": 50}])
def test_driver_rejects_bad_schema_before_sdk_calls(changes):
    robot = _robot()
    with pytest.raises(ValueError):
        robot.send_action({**dict.fromkeys(JOINT_FIELDS, 0.0), **changes})
    robot.robot_interface.getMotionControl().servoJoint.assert_not_called()


@pytest.mark.parametrize("returns", [[0], [2, 0], [-13, 0]])
def test_joint_capture_matches_legacy_phone_targets_and_servo_retries(returns):
    from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
    legacy, capture = _robot(), _robot()
    # Both a non-90 J5 and a >2 degree target must follow the legacy behavior.
    for robot in (legacy, capture):
        robot.robot_interface.getRobotAlgorithm().inverseKinematics.return_value = (
            [math.radians(5), 0, 0, 0, math.radians(85), 0], 0)
        robot.servo_max_queue_retry = 3
        robot.enable_servo_mode = Mock(return_value=True)
        robot.robot_interface.getMotionControl().servoJoint.side_effect = returns
    AuboI10Robot.send_action(legacy, _phone_target())
    target = capture.resolve_phone_joint_target(_phone_target(), _observation())
    assert target["J5"] == 85 and target["J1"] == 5
    assert capture.send_action(target) == target
    old_calls = legacy.robot_interface.getMotionControl().servoJoint.call_args_list
    new_calls = capture.robot_interface.getMotionControl().servoJoint.call_args_list
    assert len(old_calls) == len(new_calls) == len(returns)
    for old, new in zip(old_calls, new_calls):
        np.testing.assert_allclose(old.args[0], new.args[0], rtol=0, atol=1e-14)
        assert old.args[1:] == new.args[1:]
    assert legacy.enable_servo_mode.call_count == capture.enable_servo_mode.call_count
    assert capture.last_joint_command_trace["joint_target_deg"][4] == 85


def test_joint_capture_preserves_measured_j5_in_state():
    state = _entry().learning_observation({**_observation(), "J5": 92})
    assert state["J5"] == 92


def test_phone_ik_failure_never_sends():
    robot = _robot()
    robot.robot_interface.getRobotAlgorithm().inverseKinematics.return_value = ([0] * 6, -5)
    with pytest.raises(ValueError, match="inverse kinematics failed"):
        robot.resolve_phone_joint_target(_phone_target(), _observation())
    robot.robot_interface.getMotionControl().servoJoint.assert_not_called()


def test_servo_failure_never_leaves_success_trace():
    robot = _robot()
    robot.last_joint_command_trace = {"stale": True}
    robot.robot_interface.getMotionControl().servoJoint.return_value = -1
    with pytest.raises(RuntimeError):
        robot.send_action(dict.fromkeys(JOINT_FIELDS, 0.0))
    assert robot.last_joint_command_trace is None
    assert robot.last_gripper_command_trace is None


def test_stale_or_corrupt_command_trace_refused():
    robot = _robot()
    robot.last_c0_sensor_timestamps = dict.fromkeys(("robot_state", "global_rgb", "grasp_rgb"), 1.0)
    action = dict.fromkeys(JOINT_FIELDS, 0.0)
    observer = JointFrameObserver(0)
    kwargs = dict(robot=robot, dataset=None, action=action, robot_action_to_send=action, candidate_frame_index=0)
    observer.before_send(**kwargs)
    robot.send_action(action)
    robot.last_joint_command_trace["joint_target_deg"][4] = 89
    with pytest.raises(ValueError, match="SDK joint target"):
        observer.on_send_success(**kwargs, sent_action=action)


def _session(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    return JointCaptureSession(dataset_root=tmp_path / "data", evidence_root=evidence, episode_count=60,
        prepare_episode=lambda i: {"scene_id": f"scene-{i}", "placement_reference": "line B", "split": "train"},
        outcome_provider=lambda i: "single_success")


@pytest.mark.parametrize("image_shapes", [None, {"global_rgb": (1080, 1920, 3), "grasp_rgb": (480, 640, 3)}])
def test_save_finalize_and_readonly_audit(tmp_path, image_shapes):
    session = _session(tmp_path)
    robot, dataset = _record_one(tmp_path, session.episode_frame_observer(), image_shapes=image_shapes)
    original = copy.deepcopy(dataset.frames)
    for name in ("global_rgb", "grasp_rgb"):
        shape = dataset.features[f"observation.images.{name}"]["shape"]
        assert original[0][f"observation.images.{name}"].shape == shape
        assert robot.observation_features[name] == shape
    session.save_episode(robot=robot, dataset=dataset, episode_index=0)
    session.on_dataset_finalize_success(dataset=dataset, recorded_episode_count=1)
    root = dataset.root
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"features": joint_dataset_features(image_shapes), "fps": 25, "robot_type": "aubo_i10",
            "total_episodes": 1, "total_frames": 1}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/aubo_joint_contract.json").write_text(json.dumps(joint_contract_record()))
    if image_shapes:
        from lerobot.bamboo_sorting.joint_camera_config import load_joint_camera_configuration
        (root / "meta/camera_configuration.json").write_text(json.dumps(load_joint_camera_configuration("wide-global")))
    pq.write_table(pa.table({"task_index": [0], "__index_level_0__": [JOINT_TASK]}), root / "meta/tasks.parquet")
    data = {"observation.state": [original[0]["observation.state"].tolist()],
            "action": [original[0]["action"].tolist()], "episode_index": [0], "frame_index": [0], "task_index": [0]}
    parquet = root / "data/chunk-000/file-000.parquet"
    pq.write_table(pa.table(data), parquet)
    result = audit_joint_dataset(root, tmp_path / "evidence")
    assert result["audit_passed"] and not result["capture_complete"]
    assert not result["gripper_quality_passed"]
    assert result["supervised_candidate_episodes"] == []
    if image_shapes:
        assert result["image_shapes"]["global_rgb"] == [1080, 1920, 3]
        info["features"]["observation.images.global_rgb"]["shape"] = [480, 640, 3]
        (root / "meta/info.json").write_text(json.dumps(info))
        with pytest.raises(ValueError, match="invalid RGB video feature"):
            audit_joint_dataset(root, tmp_path / "evidence")
        info["features"]["observation.images.global_rgb"]["shape"] = [1080, 1920, 3]
        (root / "meta/info.json").write_text(json.dumps(info))
    data["action"][0][0] += 0.25
    pq.write_table(pa.table(data), parquet)
    with pytest.raises(ValueError, match="labels differ"):
        audit_joint_dataset(root, tmp_path / "evidence")


def test_rerecord_and_stop_never_save(tmp_path):
    session, dataset = _session(tmp_path), Dataset(tmp_path / "data")
    first = session.episode_frame_observer()
    session.on_episode_rerecord(dataset=dataset, episode_index=0)
    second = session.episode_frame_observer()
    assert second is not first and not second.frames
    session.on_episode_incomplete(dataset=dataset, episode_index=0, reason="stop")
    assert dataset.save_calls == session.saved_count == 0


def test_buffer_tampering_and_save_failure_do_not_claim_saved(tmp_path):
    session = _session(tmp_path)
    _, dataset = _record_one(tmp_path, session.episode_frame_observer())
    dataset.episode_buffer["action"][0][0] += 1
    with pytest.raises(ValueError, match="buffer differs"):
        session.save_episode(dataset=dataset, episode_index=0)
    assert dataset.save_calls == 0


def test_save_failure_does_not_claim_saved_or_retry(tmp_path):
    session = _session(tmp_path)
    _, dataset = _record_one(tmp_path, session.episode_frame_observer())
    dataset.save_episode = Mock(side_effect=OSError("disk failed"))
    with pytest.raises(OSError, match="disk failed"):
        session.save_episode(dataset=dataset, episode_index=0)
    assert session.saved_count == 0
    dataset.save_episode.assert_called_once()
    assert not (tmp_path / "evidence/episode-0000.json").exists()


def test_default_plan_is_hardware_free_and_refuses_overwrite(tmp_path, monkeypatch):
    entry = _entry()
    args = ["--dataset-root", str(tmp_path / "data"), "--evidence-root", str(tmp_path / "evidence"), "--split", "train"]
    import builtins
    real_import = builtins.__import__
    def guarded(name, *a, **k):
        if name.startswith(("pyaubo_sdk", "lerobot.robots", "lerobot.teleoperators", "lerobot.cameras")):
            raise AssertionError(f"hardware import: {name}")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert entry.main(args) == 0
    assert not (tmp_path / "data").exists() and not (tmp_path / "evidence").exists()
    (tmp_path / "data").mkdir()
    with pytest.raises(SystemExit):
        entry.parse_args(args)


def test_joint_policy_excludes_tcp_targets_and_preserves_joint_predictions():
    cfg = make_joint_smolvla_config()
    raw = torch.tensor([[[1, 2, 3, 4, 5, 6, 97.0]]])
    policy = Mock(config=cfg)
    policy.predict_action_chunk.return_value = raw
    adapter = SmolVLAJointOfflineAdapter(policy, lambda x: x, lambda x: x, contract=joint_contract_record())
    frame = {"task": JOINT_TASK, "observation.state": np.zeros(7, np.float32), "action": np.ones(7), "ee.x": 42,
             **{f"observation.images.{k}": np.zeros((480, 640, 3), np.uint8) for k in ("global_rgb", "grasp_rgb")}}
    decoded = adapter(frame)
    assert decoded.tolist() == [[1, 2, 3, 4, 5, 6, 100]]
    assert adapter.raw_actions[0, -1] == 97
    actual = policy.predict_action_chunk.call_args.args[0]
    assert "action" not in actual and "ee.x" not in actual
    assert actual["observation.state"].shape == (7,)
    with pytest.raises(ValueError):
        SmolVLAJointOfflineAdapter(policy, lambda x: x, lambda x: x, contract={})


def test_legacy_dataset_contract_is_rejected():
    features = joint_dataset_features()
    features["observation.state"]["shape"] = (13,)
    with pytest.raises(ValueError, match="7D"):
        require_joint_dataset(features, joint_contract_record(), 25)


def test_fixed_j5_data_and_checkpoint_are_not_reinterpreted_as_seven_dimensions():
    features = joint_dataset_features()
    features["action"].update(shape=(6,), names=["J1", "J2", "J3", "J4", "J6", "gripper_pos"])
    with pytest.raises(ValueError, match="7D"):
        require_joint_dataset(features, joint_contract_record(), 25)
    old_contract = {**joint_contract_record(), "schema_version": "AuboI10JointFixedJ5V1",
                    "fixed_joint_targets_deg": {"J5": 90.0}}
    with pytest.raises(ValueError, match="AuboI10JointLegacyTeleopV2"):
        require_joint_dataset(joint_dataset_features(), old_contract, 25)
    with pytest.raises(ValueError, match="AuboI10JointLegacyTeleopV2"):
        SmolVLAJointOfflineAdapter(Mock(config=make_joint_smolvla_config()), None, None, contract=old_contract)


@pytest.mark.parametrize("failure", [None, "capture", "finalize"])
def test_entry_keyboard_ownership_and_stop_before_finalize(tmp_path, monkeypatch, failure):
    """Run entry callbacks with fake devices; prompts must own stdin exclusively."""
    import sys
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.aubo_i10 import joint_capture
    from lerobot.teleoperators.phone import teleop_phone
    from lerobot.scripts import lerobot_record
    from lerobot.utils import control_utils

    entry = _entry()
    calls, listeners = [], []
    fake = _robot()
    fake.robot_interface.getMotionControl().moveJoint.return_value = 0
    fake.disable_servo_mode = lambda: calls.append("disable") or True
    fake.suction_release = lambda: True
    monkeypatch.setattr(joint_capture.AuboI10JointCaptureRobot, "__init__",
                        lambda self, config: self.__dict__.update(fake.__dict__))
    monkeypatch.setattr(joint_capture.AuboI10JointCaptureRobot, "connect", lambda self: None)
    # Only explicit entry cleanup belongs in this trace, not GC of earlier fakes.
    monkeypatch.setattr(joint_capture.AuboI10JointCaptureRobot, "__del__", lambda self: None)
    monkeypatch.setattr(joint_capture.AuboI10JointCaptureRobot, "disconnect",
                        lambda self: calls.append("robot_disconnect"))
    monkeypatch.setattr(teleop_phone, "Phone", lambda config: SimpleNamespace(
        connect=lambda: None, is_connected=True,
        disconnect=lambda: calls.append("phone_disconnect")))
    monkeypatch.setitem(sys.modules, "record_c0_batch", SimpleNamespace(
        read_fresh_c0_observation=lambda **kwargs: None,
        _load_record_helpers=lambda: SimpleNamespace(
            return_to_start=lambda robot: True)))

    def create(**kwargs):
        assert kwargs["features"]["observation.images.global_rgb"]["shape"] == (480, 640, 3)
        assert kwargs["features"]["observation.images.grasp_rgb"]["shape"] == (480, 640, 3)
        (kwargs["root"] / "meta").mkdir(parents=True)
        return Dataset(kwargs["root"])
    monkeypatch.setattr(LeRobotDataset, "create", create)

    def start_listener(events=None):
        events = {} if events is None else events
        events.update(exit_early=False, return_to_start=False, stop_recording=False, rerecord_episode=False)
        listener = SimpleNamespace(active=True, events=events)
        listener.stop = lambda: setattr(listener, "active", False)
        listeners.append(listener)
        return listener, events
    monkeypatch.setattr(control_utils, "init_keyboard_listener", start_listener)
    answers = iter(["", "  ", "train-r-scene", "", "ruler B", "invalid", "1",
                    "train-r-scene-2", "ruler C", "2"])
    def prompt(message):
        assert not any(listener.active for listener in listeners), "hotkeys compete with input()"
        return next(answers)
    monkeypatch.setattr("builtins.input", prompt)

    def press_keys(delay):
        active = [listener for listener in listeners if listener.active]
        assert len(active) == 1
        active[0].events.update(return_to_start=True, exit_early=True)
    monkeypatch.setattr(entry.time, "sleep", press_keys)

    def record(**kwargs):
        lifecycle = kwargs["lifecycle"]
        for index in range(2):
            metadata = lifecycle.prepare_episode(index)
            assert metadata["scene_id"] == ["train-r-scene", "train-r-scene-2"][index]
            assert metadata["placement_reference"] == ["ruler B", "ruler C"][index]
            assert sum(listener.active for listener in listeners) == 1
            if failure == "capture":
                calls.append("capture_failure")
                raise RuntimeError("capture failed")
            assert lifecycle.outcome_provider(index) == ["single_success", "empty"][index]
        lifecycle.saved_count = 2
    monkeypatch.setattr(lerobot_record, "record_episode_sessions", record)

    def finalize(*args, **kwargs):
        assert calls[-1] == "disable", "servo must be disabled before video finalization"
        if failure == "capture":
            assert calls.index("capture_failure") < len(calls) - 1
        calls.append("finalize")
        if failure == "finalize":
            raise RuntimeError("finalize failed")
    monkeypatch.setattr(lerobot_record, "finalize_recorded_dataset", finalize)
    args = ["--dataset-root", str(tmp_path / "data"), "--evidence-root", str(tmp_path / "evidence"),
            "--num-episodes", "2", "--split", "train", "--record"]
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            entry.main(args)
    else:
        assert entry.main(args) == 0
    assert "finalize" in calls
    assert calls.index("finalize") < calls.index("phone_disconnect") < calls.index("robot_disconnect")
    assert not any(listener.active for listener in listeners)
    saved_camera = json.loads((tmp_path / "data/meta/camera_configuration.json").read_text())
    saved_plan = json.loads((tmp_path / "evidence/plan.json").read_text())
    assert saved_camera["camera_set"] == "original-global"
    assert saved_camera["camera_mapping"] == saved_plan["camera_mapping"]
    assert saved_camera["camera_mapping"]["global_rgb"]["width"] == 640
