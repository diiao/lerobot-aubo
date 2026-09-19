from __future__ import annotations

import importlib.util
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.utils.rotation import Rotation


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "phone_to_auboi10"
    / "evaluate_split.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_split_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate_split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_split)


def test_dry_run_flag_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("DRY_RUN", raising=False)

    assert evaluate_split.env_flag("DRY_RUN", default=True) is True


def test_pure_act_defaults_disable_task_specific_chunk_lookahead() -> None:
    assert evaluate_split.USE_GRIPPER_CHUNK_LOOKAHEAD is False
    assert evaluate_split.USE_MOTION_CHUNK_LOOKAHEAD is False


def test_extended137_j6_envelope_covers_demonstrated_max() -> None:
    assert evaluate_split.J6_TARGET_MAX_RAD == -1.39
    evaluate_split.assert_task_action_envelope(
        (-1.490041, 0.1, -0.7, 0.15, 0.0, 0.0, 0.0, 0.0)
    )


def test_real_execution_requires_independent_authorization(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_split, "DRY_RUN", False)
    monkeypatch.setattr(evaluate_split, "POLICY_EXECUTION_AUTHORIZED", False)

    with pytest.raises(PermissionError, match="POLICY_EXECUTION_AUTHORIZED"):
        evaluate_split.validate_runtime_settings()


@pytest.mark.parametrize(
    ("raw", "previous", "expected", "next_state"),
    [
        (61.0, False, 100.0, True),
        (19.0, True, 0.0, False),
        (40.0, False, 0.0, False),
        (40.0, True, 100.0, True),
    ],
)
def test_gripper_hysteresis_is_explicit(raw, previous, expected, next_state) -> None:
    command, state = evaluate_split.discretize_gripper_command(
        raw, suction_on=previous
    )

    assert command == expected
    assert state is next_state


def test_gripper_chunk_lookahead_closes_when_later_step_commits() -> None:
    action = {
        "ee.j6_target": -3.0,
        "ee.x": 0.54,
        "ee.y": -0.38,
        "ee.z": 0.09,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": -3.0,
    }
    chunk = [-3.4] * 20 + [12.0, 40.0, 51.0, 55.0, 58.0]
    held, mode = evaluate_split.apply_gripper_chunk_lookahead(
        action, chunk, close_score=50.0, open_score=8.0
    )
    command, state = evaluate_split.discretize_gripper_command(
        held["ee.gripper_pos"], suction_on=False
    )

    assert mode == "close"
    assert command == 100.0
    assert state is True
    assert action["ee.gripper_pos"] == -3.0


def test_gripper_chunk_lookahead_does_not_override_open_chunk() -> None:
    action = {"ee.gripper_pos": -3.0}
    held, mode = evaluate_split.apply_gripper_chunk_lookahead(
        action, [-4.0, -3.0, -2.0], close_score=50.0, open_score=8.0
    )

    assert mode == "pass"
    assert held["ee.gripper_pos"] == -3.0


def test_live07_pick_chunk_closes_at_default_score() -> None:
    action = {"ee.gripper_pos": -4.3}
    chunk = [-4.0] * 20 + [10.0, 18.1, 23.9, 26.7, 34.6]
    held, mode = evaluate_split.apply_gripper_chunk_lookahead(
        action,
        chunk,
        close_score=evaluate_split.GRIPPER_CHUNK_CLOSE_SCORE,
        open_score=evaluate_split.GRIPPER_CHUNK_OPEN_SCORE,
    )

    assert evaluate_split.GRIPPER_CHUNK_CLOSE_SCORE == 20.0
    assert mode == "close"
    assert held["ee.gripper_pos"] == 100.0


def test_live08_chunk_flicker_holds_closed_instead_of_chatter() -> None:
    action = {"ee.gripper_pos": -4.54}
    held, mode = evaluate_split.apply_gripper_chunk_lookahead(
        action,
        [-4.54] * 24 + [18.3],
        close_score=20.0,
        open_score=8.0,
    )
    command, state = evaluate_split.discretize_gripper_command(
        held["ee.gripper_pos"], suction_on=True
    )

    assert mode == "hold"
    assert command == 100.0
    assert state is True


def test_motion_chunk_lookahead_follows_planned_descent_when_suction_off() -> None:
    action = {
        "ee.j6_target": -2.85,
        "ee.x": 0.58,
        "ee.y": -0.53,
        "ee.z": 0.22,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": -3.0,
    }
    chunk = [[-2.85, 0.58, -0.53, 0.22, 0.0, 0.0, 0.0, -3.0]] * 20
    chunk.append([-2.55, 0.54, -0.37, 0.09, 0.0, 0.0, 0.0, -3.0])
    held, overrode = evaluate_split.apply_motion_chunk_lookahead(
        action, chunk, min_z_drop_m=0.02, suction_on=False
    )

    assert overrode is True
    assert held["ee.z"] == 0.09
    assert held["ee.x"] == 0.54
    assert held["ee.j6_target"] == -2.55


def test_motion_chunk_lookahead_skips_when_vacuum_is_on() -> None:
    action = {"ee.j6_target": -2.0, "ee.x": 0.5, "ee.y": -0.4, "ee.z": 0.22}
    chunk = [[-2.0, 0.5, -0.4, 0.09, 0.0, 0.0, 0.0, 100.0]]
    held, overrode = evaluate_split.apply_motion_chunk_lookahead(
        action, chunk, min_z_drop_m=0.02, suction_on=True
    )

    assert overrode is False
    assert held["ee.z"] == 0.22


def test_live08_place_release_when_chunk_drops_below_open_score() -> None:
    action = {"ee.gripper_pos": -1.04}
    held, mode = evaluate_split.apply_gripper_chunk_lookahead(
        action,
        [-1.04] * 25,
        close_score=20.0,
        open_score=8.0,
    )
    command, state = evaluate_split.discretize_gripper_command(
        held["ee.gripper_pos"], suction_on=True
    )

    assert mode == "pass"
    assert command == 0.0
    assert state is False


def test_action_vector_uses_action_schema_order() -> None:
    action = {
        "ee.gripper_pos": 100.0,
        "ee.wz": 0.6,
        "ee.wy": 0.5,
        "ee.wx": 0.4,
        "ee.z": 0.3,
        "ee.y": -0.2,
        "ee.x": 0.1,
        "ee.j6_target": -3.0,
    }

    assert evaluate_split.action_vector(action) == (
        -3.0,
        0.1,
        -0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        100.0,
    )


def test_task_j6_envelope_rejects_slow_drift_outside_demonstrations() -> None:
    valid = (-3.0, 0.1, -0.7, 0.15, 0.0, 0.0, 0.0, 0.0)
    evaluate_split.assert_task_action_envelope(valid)

    invalid = (-4.48, 0.1, -0.7, 0.15, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="超出任务安全范围"):
        evaluate_split.assert_task_action_envelope(invalid)


def test_ik_checker_allows_reach_joint_steps_but_rejects_unreachable() -> None:
    class FakeAlgorithm:
        def __init__(self, errno: int, solution: list[float]):
            self.errno = errno
            self.solution = solution

        def inverseKinematics(self, _seed, _pose):
            return self.solution, self.errno

    class FakeRobot:
        def __init__(self, errno: int, solution: list[float]):
            algorithm = FakeAlgorithm(errno, solution)
            state = SimpleNamespace(getJointPositions=lambda: [0.0] * 6)
            self.robot_interface = SimpleNamespace(
                getRobotState=lambda: state,
                getRobotAlgorithm=lambda: algorithm,
            )

    action = (0.0, 0.32, -0.56, 0.19, 0.0, 0.0, 0.0, 0.0)
    reachable = evaluate_split.make_ik_checker(
        FakeRobot(0, [0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    unreachable = evaluate_split.make_ik_checker(
        FakeRobot(1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    flipped = evaluate_split.make_ik_checker(
        FakeRobot(0, [1.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    assert reachable(action) is True
    assert reachable.last_trace["errno"] == 0
    assert reachable.last_trace["passed"] is True
    assert unreachable(action) is False
    assert unreachable.last_trace["errno"] == 1
    assert flipped(action) is False
    assert flipped.last_trace["failed_joints"] == ["J1"]
    assert flipped.last_trace["max_abs_delta_rad"] == pytest.approx(1.2)


def test_start_pose_gate_accepts_small_error_and_rejects_large_error() -> None:
    within_tolerance = list(evaluate_split.START_JOINT_DEG)
    within_tolerance[2] += evaluate_split.START_JOINT_TOLERANCE_DEG
    evaluate_split.assert_start_pose(within_tolerance)

    outside_tolerance = list(evaluate_split.START_JOINT_DEG)
    outside_tolerance[2] += evaluate_split.START_JOINT_TOLERANCE_DEG + 0.01
    with pytest.raises(RuntimeError, match="起始位检查失败"):
        evaluate_split.assert_start_pose(outside_tolerance)


def test_split_inference_age_defaults_cover_gpu_roundtrip() -> None:
    assert evaluate_split.MAX_OBSERVATION_AGE_MS == 2000.0
    assert evaluate_split.MAX_ACTION_CHUNK_AGE_MS == 2000.0


def test_first_tcp_jump_is_clipped_instead_of_rejected() -> None:
    action = {
        "ee.j6_target": -3.0,
        "ee.x": 0.20,
        "ee.y": -0.70,
        "ee.z": 0.15,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 0.0,
    }

    clipped = evaluate_split.clip_absolute_action_to_measured_limits(
        action,
        previous_tcp_m=(0.10, -0.70, 0.15),
        previous_j6_rad=-3.0,
        max_ee_step_m=0.008,
        max_j6_step_rad=0.03,
    )

    dist = math.dist(
        (clipped["ee.x"], clipped["ee.y"], clipped["ee.z"]),
        (0.10, -0.70, 0.15),
    )
    assert dist == pytest.approx(0.008, abs=1e-6)
    assert dist <= 0.008
    assert clipped["ee.y"] == pytest.approx(-0.70)
    assert clipped["ee.j6_target"] == pytest.approx(-3.0)


def test_clipped_stale_split_inference_step_passes_thin_safety_gate() -> None:
    from lerobot.bamboo_sorting.thin_safety_gate import ThinSafetyGate, ThinSafetyGateLimits

    previous = (0.10, -0.70, 0.15)
    action = {
        "ee.j6_target": -3.20,
        "ee.x": 0.40,
        "ee.y": -0.70,
        "ee.z": 0.15,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 0.0,
    }
    clipped = evaluate_split.clip_absolute_action_to_measured_limits(
        action,
        previous_tcp_m=previous,
        previous_j6_rad=-3.0,
        max_ee_step_m=0.008,
        max_j6_step_rad=0.03,
    )
    gate = ThinSafetyGate(
        ThinSafetyGateLimits(
            workspace_min_m=evaluate_split.WORKSPACE_MIN_M,
            workspace_max_m=evaluate_split.WORKSPACE_MAX_M,
            max_ee_step_m=0.008,
            max_ee_speed_mps=0.008 * 25,
            max_j6_step_rad=0.03,
            max_observation_age_ms=2000,
            max_chunk_age_ms=2000,
            control_fps=25,
            require_previous_tcp=True,
            require_ik=False,
        )
    )

    gated = gate.evaluate(
        [evaluate_split.action_vector(clipped)],
        now_monotonic_s=2.0,
        observation_sync_timestamp_s=1.5,
        chunk_created_monotonic_s=1.99,
        previous_tcp_m=previous,
        previous_j6_rad=-3.0,
    )

    assert gated.passed
    assert clipped["ee.j6_target"] == pytest.approx(-3.03)


def test_observation_j6_is_converted_from_degrees_to_action_radians() -> None:
    tcp_m, j6_rad = evaluate_split.measured_action_anchors(
        {"ee.x": 0.1, "ee.y": -0.7, "ee.z": 0.15, "J6": -185.32}
    )

    assert tcp_m == (0.1, -0.7, 0.15)
    assert j6_rad == pytest.approx(math.radians(-185.32))


def test_large_orientation_jump_is_geodesic_clipped_to_measured_pose() -> None:
    action = {
        "ee.j6_target": -3.0,
        "ee.x": 0.10,
        "ee.y": -0.70,
        "ee.z": 0.15,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": math.radians(7.0),
        "ee.gripper_pos": 0.0,
    }
    previous = (0.0, 0.0, 0.0)
    clipped = evaluate_split.clip_absolute_action_to_measured_limits(
        action,
        previous_tcp_m=(0.10, -0.70, 0.15),
        previous_j6_rad=-3.0,
        max_ee_step_m=0.008,
        max_j6_step_rad=0.03,
        previous_rotvec=previous,
        max_ee_rot_step_rad=0.03,
    )
    sent = (clipped["ee.wx"], clipped["ee.wy"], clipped["ee.wz"])
    geodesic = evaluate_split.rotation_geodesic_angle_rad(previous, sent)

    assert geodesic == pytest.approx(0.03, abs=1e-5)
    assert geodesic < math.radians(7.0)
    assert clipped["ee.x"] == pytest.approx(0.10)


def test_equivalent_rotvec_branch_is_aligned_to_measured_representation() -> None:
    previous = (0.0, 0.0, math.radians(197.0))
    principal = Rotation.from_rotvec(np.array(previous)).as_rotvec()
    action = {
        "ee.j6_target": -3.0,
        "ee.x": 0.10,
        "ee.y": -0.70,
        "ee.z": 0.15,
        "ee.wx": float(principal[0]),
        "ee.wy": float(principal[1]),
        "ee.wz": float(principal[2]),
        "ee.gripper_pos": 0.0,
    }
    clipped = evaluate_split.clip_absolute_action_to_measured_limits(
        action,
        previous_tcp_m=(0.10, -0.70, 0.15),
        previous_j6_rad=-3.0,
        max_ee_step_m=0.008,
        max_j6_step_rad=0.03,
        previous_rotvec=previous,
        max_ee_rot_step_rad=0.03,
    )
    sent = np.array([clipped["ee.wx"], clipped["ee.wy"], clipped["ee.wz"]])

    assert evaluate_split.rotation_geodesic_angle_rad(previous, sent) == pytest.approx(
        0.0, abs=1e-6
    )
    np.testing.assert_allclose(sent, previous, atol=1e-6)
    assert abs(float(np.linalg.norm(sent)) - abs(previous[2])) == pytest.approx(0.0, abs=1e-6)


def test_live02_terminal_orientation_drift_is_clipped_before_ik() -> None:
    measured = (-2.9182, 0.0127, -1.3068)
    action = {
        "ee.j6_target": -2.033267021179199,
        "ee.x": 0.44116705656051636,
        "ee.y": -0.4426851272583008,
        "ee.z": 0.19424687325954437,
        "ee.wx": -2.819065809249878,
        "ee.wy": 0.012301926501095295,
        "ee.wz": -1.2201869487762451,
        "ee.gripper_pos": 0.0,
    }
    raw_geodesic = evaluate_split.rotation_geodesic_angle_rad(
        measured, (action["ee.wx"], action["ee.wy"], action["ee.wz"])
    )
    clipped = evaluate_split.clip_absolute_action_to_measured_limits(
        action,
        previous_tcp_m=(0.4422, -0.4422, 0.1931),
        previous_j6_rad=-2.0209,
        max_ee_step_m=0.008,
        max_j6_step_rad=0.03,
        previous_rotvec=measured,
        max_ee_rot_step_rad=evaluate_split.MAX_EE_ROT_STEP_RAD,
    )
    sent = (clipped["ee.wx"], clipped["ee.wy"], clipped["ee.wz"])
    sent_geodesic = evaluate_split.rotation_geodesic_angle_rad(measured, sent)

    assert raw_geodesic == pytest.approx(math.radians(7.34), abs=0.05)
    assert sent_geodesic <= evaluate_split.MAX_EE_ROT_STEP_RAD
    assert sent_geodesic == pytest.approx(evaluate_split.MAX_EE_ROT_STEP_RAD, abs=1e-5)


def test_locked_ee_pose_ignores_model_orientation_drift() -> None:
    locked = (-3.11, 0.012, -1.86)
    action = {
        "ee.j6_target": -2.0,
        "ee.x": 0.45,
        "ee.y": -0.44,
        "ee.z": 0.19,
        "ee.wx": -2.82,
        "ee.wy": 0.012,
        "ee.wz": -1.22,
        "ee.gripper_pos": 0.0,
    }
    held = evaluate_split.apply_locked_ee_pose(action, locked)

    assert (held["ee.wx"], held["ee.wy"], held["ee.wz"]) == locked
    assert held["ee.x"] == action["ee.x"]
    assert held["ee.j6_target"] == action["ee.j6_target"]


class _BytesSocket:
    def __init__(self, payload: bytes):
        self.payload = bytearray(payload)

    def recv(self, count: int) -> bytes:
        chunk = bytes(self.payload[:count])
        del self.payload[:count]
        return chunk


def test_oversized_server_response_is_rejected_before_payload_read() -> None:
    sock = _BytesSocket(struct.pack(">I", 4097))

    with pytest.raises(RuntimeError, match="响应长度非法"):
        evaluate_split.recv_msg(sock, max_bytes=4096)


def test_local_dataset_resolution_checks_metadata(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    repo_id, root = evaluate_split.resolve_local_dataset_path(
        "./dataset", must_exist=True
    )

    assert repo_id == "dataset"
    assert root == dataset.resolve()


def test_jsonl_trace_writer_refuses_overwrite_and_writes_valid_json(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = evaluate_split.JsonlTraceWriter(path)
    writer.write({"schema_version": "test_v1", "value": 1.25})
    writer.close()

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": "test_v1",
        "value": 1.25,
    }
    with pytest.raises(FileExistsError):
        evaluate_split.JsonlTraceWriter(path)


def test_attempt_trace_records_distinguish_rerecorded_attempts() -> None:
    first_end = evaluate_split.attempt_trace_record(
        record_type="attempt_end",
        episode_index=3,
        attempt_index=0,
        disposition="rerecorded",
    )
    second_start = evaluate_split.attempt_trace_record(
        record_type="attempt_start",
        episode_index=3,
        attempt_index=1,
    )

    assert first_end["attempt_id"] == "episode-0003-attempt-000"
    assert second_start["attempt_id"] == "episode-0003-attempt-001"
    assert first_end["disposition"] == "rerecorded"
    assert second_start["disposition"] is None


def test_attempt_trace_rejects_invalid_lifecycle_values() -> None:
    with pytest.raises(ValueError, match="未知 attempt disposition"):
        evaluate_split.attempt_trace_record(
            record_type="attempt_end",
            episode_index=0,
            attempt_index=0,
            disposition="completed",
        )


def test_dry_run_loop_never_calls_robot_send_action(monkeypatch) -> None:
    action_values = [-3.0, 0.1, -0.7, 0.15, 0.0, 0.0, 0.0, 0.0]

    class Robot:
        robot_type = "aubo_i10"
        send_count = 0

        def get_observation(self):
            return {
                "J6": math.degrees(-3.0),
                "ee.x": 0.1,
                "ee.y": -0.7,
                "ee.z": 0.15,
                "ee.wx": 0.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
                "gripper_pos": 0.0,
            }

        def send_action(self, _action):
            self.send_count += 1

    class Dataset:
        features = {}
        frame_count = 0

        def add_frame(self, _frame):
            self.frame_count += 1

    robot = Robot()
    dataset = Dataset()
    monkeypatch.setattr(evaluate_split, "build_dataset_frame", lambda _f, value, prefix: value)
    monkeypatch.setattr(evaluate_split, "encode_obs", lambda *_args: {})
    monkeypatch.setattr(evaluate_split, "send_msg", lambda *_args: None)
    monkeypatch.setattr(
        evaluate_split,
        "recv_msg",
        lambda _sock: {"actions": action_values},
    )
    monkeypatch.setattr(
        evaluate_split,
        "make_robot_action",
        lambda _tensor, _features: dict(
            zip(evaluate_split.ACTION_FIELD_NAMES, action_values, strict=True)
        ),
    )
    monkeypatch.setattr(evaluate_split, "make_ik_checker", lambda _robot: lambda _a: True)

    evaluate_split.run_episode(
        robot=robot,
        events={"exit_early": False},
        fps=25,
        sock=object(),
        dataset=dataset,
        control_time_s=0.0001,
        single_task="抓取竹条",
        ee_mode_processor=lambda pair: pair[0],
        robot_observation_processor=lambda observation: observation,
        safety_gate=SimpleNamespace(
            evaluate=lambda *_args, **_kwargs: SimpleNamespace(passed=True, reasons=())
        ),
        execute_actions=False,
    )

    assert robot.send_count == 0
    assert dataset.frame_count == 1


def test_send_failure_does_not_commit_hysteresis_state_and_stops_servo(monkeypatch) -> None:
    action_values = [-3.0, 0.1, -0.7, 0.15, 0.0, 0.0, 0.0, 100.0]

    class Robot:
        robot_type = "aubo_i10"
        suction_on_pin = 2
        suction_off_pin = 3
        is_suction_on = False
        disable_count = 0
        last_gripper_command_trace = {
            "do_api_success": False,
            "commanded_state_before": False,
            "commanded_state_after": False,
            "error": "second DO failed",
        }

        def get_observation(self):
            return {
                "J6": math.degrees(-3.0),
                "ee.x": 0.1,
                "ee.y": -0.7,
                "ee.z": 0.15,
                "ee.wx": 0.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
                "gripper_pos": 0.0,
            }

        def send_action(self, _action):
            assert self.is_suction_on is False
            raise RuntimeError("second DO failed")

        def disable_servo_mode(self):
            self.disable_count += 1

    class Dataset:
        features = {}

        def add_frame(self, _frame):
            raise AssertionError("failed action must not be recorded as executed")

    class TraceWriter:
        def __init__(self):
            self.records = []

        def write(self, record):
            self.records.append(record)

    robot = Robot()
    trace_writer = TraceWriter()
    monkeypatch.setattr(evaluate_split, "build_dataset_frame", lambda _f, value, prefix: value)
    monkeypatch.setattr(evaluate_split, "encode_obs", lambda *_args: {})
    monkeypatch.setattr(evaluate_split, "send_msg", lambda *_args: None)
    monkeypatch.setattr(
        evaluate_split,
        "recv_msg",
        lambda _sock: {
            "actions": action_values,
            "trace": {
                "selected_normalized_action": [0.0] * 7 + [1.2],
                "selected_denormalized_action": action_values,
            },
        },
    )
    monkeypatch.setattr(
        evaluate_split,
        "make_robot_action",
        lambda _tensor, _features: dict(
            zip(evaluate_split.ACTION_FIELD_NAMES, action_values, strict=True)
        ),
    )
    monkeypatch.setattr(evaluate_split, "make_ik_checker", lambda _robot: lambda _a: True)

    with pytest.raises(RuntimeError, match="已停止本轮并关闭伺服"):
        evaluate_split.run_episode(
            robot=robot,
            events={"exit_early": False},
            fps=25,
            sock=object(),
            dataset=Dataset(),
            control_time_s=0.0001,
            single_task="抓取竹条",
            ee_mode_processor=lambda pair: pair[0],
            robot_observation_processor=lambda observation: observation,
            safety_gate=SimpleNamespace(
                evaluate=lambda *_args, **_kwargs: SimpleNamespace(passed=True, reasons=())
            ),
            execute_actions=True,
            trace_writer=trace_writer,
            episode_index=3,
            attempt_index=2,
        )

    assert robot.is_suction_on is False
    assert robot.disable_count == 1
    assert len(trace_writer.records) == 1
    record = trace_writer.records[0]
    assert record["record_type"] == "step"
    assert record["attempt_index"] == 2
    assert record["attempt_id"] == "episode-0003-attempt-002"
    assert record["gripper"]["hysteresis_state_before"] is False
    assert record["gripper"]["hysteresis_state_candidate"] is True
    assert record["actuator"]["commanded_state_after"] is False
    assert record["actuator"]["success"] is False
    assert record["outcome"] == "actuator_error"
    assert record["pose_clip"]["max_ee_rot_step_rad"] == evaluate_split.MAX_EE_ROT_STEP_RAD
    assert record["pose_clip"]["sent_rotvec"] == [0.0, 0.0, 0.0]
    assert record["pose_clip"]["hold_locked_ee_pose"] is True
    assert record["pose_clip"]["locked_rotvec"] == [0.0, 0.0, 0.0]
