from __future__ import annotations

from unittest.mock import Mock

import pytest

from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot


class FakeIO:
    def __init__(
        self,
        *,
        suction_on: bool,
        fail_at: int | None = None,
        exception_at: int | None = None,
    ):
        self.outputs = {2: suction_on, 3: not suction_on}
        self.fail_at = fail_at
        self.exception_at = exception_at
        self.calls: list[tuple[int, bool]] = []

    def setStandardDigitalOutput(self, pin: int, value: bool) -> int:
        self.calls.append((pin, value))
        if self.exception_at == len(self.calls):
            raise RuntimeError(f"sdk write failed at call {len(self.calls)}")
        if self.fail_at == len(self.calls):
            return -1
        self.outputs[pin] = value
        return 0

    def getStandardDigitalOutput(self, pin: int) -> bool:
        return self.outputs[pin]


def make_robot(*, suction_on: bool = False, io_control=None) -> AuboI10Robot:
    robot = object.__new__(AuboI10Robot)
    robot.suction_on_pin = 2
    robot.suction_off_pin = 3
    robot.is_suction_on = suction_on
    robot.io_control = io_control
    robot.last_gripper_command_trace = None
    robot.suction_do_readback_attempts = 1
    robot.suction_do_readback_interval_s = 0.0
    return robot


def test_close_success_updates_state_only_after_both_writes_and_readback() -> None:
    io = FakeIO(suction_on=False)
    robot = make_robot(io_control=io)

    assert robot.suction_activate() is True
    assert io.calls == [(3, False), (2, True)]
    assert robot.is_suction_on is True
    assert robot.last_gripper_command_trace["do_api_success"] is True
    assert robot.last_gripper_command_trace["do_readback"] == {"2": True, "3": False}
    assert robot.last_gripper_command_trace["controller_output_state_known"] is True
    assert robot.last_gripper_command_trace["controller_output_matches_requested"] is True
    assert robot.last_gripper_command_trace["partial_write_possible"] is False


def test_release_success_updates_state_only_after_both_writes_and_readback() -> None:
    io = FakeIO(suction_on=True)
    robot = make_robot(suction_on=True, io_control=io)

    assert robot.suction_release() is True
    assert io.calls == [(2, False), (3, True)]
    assert robot.is_suction_on is False
    assert robot.last_gripper_command_trace["do_readback"] == {"2": False, "3": True}


@pytest.mark.parametrize("fail_at", [1, 2])
def test_close_write_failure_keeps_previous_commanded_state(fail_at: int) -> None:
    io = FakeIO(suction_on=False, fail_at=fail_at)
    robot = make_robot(io_control=io)

    assert robot.suction_activate() is False
    assert robot.is_suction_on is False
    assert robot.last_gripper_command_trace["do_api_success"] is False
    assert robot.last_gripper_command_trace["commanded_state_after"] is False
    assert f"returned -1" in robot.last_gripper_command_trace["error"]
    assert robot.last_gripper_command_trace["do_write_attempt_count"] == fail_at
    assert robot.last_gripper_command_trace["partial_write_possible"] is True
    assert robot.last_gripper_command_trace["controller_output_state_known"] is True
    assert robot.last_gripper_command_trace["do_readback_after_failure"] == {
        "2": io.outputs[2],
        "3": io.outputs[3],
    }


def test_io_control_none_fails_without_state_change() -> None:
    robot = make_robot(io_control=None)

    assert robot.suction_activate() is False
    assert robot.is_suction_on is False
    assert robot.last_gripper_command_trace["error"] == "io_control is None"
    assert robot.last_gripper_command_trace["partial_write_possible"] is False
    assert robot.last_gripper_command_trace["controller_output_state_known"] is False


class DelayedReadIO(FakeIO):
    def __init__(self, *, suction_on: bool, stale_attempts: int):
        super().__init__(suction_on=suction_on)
        self.stale = dict(self.outputs)
        self.stale_attempts = stale_attempts
        self.read_attempts = 0

    def getStandardDigitalOutput(self, pin: int) -> bool:
        if pin == 2:
            self.read_attempts += 1
        if self.read_attempts <= self.stale_attempts:
            return self.stale[pin]
        return self.outputs[pin]


def test_controller_readback_mismatch_is_not_reported_as_success() -> None:
    io = FakeIO(suction_on=False)
    io.getStandardDigitalOutput = Mock(return_value=False)
    robot = make_robot(io_control=io)

    assert robot.suction_activate() is False
    assert robot.is_suction_on is False
    assert robot.last_gripper_command_trace["do_api_success"] is False
    assert "readback mismatch" in robot.last_gripper_command_trace["error"]


def test_stale_do_readback_succeeds_after_retry(monkeypatch) -> None:
    io = DelayedReadIO(suction_on=False, stale_attempts=2)
    robot = make_robot(io_control=io)
    robot.suction_do_readback_attempts = 4
    robot.suction_do_readback_interval_s = 0.025
    monkeypatch.setattr("lerobot.robots.aubo_i10.aubo_i10.time.sleep", lambda _s: None)

    assert robot.suction_activate() is True
    assert robot.is_suction_on is True
    assert robot.last_gripper_command_trace["do_readback"] == {"2": True, "3": False}
    assert robot.last_gripper_command_trace["do_readback_attempts"] == 3


def test_failed_write_and_failed_readback_marks_controller_state_unknown() -> None:
    io = FakeIO(suction_on=False, fail_at=2)
    io.getStandardDigitalOutput = Mock(side_effect=RuntimeError("readback unavailable"))
    robot = make_robot(io_control=io)

    assert robot.suction_activate() is False
    assert robot.is_suction_on is False
    assert robot.last_gripper_command_trace["partial_write_possible"] is True
    assert robot.last_gripper_command_trace["controller_output_state_known"] is False
    assert robot.last_gripper_command_trace["do_readback_after_failure"] is None
    assert robot.last_gripper_command_trace["do_readback_error"] == "readback unavailable"


def test_target_state_does_not_repeat_do_writes() -> None:
    io = FakeIO(suction_on=False)
    robot = make_robot(io_control=io)

    trace = robot._control_suction_based_on_gripper(0.0)

    assert io.calls == []
    assert trace["requested_transition"] is False
    assert trace["do_api_success"] is None


def configure_send_action(robot: AuboI10Robot) -> None:
    motion = Mock()
    motion.setSpeedFraction.return_value = 0
    motion.isServoModeEnabled.return_value = True
    robot.robot_rpc_client = Mock()
    robot.robot_rpc_client.hasConnected.return_value = True
    robot.robot_interface = Mock()
    robot.robot_interface.getMotionControl.return_value = motion
    robot.is_servo_mode_enabled = True
    robot._send_ee_action_servo_absolute = Mock()


def absolute_action(*, gripper: float) -> dict[str, float | str]:
    return {
        "ee_mode": "absolute",
        "ee.x": 0.1,
        "ee.y": -0.7,
        "ee.z": 0.15,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": gripper,
    }


def test_send_action_propagates_gripper_failure_without_state_change() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False, fail_at=2))
    configure_send_action(robot)

    with pytest.raises(RuntimeError, match="夹爪 DO 状态切换失败"):
        robot.send_action(absolute_action(gripper=100.0))

    assert robot.is_suction_on is False


def test_send_action_success_keeps_return_value_compatible() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    action = absolute_action(gripper=100.0)

    assert robot.send_action(action) is action
    assert robot.is_suction_on is True


def test_send_action_disconnected_raises_instead_of_reporting_success() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    robot.robot_rpc_client = Mock()
    robot.robot_rpc_client.hasConnected.return_value = False

    with pytest.raises(ConnectionError, match="未连接"):
        robot.send_action(absolute_action(gripper=100.0))


def test_send_action_servo_enable_failure_raises() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    robot.is_servo_mode_enabled = False
    robot.enable_servo_mode = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="无法开启伺服模式"):
        robot.send_action(absolute_action(gripper=100.0))


def test_send_action_speed_fraction_failure_raises() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    motion = robot.robot_interface.getMotionControl.return_value
    motion.setSpeedFraction.return_value = -1

    with pytest.raises(RuntimeError, match="设置速度比例失败"):
        robot.send_action(absolute_action(gripper=100.0))


def test_send_action_rejects_incomplete_action() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)

    with pytest.raises(ValueError, match="缺少必要的控制参数"):
        robot.send_action({"ee.x": 0.1, "ee.gripper_pos": 100.0})


@pytest.mark.parametrize("return_code", [-5, 2])
def test_joint_servo_failure_return_code_raises(return_code: int) -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    robot.fixed_axis5_deg = 90.0
    robot.joint_acceleration = 1.0
    robot.joint_velocity = 1.0
    robot.servo_time = 0.04
    robot.servo_blend_radius = 0.0
    robot.servo_max_queue_retry = 1
    motion = Mock()
    motion.servoJoint.return_value = return_code

    with pytest.raises(RuntimeError, match="非 0 码|队列持续满载"):
        robot._send_joint_action_servo(
            {f"J{index}": 0.0 for index in range(1, 7)},
            motion,
        )


def test_j6yaw_servojoint_minus13_retries_after_reenable() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    robot.joint_acceleration = 1.0
    robot.joint_velocity = 1.0
    robot.servo_time = 0.04
    robot.servo_blend_radius = 0.0
    robot.servo_max_queue_retry = 3
    robot.is_servo_mode_enabled = True
    robot.enable_servo_mode = Mock(return_value=True)
    state = Mock()
    state.getJointPositions.return_value = [0.0] * 6
    algo = Mock()
    algo.inverseKinematics.return_value = ([0.0] * 6, 0)
    robot.robot_interface = Mock()
    robot.robot_interface.getRobotState.return_value = state
    robot.robot_interface.getRobotAlgorithm.return_value = algo
    motion = Mock()
    motion.servoJoint.side_effect = [-13, 0]
    action = {
        "ee.x": 0.1,
        "ee.y": -0.7,
        "ee.z": 0.15,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.j6_target": -3.0,
    }

    robot._send_position_j6yaw(action, motion)

    assert motion.servoJoint.call_count == 2
    robot.enable_servo_mode.assert_called_once()


# --- send_action trace freshness (C0-R4A) -----------------------------------


def _stale_trace() -> dict:
    return {"requested_transition": True, "stale_previous_cycle": True}


def test_send_action_clears_previous_trace_before_connection_check() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    robot.robot_rpc_client = Mock()
    robot.robot_rpc_client.hasConnected.return_value = False
    robot.last_gripper_command_trace = _stale_trace()

    with pytest.raises(ConnectionError, match="未连接"):
        robot.send_action(absolute_action(gripper=100.0))

    assert robot.last_gripper_command_trace is None


def test_send_action_clears_previous_trace_before_speed_fraction_failure() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    motion = robot.robot_interface.getMotionControl.return_value
    motion.setSpeedFraction.return_value = -1
    robot.last_gripper_command_trace = _stale_trace()

    with pytest.raises(RuntimeError, match="设置速度比例失败"):
        robot.send_action(absolute_action(gripper=100.0))

    assert robot.last_gripper_command_trace is None


def test_send_action_clears_previous_trace_before_servo_failure() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    robot.is_servo_mode_enabled = False
    robot.enable_servo_mode = Mock(return_value=False)
    robot.last_gripper_command_trace = _stale_trace()

    with pytest.raises(RuntimeError, match="无法开启伺服模式"):
        robot.send_action(absolute_action(gripper=100.0))

    assert robot.last_gripper_command_trace is None


def test_send_action_clears_previous_trace_when_motion_stage_fails_before_gripper() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    robot._send_ee_action_servo_absolute = Mock(side_effect=RuntimeError("IK solve failed"))
    robot.last_gripper_command_trace = _stale_trace()

    with pytest.raises(RuntimeError, match="IK solve failed"):
        robot.send_action(absolute_action(gripper=100.0))

    # Motion failed before any gripper call: no stale trace may survive and
    # no pseudo gripper evidence may be fabricated for this cycle.
    assert robot.last_gripper_command_trace is None
    robot._send_ee_action_servo_absolute.assert_called_once()


def test_send_action_success_produces_its_own_trace_not_previous_one() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    robot.last_gripper_command_trace = _stale_trace()

    assert robot.send_action(absolute_action(gripper=100.0)) is not None
    trace = robot.last_gripper_command_trace

    assert trace is not None
    assert "stale_previous_cycle" not in trace
    assert trace["requested_transition"] is True
    assert trace["do_api_success"] is True


def test_send_action_no_transition_produces_fresh_hold_trace() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False))
    configure_send_action(robot)
    robot.last_gripper_command_trace = _stale_trace()

    robot.send_action(absolute_action(gripper=0.0))

    trace = robot.last_gripper_command_trace
    assert trace is not None
    assert trace["requested_transition"] is False
    assert trace["do_api_success"] is None
    assert trace["commanded_state_before"] is False


def test_send_action_do_failure_produces_its_own_failed_trace() -> None:
    robot = make_robot(io_control=FakeIO(suction_on=False, fail_at=2))
    configure_send_action(robot)
    robot.last_gripper_command_trace = _stale_trace()

    with pytest.raises(RuntimeError, match="夹爪 DO 状态切换失败"):
        robot.send_action(absolute_action(gripper=100.0))

    trace = robot.last_gripper_command_trace
    assert trace is not None
    assert "stale_previous_cycle" not in trace
    assert trace["do_api_success"] is False
    assert trace["do_write_attempt_count"] == 2


def test_sdk_first_write_exception_records_attempt_without_writes() -> None:
    io = FakeIO(suction_on=False, exception_at=1)
    robot = make_robot(io_control=io)

    assert robot.suction_activate() is False

    assert io.calls == [(3, False)]
    trace = robot.last_gripper_command_trace
    assert trace["do_write_attempt_count"] == 1
    assert trace["do_writes"] == []
    assert trace["do_api_success"] is False
    assert trace["partial_write_possible"] is True
    assert "sdk write failed at call 1" in trace["error"]


def test_sdk_second_write_exception_records_only_first_write() -> None:
    io = FakeIO(suction_on=False, exception_at=2)
    robot = make_robot(io_control=io)

    assert robot.suction_activate() is False

    assert io.calls == [(3, False), (2, True)]
    trace = robot.last_gripper_command_trace
    assert trace["do_write_attempt_count"] == 2
    assert trace["do_writes"] == [{"pin": 3, "value": False, "return_code": 0}]
    assert trace["do_api_success"] is False
    assert trace["partial_write_possible"] is True
    assert "sdk write failed at call 2" in trace["error"]
