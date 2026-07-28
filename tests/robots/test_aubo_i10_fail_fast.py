from unittest.mock import Mock

import pytest

from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config


def _bare_robot(*, cameras: dict, connected: bool = False) -> AuboI10Robot:
    robot = object.__new__(AuboI10Robot)
    robot.cameras = cameras
    robot.robot_ip = "192.0.2.1"
    robot.robot_port = 30004
    robot.robot_interface = None
    robot.io_control = None
    robot.is_servo_mode_enabled = False
    robot.robot_rpc_client = Mock()
    robot.robot_rpc_client.hasConnected.return_value = connected
    robot.robot_rpc_client.hasLogined.return_value = False
    return robot


def test_connect_stops_before_robot_when_camera_fails():
    camera = Mock()
    camera.connect.side_effect = ConnectionError("camera unavailable")
    camera.is_connected = False
    robot = _bare_robot(cameras={"handeye": camera})

    with pytest.raises(ConnectionError, match="configured camera 'handeye'"):
        robot.connect()

    camera.connect.assert_called_once_with(warmup=True)
    robot.robot_rpc_client.connect.assert_not_called()


def test_get_observation_raises_instead_of_returning_black_placeholder():
    camera = Mock()
    camera.read_latest.side_effect = TimeoutError("stale")
    camera.async_read.side_effect = TimeoutError("no new frame")
    robot = _bare_robot(cameras={"handeye": camera})

    with pytest.raises(RuntimeError, match="避免黑帧污染数据"):
        robot.get_observation()


def test_disconnect_releases_connected_cameras():
    camera = Mock()
    camera.is_connected = True
    robot = _bare_robot(cameras={"handeye": camera})

    robot.disconnect()

    camera.disconnect.assert_called_once_with()
    robot.robot_rpc_client.disconnect.assert_called_once_with()


def test_servo_period_follows_configured_control_fps():
    robot = AuboI10Robot(AuboI10Config(control_fps=25))

    assert robot.servo_time == pytest.approx(0.04)


def test_control_fps_must_be_positive():
    with pytest.raises(ValueError, match="control_fps"):
        AuboI10Config(control_fps=0)
