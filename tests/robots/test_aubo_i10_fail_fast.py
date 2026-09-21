from unittest.mock import Mock
import threading

import numpy as np

import pytest

from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


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
    robot.last_c0_sensor_timestamps = {"stale": 1.0}

    with pytest.raises(RuntimeError, match="避免黑帧污染数据"):
        robot.get_observation()
    assert robot.last_c0_sensor_timestamps is None


def test_get_observation_exposes_fresh_host_monotonic_sensor_timestamps():
    camera = Mock()
    camera.read_latest.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    camera.frame_lock = threading.Lock()
    camera.latest_timestamp = 12.5
    robot = _bare_robot(cameras={"global_rgb": camera}, connected=True)
    robot.is_suction_on = False
    state = Mock()
    state.getJointPositions.return_value = [0.0] * 6
    state.getTcpPose.return_value = [0.0] * 6
    robot.robot_interface = Mock()
    robot.robot_interface.getRobotState.return_value = state

    observation = robot.get_observation()

    assert "global_rgb" in observation
    assert robot.last_c0_sensor_timestamps["global_rgb"] == 12.5
    assert robot.last_c0_sensor_timestamps["robot_state"] >= 0.0


@pytest.mark.parametrize("fallback", [False, True])
def test_opencv_observation_keeps_image_and_timestamp_paired(monkeypatch, fallback):
    """Publish a newer frame just after unlocking, including the async fallback."""
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0))
    monkeypatch.setattr(OpenCVCamera, "is_connected", property(lambda self: True))
    monkeypatch.setattr("lerobot.cameras.opencv.camera_opencv.time.perf_counter", lambda: 12.6)
    camera.thread = Mock()
    camera.thread.is_alive.return_value = True
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones_like(first)
    camera.latest_frame = first
    camera.latest_timestamp = 10.0 if fallback else 12.5

    class PublishAfterUnlock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            camera.latest_frame = second
            camera.latest_timestamp = 12.6

    camera.frame_lock = PublishAfterUnlock()
    camera.new_frame_event = Mock()

    def new_frame(timeout):
        camera.latest_frame = first
        camera.latest_timestamp = 12.5
        return True

    camera.new_frame_event.wait.side_effect = new_frame
    robot = _bare_robot(cameras={"global_rgb": camera})
    observation = robot.get_observation()

    assert observation["global_rgb"] is first
    assert camera.latest_frame is second
    assert robot.last_c0_sensor_timestamps["global_rgb"] == 12.5
    assert camera.new_frame_event.wait.call_count == int(fallback)


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
