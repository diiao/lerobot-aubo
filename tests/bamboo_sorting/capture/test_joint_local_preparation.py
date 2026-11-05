"""Preparation control boundaries, with fake devices only."""

from unittest.mock import Mock

import pytest

from lerobot.bamboo_sorting.capture.joint_local_preparation import prepare_local_start
from tests.bamboo_sorting.capture_fakes import entry, observation


@pytest.mark.parametrize("ending", ["stop", "failure", "rerecord", "closed"])
def test_preparation_cancel_failure_and_retry_never_record(ending):
    robot = Mock()
    robot.disable_servo_mode.return_value = True
    robot.get_observation.side_effect = [dict(observation(), gripper_pos=100), observation()] \
        if ending == "closed" else [observation()]
    processor = Mock()
    events = dict(exit_early=False, stop_recording=False, rerecord_episode=False, return_to_start=False)
    calls = []
    def run_loop(**kwargs):
        assert kwargs["dataset"] is None and kwargs["observer"] is None
        calls.append(kwargs)
        if ending == "failure":
            raise RuntimeError("SDK failed")
        if ending == "stop":
            events["stop_recording"] = True
        if ending == "rerecord" and len(calls) == 1:
            events["rerecord_episode"] = True
    kwargs = dict(robot=robot, teleop=Mock(), events=events, teleop_action_processor=processor,
                  robot_action_processor=Mock(), robot_observation_processor=Mock(), record_loop=run_loop)
    if ending == "failure":
        with pytest.raises(RuntimeError, match="SDK failed"):
            prepare_local_start(**kwargs)
    else:
        prepare_local_start(**kwargs)
    expected = 2 if ending in ("rerecord", "closed") else 1
    assert len(calls) == robot.disable_servo_mode.call_count == expected
    assert processor.reset.call_count == expected + 1
    assert events["exit_early"] == (ending == "stop")
    assert not events["rerecord_episode"] and not events["return_to_start"]
    robot.send_action.assert_not_called()


def test_local_plan_is_explicit_and_writes_nothing(tmp_path, capsys):
    args = ["--dataset-root", str(tmp_path/"data"), "--evidence-root", str(tmp_path/"evidence"),
            "--split", "train", "--demo-scope", "local_alignment"]
    module = entry()
    with pytest.raises(SystemExit):
        module.parse_args(args)
    assert module.main([*args, "--stack-demonstrations"]) == 0
    plan_output = capsys.readouterr().out
    assert '"demo_scope": "local_alignment"' in plan_output
    assert "start_pose_group" not in plan_output
    assert '"fresh_camera_frame_timeout_s": 0.5' in plan_output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stop_timing", ["before_restart", "during_restart", "none"])
def test_listener_handoff_does_not_erase_a_pending_stop(monkeypatch, stop_timing):
    from lerobot.utils import control_utils
    events = dict(exit_early=False, stop_recording=False, rerecord_episode=False, return_to_start=False)
    def stop():
        events.update(stop_recording=True, exit_early=True)
    if stop_timing == "before_restart":
        stop()  # Esc delivered while the preceding listener was stopping.
    def listener(*args, **kwargs):
        if stop_timing == "during_restart":
            stop()
        return Mock()
    monkeypatch.setattr(control_utils, "_StdinHotkeyListener", listener)
    monkeypatch.setattr(control_utils, "is_headless", lambda: True)
    monkeypatch.setattr(control_utils.sys.stdin, "isatty", lambda: False)
    handle, returned = control_utils.init_keyboard_listener(events, enable_return_to_start=False, reset_events=False)
    assert returned is events
    robot, loop = Mock(), Mock()
    robot.disable_servo_mode.return_value = True
    robot.get_observation.return_value = observation()
    prepare_local_start(robot=robot, teleop=Mock(), events=events,
                        teleop_action_processor=Mock(), robot_action_processor=Mock(),
                        robot_observation_processor=Mock(), record_loop=loop)
    handle.stop()
    assert loop.call_count == (1 if stop_timing == "none" else 0)
    assert events["stop_recording"] == (stop_timing != "none")
    assert events["exit_early"] == (stop_timing != "none")
    if stop_timing != "none":
        robot.get_observation.assert_not_called()


def test_record_loop_honors_stop_even_if_phase_exit_was_consumed():
    from lerobot.scripts.lerobot_record import record_loop
    events = dict(exit_early=False, stop_recording=True)
    robot = Mock()
    record_loop(robot=robot, events=events, fps=25, control_time_s=1,
                teleop_action_processor=Mock(), robot_action_processor=Mock(), robot_observation_processor=Mock())
    robot.get_observation.assert_not_called()
    robot.send_action.assert_not_called()
    assert events["stop_recording"] is True
