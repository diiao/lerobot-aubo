"""Unrecorded, operator-controlled positioning before a local demonstration."""

from ..aubo_joint_contract import JOINT_FIELDS, JOINT_FPS, JOINT_TASK, joint_command


def prepare_local_start(*, robot, teleop, events, teleop_action_processor,
                        robot_action_processor, robot_observation_processor, record_loop):
    """Use the existing guarded teleop path, with no dataset or frame observer.

    Right arrow finishes preparation, left arrow repeats it, Esc cancels. No
    waypoint, automatic home or grasp is executed here. The next recording
    resets the phone integrators against its fresh observation.
    """
    try:
        while not events["stop_recording"]:
            events["rerecord_episode"] = False
            teleop_action_processor.reset()
            print("准备遥操（不录制）：移到竹条堆上方，保留待纠正的位置/角度偏差；"
                  "张开夹爪、松开手机移动按钮后按 → 录制，← 重新准备，Esc 停止。")
            try:
                record_loop(robot=robot, teleop=teleop, events=events, fps=JOINT_FPS,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            dataset=None, observer=None, control_time_s=float("inf"),
                            single_task=JOINT_TASK, display_data=False)
            finally:
                if not robot.disable_servo_mode():
                    raise RuntimeError("cannot disable servo after local start preparation")
            if events["stop_recording"]:
                break
            if events["rerecord_episode"]:
                continue
            raw = robot.get_observation()
            state = joint_command({key: raw[key] for key in JOINT_FIELDS})
            if state["gripper_pos"] != 0:
                print("夹爪命令仍为闭合；返回准备遥操，请先张开，再按 →。")
                continue
            print("正式录制：精确对准→夹起→搬运→放下；→ 结束，← 重录，Esc 停止。")
            break
    finally:
        teleop_action_processor.reset()
        events["rerecord_episode"] = events["return_to_start"] = False
        # record_loop consumes exit_early. Restore it on cancel so the following
        # recording cannot send a command or add even one frame.
        events["exit_early"] = bool(events["stop_recording"])
