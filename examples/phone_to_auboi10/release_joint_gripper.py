#!/usr/bin/env python
"""Release the pneumatic gripper only; no cameras, homing or arm commands.

Without --execute, print the plan without connecting hardware. The operator
must clear the gripper and confirm it is not holding anything before execution.
"""
import argparse
import json

from lerobot.bamboo_sorting.runtime.station import ReadOnlyStation


def release_once(io):
    from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
    from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config

    robot = AuboI10Robot(AuboI10Config())
    # Attach only the existing IO connection. Do not call robot.connect(),
    # robot.disconnect() or acquire/send arm motion commands.
    robot.io_control = io
    try:
        if not robot.suction_release():
            raise RuntimeError('gripper release failed; inspect DO readback before retrying')
    finally:
        print(json.dumps(robot.last_gripper_command_trace, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Operator confirms the gripper is clear and not holding anything; send release')
    args = parser.parse_args(argv)
    if not args.execute:
        print('计划：仅松开夹爪，DO2=OFF → DO3=ON，核对读回后断开连接并保留输出。'
              '\n不连接相机、不移动或归位。现场确认未持物且手已离开后，加 --execute 执行。')
        return 0

    station = ReadOnlyStation()
    try:
        station.connect(include_cameras=False)
        try:
            state, _, _ = station.current(require_stationary=True)
        except ValueError as exc:
            # current() checks power, safety and stationarity before the DOs.
            # Both low is the known uninitialized pair; other failures stop.
            if str(exc) != 'unknown suction output pair: (False, False)':
                raise
        else:
            if state[-1] == 0:
                print('读回已是 DO2=OFF、DO3=ON，无需重复发送；请目视确认夹爪张开。')
                return 0
        print('发送夹爪松开命令：DO2=OFF → DO3=ON。', flush=True)
        release_once(station.io_readback)
        state, _, _ = station.current(require_stationary=True)
        if state[-1] != 0:
            raise RuntimeError('release output state was not confirmed')
        print('松开命令及读回通过：DO2=OFF、DO3=ON。请目视确认夹爪张开，再运行影子采集。')
        return 0
    finally:
        station.close()


if __name__ == '__main__':
    raise SystemExit(main())
