#!/usr/bin/env python
"""把机器人移动到训练起始关节姿态（moveJoint，非伺服）。

训练 episode 0 第 0 帧的关节角。用前确认工作区清空、急停在手。
会先关闭伺服模式（moveJoint 必须在非伺服下执行）。
"""
import math
import sys
import time

import pyaubo_sdk

# 训练起始关节角（度）—— 当前位姿设为新起点（2026-07-05）
# 注意：旧 10 条数据是旧起点（J1=-23.85,J2=-10.25,J3=104.34,J4=26.11,J5=89.40,J6=-155.92）采集的，
# 用本新起点需重新采集 + 重训，新旧数据不可混用。
TARGET_DEG = [-21.14, -0.98, 113.74, 26.40, 89.31, -153.21]
ROBOT_IP = "192.168.31.200"
ROBOT_PORT = 30004


def wait_arrival(robot_interface):
    """阻塞到运动完成（getExecId 返回 -1 表示无运动/完成）。"""
    exec_id = robot_interface.getMotionControl().getExecId()
    cnt = 0
    while exec_id == -1:  # 等待运动开始
        if cnt > 100:
            return -1
        time.sleep(0.05)
        cnt += 1
        exec_id = robot_interface.getMotionControl().getExecId()
    while robot_interface.getMotionControl().getExecId() != -1:  # 等待完成
        time.sleep(0.05)
    return 0


def main():
    target_rad = [math.radians(d) for d in TARGET_DEG]

    rpc = pyaubo_sdk.RpcClient()
    rpc.setRequestTimeout(2000)
    rpc.connect(ROBOT_IP, ROBOT_PORT)
    if not rpc.hasConnected():
        print("连接失败"); sys.exit(1)
    rpc.login("aubo", "123456")
    if not rpc.hasLogined():
        print("登录失败"); sys.exit(1)

    name = rpc.getRobotNames()[0]
    iface = rpc.getRobotInterface(name)
    motion = iface.getMotionControl()

    # 关闭伺服模式（若上次 eval 留着伺服）
    if motion.isServoModeEnabled():
        print("关闭伺服模式...")
        motion.setServoMode(False)
        time.sleep(0.5)

    cur = iface.getRobotState().getJointPositions()
    print("当前关节角(deg):", [round(math.degrees(x), 1) for x in cur])
    print("目标关节角(deg):", TARGET_DEG)

    max_diff = max(abs(math.degrees(c) - t) for c, t in zip(cur, TARGET_DEG))
    print(f"最大关节差: {max_diff:.1f}°")

    motion.setSpeedFraction(0.5)  # 50% 速度，安全些
    print("执行 moveJoint ...")
    motion.moveJoint(target_rad, 80 * (math.pi / 180), 60 * (math.pi / 180), 0, 0)
    ret = wait_arrival(iface)
    print("moveJoint 结果:", "成功" if ret == 0 else "失败/超时")

    cur = iface.getRobotState().getJointPositions()
    print("到位后关节角(deg):", [round(math.degrees(x), 1) for x in cur])
    err = max(abs(math.degrees(c) - t) for c, t in zip(cur, TARGET_DEG))
    print(f"到位误差: {err:.2f}°")

    rpc.logout()
    rpc.disconnect()
    print("完成，可以跑 eval_trial.py 了")


if __name__ == "__main__":
    main()
