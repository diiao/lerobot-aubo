#!/usr/bin/env python
"""把机器人移动到训练起始关节姿态（moveJoint，非伺服）。

训练 episode 0 第 0 帧的关节角。用前确认工作区清空、急停在手。
会先关闭伺服模式（moveJoint 必须在非伺服下执行）。
"""
import math
import sys
import time

import pyaubo_sdk

# 训练起始关节角(度) -- 48/50 条数据真实起始位姿中位数 (EE约 0.11,-0.72,0.15)。旧 [-21.14,...] 只 ep0/ep24 覆盖、分布外，评估乱抖，故改用真实起始位。
TARGET_DEG = [-65.29, -5.88, 113.77, 31.07, 90.88, -185.32]
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

    # 松开吸盘（防止上一轮评估吸合后未释放，回起点时拖动物料导致下轮场景偏离）
    try:
        io = iface.getIoControl()
        io.setStandardDigitalOutput(2, False)  # 端口2(吸)拉低
        io.setStandardDigitalOutput(3, True)   # 端口3(放)拉高
        print("吸盘已释放（端口2=OFF, 端口3=ON）")
    except Exception as e:
        print(f"吸盘释放失败: {e}")

    rpc.logout()
    rpc.disconnect()
    print("完成，可以跑 evaluate_split.py 了")


if __name__ == "__main__":
    main()
