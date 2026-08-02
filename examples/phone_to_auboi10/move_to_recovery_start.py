#!/usr/bin/env python
"""把 AUBO i10 移到可重复的 ACT 恢复示教起点。

该脚本来自 bamboo_newview_eval_run05_trial02 的真实失败回放：末端已经到达竹条上方，
但模型没有继续下降或吸取。恢复示教应从这个状态连续示范下降、吸取、抬升、放置和释放。

安全设计：
* 不带 --confirm 时只读取并展示当前状态，绝不发送运动命令；
* 仅允许从统一正常起点附近（最大关节误差 <= 5 度）出发，避免从未知姿态直接运动；
* 带 --confirm 后以 20% 速度执行 moveJoint，并在到位后释放吸盘。

运行前必须由现场操作者确认工作区清空、急停可用、没有人员处于机械臂活动范围内。
"""

import argparse
import math
import sys
import time

import pyaubo_sdk


ROBOT_IP = "192.168.31.200"
ROBOT_PORT = 30004

# 当前 30 条基线的统一正常录制/推理起点；恢复动作只允许从此姿态附近开始。
NORMAL_START_DEG = [-65.29, -5.88, 113.77, 31.07, 90.88, -185.32]

# run05 trial02 第 472 帧的实际关节状态：TCP 约 (0.5749, -0.4706, 0.1716) m，
# 即竹条上方的典型闭环失败状态。它不是抓取点，也不会替代 NORMAL_START_DEG。
RECOVERY_START_DEG = [-23.27, -5.91, 111.13, 28.57, 89.55, -162.21]
EXPECTED_TCP_XYZ_M = [0.5749, -0.4706, 0.1716]

NORMAL_START_TOLERANCE_DEG = 5.0
SPEED_FRACTION = 0.20
JOINT_SPEED_RAD_S = math.radians(30.0)
JOINT_ACCEL_RAD_S2 = math.radians(30.0)


def joint_angles_deg(joints_rad):
    return [math.degrees(joint) for joint in joints_rad]


def max_joint_error_deg(actual_deg, target_deg):
    return max(abs(actual - target) for actual, target in zip(actual_deg, target_deg))


def wait_arrival(robot_interface, start_timeout_s=5.0):
    """等待 AUBO 的 moveJoint 启动并完成。"""
    motion = robot_interface.getMotionControl()
    deadline = time.monotonic() + start_timeout_s
    while motion.getExecId() == -1:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)

    while motion.getExecId() != -1:
        time.sleep(0.05)
    return True


def print_state(iface, label):
    state = iface.getRobotState()
    joints_deg = joint_angles_deg(state.getJointPositions())
    tcp = state.getTcpPose()
    print(f"\n=== {label} ===")
    print("关节角 (deg):", [round(value, 2) for value in joints_deg])
    print(f"TCP x,y,z (m): {tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}")
    return joints_deg, tcp


def release_suction(iface):
    """到达恢复起点后显式设为释放状态；失败时只报告，不掩盖移动结果。"""
    io = iface.getIoControl()
    io.setStandardDigitalOutput(2, False)
    io.setStandardDigitalOutput(3, True)
    print("吸盘已释放（端口2=OFF, 端口3=ON）")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="确认已检查工位和急停后，才实际执行低速 moveJoint。缺省只预览。",
    )
    args = parser.parse_args()

    rpc = pyaubo_sdk.RpcClient()
    try:
        rpc.setRequestTimeout(2000)
        rpc.connect(ROBOT_IP, ROBOT_PORT)
        if not rpc.hasConnected():
            print("连接失败")
            return 1
        rpc.login("aubo", "123456")
        if not rpc.hasLogined():
            print("登录失败")
            return 1

        name = rpc.getRobotNames()[0]
        iface = rpc.getRobotInterface(name)
        current_deg, _ = print_state(iface, "当前状态")
        start_error = max_joint_error_deg(current_deg, NORMAL_START_DEG)
        print("\n统一正常起点 (deg):", NORMAL_START_DEG)
        print("恢复示教起点 (deg):", RECOVERY_START_DEG)
        print(f"距统一正常起点的最大关节误差: {start_error:.2f}°")

        if start_error > NORMAL_START_TOLERANCE_DEG:
            print(
                f"拒绝运动：当前姿态不在统一正常起点 ±{NORMAL_START_TOLERANCE_DEG:.1f}° 内。"
            )
            print("请先由现场操作者使用 move_to_start.py 归位，再重新运行本脚本。")
            return 2

        if not args.confirm:
            print("\n预览完成：未发送任何运动或 IO 指令。")
            print("确认工作区清空、急停可用、无人员在活动范围内后，执行：")
            print("  ../../.venv/bin/python move_to_recovery_start.py --confirm")
            return 0

        motion = iface.getMotionControl()
        if motion.isServoModeEnabled():
            print("关闭伺服模式...")
            motion.setServoMode(False)
            time.sleep(0.5)

        print(
            f"\n执行恢复起点 moveJoint（速度比例 {SPEED_FRACTION:.0%}，"
            "请持续现场监护）..."
        )
        motion.setSpeedFraction(SPEED_FRACTION)
        target_rad = [math.radians(value) for value in RECOVERY_START_DEG]
        motion.moveJoint(target_rad, JOINT_SPEED_RAD_S, JOINT_ACCEL_RAD_S2, 0, 0)
        if not wait_arrival(iface):
            print("moveJoint 未在 5 秒内启动；未执行后续吸盘 IO。")
            return 3

        arrived_deg, tcp = print_state(iface, "到位后状态")
        joint_error = max_joint_error_deg(arrived_deg, RECOVERY_START_DEG)
        tcp_xyz_error_mm = 1000.0 * math.dist(tcp[:3], EXPECTED_TCP_XYZ_M)
        print(f"到位最大关节误差: {joint_error:.2f}°")
        print(f"相对记录恢复 TCP 的位置差: {tcp_xyz_error_mm:.1f} mm")

        try:
            release_suction(iface)
        except Exception as exc:
            print(f"吸盘释放失败: {exc}")
            return 4

        print("完成。现在可以启动 record.py，并在开始提示时不要按 r。")
        return 0
    finally:
        if rpc.hasLogined():
            rpc.logout()
        if rpc.hasConnected():
            rpc.disconnect()


if __name__ == "__main__":
    sys.exit(main())
