#!/usr/bin/env python
"""读取 Aubo I10 当前 TCP 位姿，姿态为 rotvec（弧度）。

用途：确定"垂直向下"等目标姿态对应的 rotvec 值。
不发送任何运动指令，只读。
"""
import math
import pyaubo_sdk


def main():
    rpc = pyaubo_sdk.RpcClient()
    rpc.setRequestTimeout(1000)
    rpc.connect("192.168.31.200", 30004)
    if not rpc.hasConnected():
        print("连接失败")
        return
    rpc.login("aubo", "123456")
    name = rpc.getRobotNames()[0]
    iface = rpc.getRobotInterface(name)
    state = iface.getRobotState()

    pose = state.getTcpPose()  # [x, y, z, rx, ry, rz] 米 + rotvec 弧度
    joints = state.getJointPositions()  # 弧度

    print("=== 当前 TCP 位姿 ===")
    print(f"位置  x,y,z (m) = {pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}")
    print(f"姿态  rotvec (rad) = {pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}")
    print(f"姿态  rotvec (deg) = {math.degrees(pose[3]):.2f}, "
          f"{math.degrees(pose[4]):.2f}, {math.degrees(pose[5]):.2f}")
    angle = math.sqrt(pose[3] ** 2 + pose[4] ** 2 + pose[5] ** 2)
    print(f"rotvec 模长 (deg) = {math.degrees(angle):.2f}")

    print("\n=== 当前关节角 (deg) ===")
    for i, j in enumerate(joints, 1):
        print(f"  J{i} = {math.degrees(j):.2f}")

    rpc.logout()
    rpc.disconnect()


if __name__ == "__main__":
    main()
