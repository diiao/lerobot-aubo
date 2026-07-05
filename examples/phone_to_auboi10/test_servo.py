#!/usr/bin/env python
"""诊断 servoCartesian 在 absolute 模式下返回 -5 的根因。
逐步测试：用当前 TCP 位姿原样下发，看是否被拒。
不发增量，只发"原地"目标。
"""
import math
import time
import pyaubo_sdk


def main():
    rpc = pyaubo_sdk.RpcClient()
    rpc.setRequestTimeout(1000)
    rpc.connect("192.168.31.200", 30004)
    rpc.login("aubo", "123456")
    name = rpc.getRobotNames()[0]
    iface = rpc.getRobotInterface(name)
    motion = iface.getMotionControl()
    state = iface.getRobotState()

    cur = state.getTcpPose()
    print(f"当前 TCP: pos=[{cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f}] "
          f"rotvec=[{cur[3]:.3f},{cur[4]:.3f},{cur[5]:.3f}]")

    def reset_servo():
        motion.setServoMode(False)
        for _ in range(40):
            if not motion.isServoModeEnabled():
                break
            time.sleep(0.005)
        motion.setServoMode(True)
        for _ in range(40):
            if motion.isServoModeEnabled():
                break
            time.sleep(0.005)
        time.sleep(0.1)

    motion.setServoMode(True)
    for _ in range(20):
        if motion.isServoModeEnabled():
            break
        time.sleep(0.005)
    print(f"伺服模式: {motion.isServoModeEnabled()}")

    # 测试1: 原位姿原样下发 (acc=vel=0, time=0.1 纯时间模式)
    print("\n[测试1] 原位姿 + 纯时间模式 (acc=vel=0, time=0.1)")
    for i in range(5):
        ret = motion.servoCartesian(cur, 0.0, 0.0, 0.1, 0.0, 0.0)
        print(f"  帧{i} ret={ret}")
        time.sleep(0.05)

    # 测试2: 原位姿 + 旧参数 (acc=1.2, vel=0.25, time=0.05)
    reset_servo()
    print("\n[测试2] 原位姿 + 旧参数 (acc=1.2, vel=0.25, time=0.05)")
    for i in range(5):
        ret = motion.servoCartesian(cur, 1.2, 0.25, 0.05, 0.0, 0.0)
        print(f"  帧{i} ret={ret}")
        time.sleep(0.05)

    # 测试3: rotvec 双值性 — 用反向表示 [2.155,0.013,1.378] (与 cur 等价的另一种)
    reset_servo()
    alt = [cur[0], cur[1], cur[2], 2.155, 0.013, 1.378]
    print(f"\n[测试3] 等价 rotvec [2.155,0.013,1.378] + 纯时间模式")
    for i in range(5):
        ret = motion.servoCartesian(alt, 0.0, 0.0, 0.1, 0.0, 0.0)
        print(f"  帧{i} ret={ret}")
        time.sleep(0.05)

    # 测试4: 对齐后的 rotvec (与 cur 等价但表示接近 cur)
    reset_servo()
    import numpy as np
    from lerobot.robots.aubo_i10.robot_processor import _align_rotvec
    aligned = _align_rotvec(np.array([2.155, 0.013, 1.378]), np.array(cur[3:6]))
    alt2 = [cur[0], cur[1], cur[2], float(aligned[0]), float(aligned[1]), float(aligned[2])]
    print(f"\n[测试4] 对齐 rotvec [{alt2[3]:.3f},{alt2[4]:.3f},{alt2[5]:.3f}] + 纯时间模式")
    for i in range(5):
        ret = motion.servoCartesian(alt2, 0.0, 0.0, 0.1, 0.0, 0.0)
        print(f"  帧{i} ret={ret}")
        time.sleep(0.05)

    motion.setServoMode(False)
    rpc.logout()
    rpc.disconnect()


if __name__ == "__main__":
    main()
