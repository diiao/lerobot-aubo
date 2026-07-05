#!/usr/bin/env python
"""SO101 leader → Aubo I10 末端位置 + J6 直控遥操（EE 增量 + 姿态锁竖直 + wrist_roll→J6）。

控制律与 phone 的 abs_j6yaw 模式同构，leader 从手机换成 SO101：
  leader.get_action() (5 关节度 + gripper.pos)
    → FK → leader 末端位置(leader 基座系)
    → 位置: 增量 + 非线性鼠标增益(小位移精确、大位移快) → Aubo EE 绝对位置
    → 姿态: 锁定启动 TCP(竖直), pitch/roll/yaw 全锁死, 不跟 leader
    → J6: wrist_roll 相对启动角的增量 × 比例 → J6 绝对目标(弧度), 直接驱动末轴
    → send_action(ee_mode="abs_j6yaw") → IK 求 J1-J5(位置+竖直姿态), J6 直控 servoJoint
  gripper.pos → ee.gripper_pos → 软爪

与 teleop_fk.py 的区别：
  - teleop_fk 用 servoCartesian 全姿态伺服（IK 会把偏航误分到 J4/J5，绕竖直转末端会俯仰）。
  - 本脚本用 abs_j6yaw：姿态锁死竖直，偏航只由 J6 承担（wrist_roll 直控），彻底规避该问题。
  位置映射逻辑（FK + 增量 + 非线性鼠标增益 + 启动待机 + 软启动 + 下降增益）与 teleop_fk 一致。

前提：两臂基座系朝向一致（X前/Y左/Z上，R_align=I）；已生成 calibration.json；placo 已装。
用法：PYTHONPATH=src python examples/so101_to_auboi10/teleop_ee_j6.py
"""
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Config, AuboI10Robot
from lerobot.robots.aubo_i10.robot_processor import AuboEEBoundsAndSafety
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

URDF_PATH = Path(__file__).parent / "so101_new_calib.urdf"
CALIB_PATH = Path(__file__).parent / "calibration.json"
TARGET_FRAME = "gripper_frame_link"
FPS = 50  # 喂点频率；高频采样让小动作更跟手

# ── 位置映射（与 teleop_fk 一致：增量 + 非线性鼠标增益）─────────────────────────
# 非线性增益：小幅输入低增益(精细微调)、大幅输入高增益(快速大范围移动)。
# 单帧 leader 位移 |d|(米) 的增益 = FINE_GAIN_MIN + (FINE_GAIN_MAX-FINE_GAIN_MIN) * |d|/FINE_KNEE，
# 在 |d| >= FINE_KNEE 后饱和到 FINE_GAIN_MAX。即对 delta 整体乘一个随模长递增的系数。
FINE_GAIN_MIN = 0.5
FINE_GAIN_MAX = 1.2
FINE_KNEE = 0.01  # 米：增益饱和阈值

MAX_POS_STEP_M = 0.0025   # 单帧最大位置步长(各向同性，限速)
DOWN_GAIN = 0.6           # 下降方向(Z负)增益：防戳桌锁死；1.0=关闭

# 启动待机门控：启动后 leader 必须先累计移动 > ARM_UNLOCK_M 才解锁跟随，避免首帧大跳碰撞。
ARM_UNLOCK_M = 0.03
SOFTSTART_FRAMES = 100    # 解锁后软启动帧数(约2秒@50Hz)，增益 0→1 平滑接手

DEADZONE_GRIPPER = 1.0    # 夹爪死区(0-100)，避免抖动误触发软爪

# ── J6 直控（wrist_roll → J6）──────────────────────────────────────────────────
# j6_target = j6_ref + radians((wrist_roll - wrist_roll0) * J6_DIR * J6_SCALE)
# 相对启动角映射：松手归位 → J6 归位；J6_DIR 反方向，J6_SCALE 调幅度。
J6_DIR = 1.0       # 1=同向；-1=反向
J6_SCALE = 1.0     # wrist_roll 转 J6 的比例；1.0=1:1
J6_LIMIT_DEG = 350.0  # J6 绝对目标钳位(±)，防撞硬限位

SERVO_TIME = 1.0 / FPS  # servoJoint 每路点插值时长 = 1/FPS


def load_calibration():
    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"找不到 {CALIB_PATH}，请先跑 calibrate.py")
    calib = json.loads(CALIB_PATH.read_text())
    calib["R_align"] = np.array(calib["R_align"])
    calib["t"] = np.array(calib["t"])
    return calib


def leader_ee_position(action, kinematics, motor_names, calib):
    """leader 关节 → FK 末端位置(经 scale·R_align 映射到 Aubo 系，未加基准偏移)。"""
    q = np.array([action[f"{m}.pos"] for m in motor_names], dtype=float)
    T = kinematics.forward_kinematics(q)
    p_leader = T[:3, 3]
    return calib["scale"] * (calib["R_align"] @ p_leader)


def main():
    init_logging()
    logging.getLogger().setLevel(logging.INFO)

    calib = load_calibration()
    motor_names = calib["motor_names"]
    kinematics = RobotKinematics(str(URDF_PATH), TARGET_FRAME, joint_names=motor_names)
    assert list(kinematics.joint_names) == list(motor_names), (
        f"joint_names 顺序不一致！kinematics={kinematics.joint_names} vs motor_names={motor_names}"
    )

    leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", id="so101_leader", use_degrees=True))
    follower = AuboI10Robot(AuboI10Config(id="aubo_i10"))

    safety = AuboEEBoundsAndSafety(
        end_effector_bounds={"min": calib["bounds_min"], "max": calib["bounds_max"]},
        max_ee_step_m=MAX_POS_STEP_M,
    )

    leader.connect()
    follower.connect()
    if not leader.is_connected or not follower.is_connected:
        raise RuntimeError("leader 或 follower 未连接")

    # servoJoint 时长对齐 FPS（abs_j6yaw 走关节伺服）
    follower.servo_time = SERVO_TIME

    obs = follower.get_observation()
    cur_tcp = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=float)
    safety._last_pos = cur_tcp.copy()
    # 姿态锁定为启动 TCP（竖直），全程不跟随 leader
    locked_rotvec = np.array([obs["ee.wx"], obs["ee.wy"], obs["ee.wz"]], dtype=float)
    # J6 基准：启动时的 J6（弧度）+ 启动时 wrist_roll 读数
    j6_ref = math.radians(float(obs["J6"]))
    logging.info(f"当前 TCP: {cur_tcp}，姿态锁定 rotvec={locked_rotvec}，J6 基准={math.degrees(j6_ref):.1f}°")

    # 手机式增量跟随：维护"当前目标点"，每帧加 leader 位移增量。停手 → 目标不变 → 即停。
    action0 = leader.get_action()
    p_prev = leader_ee_position(action0, kinematics, motor_names, calib).copy()
    leader_start = p_prev.copy()
    cur_target_pos = cur_tcp.copy()
    p_cmd_prev = cur_tcp.copy()
    wrist_roll0 = float(action0["wrist_roll.pos"])

    last_gripper = None
    frame_idx = 0
    unlocked = False
    accum_move = 0.0
    softstart_idx = 0

    print(f"开始 EE+J6 直控遥操 (FPS={FPS})。先移动小臂累计>{ARM_UNLOCK_M*100:.0f}cm 解锁。Ctrl-C 退出。")
    try:
        while True:
            t0 = time.perf_counter()
            action = leader.get_action()
            p_now = leader_ee_position(action, kinematics, motor_names, calib)
            delta = p_now - p_prev
            p_prev = p_now.copy()
            if np.linalg.norm(delta) < 1e-4:  # 滤静止抖动
                delta = np.zeros(3)

            # 启动待机门控
            if not unlocked:
                accum_move = float(np.linalg.norm(p_now - leader_start))
                if accum_move > ARM_UNLOCK_M:
                    unlocked = True
                    logging.info(f"跟随已解锁(leader 累计 {accum_move*100:.1f}cm)，软启动中")
                delta = np.zeros(3)
                gain = 0.0
            else:
                gain = min(1.0, softstart_idx / SOFTSTART_FRAMES) if SOFTSTART_FRAMES > 0 else 1.0
                softstart_idx += 1
                d_norm = float(np.linalg.norm(delta))
                if d_norm > 1e-9:
                    fine = FINE_GAIN_MIN + (FINE_GAIN_MAX - FINE_GAIN_MIN) * min(1.0, d_norm / FINE_KNEE)
                    gain *= fine
            if delta[2] < 0.0:  # 下降方向额外降增益
                delta = delta.copy()
                delta[2] *= DOWN_GAIN
            cur_target_pos = cur_target_pos + delta * gain

            # 位置：边界裁剪 + 相对上次下发的单帧步长硬 clamp（防漂移、防大跳）
            p_desired = np.clip(cur_target_pos, calib["bounds_min"], calib["bounds_max"])
            step = p_desired - p_cmd_prev
            step_norm = float(np.linalg.norm(step))
            if step_norm > MAX_POS_STEP_M and step_norm > 1e-9:
                step = step * (MAX_POS_STEP_M / step_norm)
            p_cmd = p_cmd_prev + step
            p_cmd_prev = p_cmd.copy()

            # J6：wrist_roll 相对启动角 → J6 绝对目标(弧度)，钳位 ±J6_LIMIT_DEG
            wr = float(action["wrist_roll.pos"])
            j6_target = j6_ref + math.radians((wr - wrist_roll0) * J6_DIR * J6_SCALE)
            j6_limit = math.radians(J6_LIMIT_DEG)
            j6_target = max(-j6_limit, min(j6_limit, j6_target))

            gripper_pos = float(action["gripper.pos"])
            # 夹爪死区：变化太小则保持上次，避免抖动误触发软爪
            if last_gripper is not None and abs(gripper_pos - last_gripper) < DEADZONE_GRIPPER:
                gripper_pos = last_gripper
            last_gripper = gripper_pos

            cmd = {
                "ee.x": float(p_cmd[0]),
                "ee.y": float(p_cmd[1]),
                "ee.z": float(p_cmd[2]),
                "ee.wx": float(locked_rotvec[0]),
                "ee.wy": float(locked_rotvec[1]),
                "ee.wz": float(locked_rotvec[2]),
                "ee.j6_target": float(j6_target),
                "ee.gripper_pos": gripper_pos,
                "ee_mode": "abs_j6yaw",
            }
            follower.send_action(cmd)
            frame_idx += 1
            if frame_idx % 100 == 0:
                logging.debug(
                    "p=[%.3f,%.3f,%.3f] J6=%.1f° grip=%.0f",
                    p_cmd[0], p_cmd[1], p_cmd[2], math.degrees(j6_target), gripper_pos,
                )

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n遥操停止")
    finally:
        try:
            follower._control_softpaws_based_on_gripper(0.0)
        except Exception:
            pass
        try:
            follower.disable_servo_mode()
        except Exception:
            pass
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
