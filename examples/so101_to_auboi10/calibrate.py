#!/usr/bin/env python
"""SO101 leader 基座系 → Aubo I10 基座系 标定脚本。

采集若干对应点对 (leader 末端 FK 位姿, Aubo TCP 位姿)，解算:
  - R_align: leader 基座系 → Aubo 基座系 的旋转（由姿态对应点用四元数平均求解）
  - scale s、平移 t: 位置相似变换 p_aubo = t + s * (R_align @ p_leader)
  - neutral_rotvec: Aubo 中性姿态(吸盘平贴)的 rotvec，遥操时对"相对中性的偏差"
    做 ZYX 分解，锁定左右侧倾(roll)，控制 heading(yaw)与前后俯仰(pitch)

结果写入 calibration.json，供 teleop_fk.py 加载。

用法:
  PYTHONPATH=src python examples/so101_to_auboi10/calibrate.py
"""
import json
import math
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Config, AuboI10Robot
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.rotation import Rotation

from calib_math import (
    IDEAL_R_ALIGN,
    solve_scale_translation_fixed_r,
    angle_between,
)

URDF_PATH = Path(__file__).parent / "so101_new_calib.urdf"
CALIB_PATH = Path(__file__).parent / "calibration.json"
TARGET_FRAME = "gripper_frame_link"
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


# --------------------------------------------------------------------------- #
# 采集与主流程
# --------------------------------------------------------------------------- #
def _read_leader_pose(leader, kinematics):
    action = leader.get_action()
    q = np.array([action[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
    T = kinematics.forward_kinematics(q)
    p = T[:3, 3]
    R = Rotation.from_matrix(T[:3, :3]).as_matrix()
    return p, R, float(action["gripper.pos"])


def _read_aubo_pose(robot):
    obs = robot.get_observation()
    p = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=float)
    rv = np.array([obs["ee.wx"], obs["ee.wy"], obs["ee.wz"]], dtype=float)
    R = Rotation.from_rotvec(rv).as_matrix()
    return p, R


def main():
    print(f"加载 URDF: {URDF_PATH}")
    kinematics = RobotKinematics(str(URDF_PATH), TARGET_FRAME, joint_names=MOTOR_NAMES)
    print(f"kinematics.joint_names: {kinematics.joint_names}")
    assert kinematics.joint_names == MOTOR_NAMES, "joint_names 顺序与 MOTOR_NAMES 不一致！"

    leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", id="so101_leader", use_degrees=True))
    robot = AuboI10Robot(AuboI10Config(id="aubo_i10"))
    leader.connect()
    robot.connect()
    if not leader.is_connected or not robot.is_connected:
        raise RuntimeError("leader 或 robot 未连接")

    pairs = []
    n_fit = int(input("输入标定点数(建议 6-8): ").strip() or "6")
    for i in range(n_fit):
        input(f"\n[点 {i + 1}/{n_fit}] 把 leader 末端放到标定点，回车读取 leader...")
        p_l, R_l, _ = _read_leader_pose(leader, kinematics)
        print(f"  leader FK: p={p_l}")
        input(f"  用示教器把 Aubo TCP 移到同一物理点，回车读取 Aubo...")
        p_a, R_a = _read_aubo_pose(robot)
        print(f"  aubo TCP:  p={p_a}")
        pairs.append((p_l, R_l, p_a, R_a))

    # hold-out 验证点
    n_val = int(input("\n输入验证点数(建议 2，可填 0): ").strip() or "2")
    val_pairs = []
    for i in range(n_val):
        input(f"\n[验证点 {i + 1}/{n_val}] 放好 leader，回车...")
        p_l, R_l, _ = _read_leader_pose(leader, kinematics)
        input(f"  Aubo TCP 移到同一点，回车...")
        p_a, R_a = _read_aubo_pose(robot)
        val_pairs.append((p_l, R_l, p_a, R_a))

    # 中性姿态(吸盘平贴取片)：用于遥操时"相对中性偏差"的 ZYX 分解，
    # 锁定左右侧倾(Rx)，控制 heading(yaw)与前后俯仰(pitch)。
    neutral_rotvec = np.zeros(3)
    ans = input("\n记录中性姿态(吸盘平贴)? (y/n，默认 n，中性=单位旋转): ").strip().lower()
    if ans == "y":
        input("  把 Aubo 摆到吸盘平贴的期望姿态，回车...")
        _, R_neutral = _read_aubo_pose(robot)
        neutral_rotvec = Rotation.from_matrix(R_neutral).as_rotvec()
        print(f"  neutral rotvec = {np.degrees(neutral_rotvec)} (度)")

    # 解算：默认两臂基座系朝向一致(R_align=单位阵) + 位置最小二乘解 s/t。
    # 不用姿态对应点解 R_align(手工摆姿态误差大，会把方向带偏)。
    # 若两臂物理摆放有 90° 转角，改 calib_math.IDEAL_R_ALIGN 为 90° 矩阵后重跑。
    R_align = IDEAL_R_ALIGN
    t, s = solve_scale_translation_fixed_r(pairs, R_align)
    print("\n=== 解算结果 (基座系朝向一致) ===")
    print("R_align = 单位阵: leader前→Aubo前, leader左→Aubo左, 上→上")
    print(f"det(R_align) = {np.linalg.det(R_align):.6f} (应为 +1)")
    print(f"scale s = {s:.4f}")
    print(f"t = {t}")

    # 残差(仅位置；姿态遥操时用 neutral_rotvec 偏差法，不在此评估)
    def report(dataset, name):
        pos_err = []
        for p_l, R_l, p_a, R_a in dataset:
            p_pred = t + s * (R_align @ p_l)
            pos_err.append(np.linalg.norm(p_pred - p_a))
        if pos_err:
            print(f"  {name}: 位置残差 mean={np.mean(pos_err) * 1000:.1f}mm "
                  f"max={np.max(pos_err) * 1000:.1f}mm")
    print("--- 残差 ---")
    report(pairs, "拟合点")
    report(val_pairs, "验证点")

    # bounds：实测桌面高度后用户可手改
    calib = {
        "R_align": R_align.tolist(),
        "t": t.tolist(),
        "scale": s,
        "neutral_rotvec": neutral_rotvec.tolist(),
        "bounds_min": [-0.8, -1.2, 0.05],
        "bounds_max": [0.8, 0.0, 0.8],
        "max_ee_step_m": 0.05,
        "motor_names": MOTOR_NAMES,
    }
    CALIB_PATH.write_text(json.dumps(calib, indent=2))
    print(f"\n已写入 {CALIB_PATH}")
    print("注意：bounds_min/max 请按实际桌面与工作空间手改，z_min 应高于桌面 2~3cm。")

    leader.disconnect()
    robot.disconnect()


if __name__ == "__main__":
    main()
