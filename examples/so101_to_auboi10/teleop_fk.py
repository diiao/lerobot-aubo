#!/usr/bin/env python
"""SO101 leader → Aubo I10 末端到末端 FK 遥操。

前提：两臂基座系朝向一致（X前/Y左/Z上 对齐，R_align=单位阵），仅靠 scale=3
做尺寸缩放（SO101 小臂 → Aubo I10 大臂）。若两臂物理摆放有 90° 转角，
需改 calibration.json 的 R_align 并重跑 calibrate.py。

链路:
  leader.get_action() (5 关节度 + gripper.pos)
    → forward_kinematics → leader 末端位姿 (leader 基座系)
    → 位置: delta_leader 经 scale 缩放(R_align=I 故方向不变) → Aubo 系位移增量
    → 姿态: R_aligned = R_leader(基座系一致故不旋转)，再对"相对中性姿态的偏差"
      做 ZYX 分解: 控制 yaw(夹爪自转)+pitch(前后俯仰)，roll(左右侧倾)固定为正值
    → R_target = Rz(yaw)·Ry(pitch)·Rx(roll)·R_neutral  → rotvec
    → 单帧步长 clip + 边界裁剪
    → follower.send_action(ee_mode="absolute")  → servoCartesian 纯时间模式(acc=vel=0,t=0.1)
  gripper.pos → ee.gripper_pos → 复用软爪 IO 通路驱动吸盘

前置:
  - 已生成 calibration.json (跑 calibrate.py)
  - placo 已装
用法:
  PYTHONPATH=src python examples/so101_to_auboi10/teleop_fk.py
"""
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Config, AuboI10Robot
from lerobot.robots.aubo_i10.robot_processor import AuboEEBoundsAndSafety
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

sys.path.insert(0, str(Path(__file__).parent))
from calib_math import extract_target_rotmat, step_toward_orientation, rotmat_to_rotvec  # noqa: E402

URDF_PATH = Path(__file__).parent / "so101_new_calib.urdf"
CALIB_PATH = Path(__file__).parent / "calibration.json"
TARGET_FRAME = "gripper_frame_link"
FPS = 50  # 喂点频率。高频采样让小动作更跟手（降低群延迟）；官方例程用 20Hz，这里为灵敏度提到 50

# 姿态跟随 leader：yaw(夹爪自转)+pitch(前后俯仰) 跟随，roll(左右侧倾)固定为正值。
# 设 True 则姿态整体锁死在启动 TCP，仅用于隔离验证位置映射。
# 当前阶段：先只测位置跟随，姿态锁定，位置 OK 后改 False 再调姿态。
POSITION_ONLY = True

# 左右侧倾(Rx)固定值：不跟随 leader，恒为该正值。负=向一侧倾，正=向另一侧倾。
# 跑起来若方向反了或角度不合适，调这里（弧度）。
FIXED_ROLL_RAD = math.radians(15.0)

# servoCartesian 速度控制（对齐官方 example_servo_cartesian.py 的纯时间模式约定）：
#   官方: mc.servoCartesian(p, 0.0, 0.0, 0.1, 0.0, 0.0)  即 (pose, acc=0, vel=0, time, blend=0, max_radius=0)
# acc=vel=0 时为"纯时间模式"：臂在 `time` 秒内插值到该路点。速度上限 = 单帧步长 / time。
# 速度上限 = MAX_POS_STEP_M / SERVO_TIME = 0.002/0.05 = 0.04 m/s。
# 灵敏度(跟手度)由采样率 FPS + SERVO_TIME 的群延迟决定：FPS 越高、SERVO_TIME 越小 → 小动作响应越快。
# 想更灵敏就同比例缩小 SERVO_TIME 和 MAX_POS_STEP_M(保持比值=速度不变)，并提高 FPS。
# 想更慢就同比例放大两者(保持比值不变)，或单独放大 SERVO_TIME。
LINE_VELOCITY = 0.0         # m/s：纯时间模式置 0（对齐官方）
LINE_ACCELERATION = 0.0     # m/s²：纯时间模式置 0
SERVO_TIME = 0.05           # s：每路点插值时长(主限速旋钮之一)。0.05=较灵敏；想更慢调大
# 单帧最大位置步长(各向同性，与 SERVO_TIME 共同决定速度上限)。
# 0.0025/0.05=0.05m/s，比 0.002 略快(大范围移动稍利索)又不过猛。
MAX_POS_STEP_M = 0.0025
MAX_ORI_STEP_RAD = math.radians(1.0)  # 单帧最大旋转步长：1°/帧

# 下降方向(Z负)增益：对 leader 下降位移分量额外乘此系数(<1 让下降更保守，防戳桌锁死)。
# 这是温和手段：只缩小下降输入，不破坏步长 clamp 的各向同性，避免轨迹扭曲顿挫。
DOWN_GAIN = 0.6             # 0.6=下降速度为水平/上升的 60%；想更安全调小(如0.4)，1.0=关闭

# 启动待机门控：启动后 leader 必须先产生足够大的累计位移(ARM)才解锁跟随。
# 解锁前大臂完全冻结在启动 TCP，避免 leader 初始位置对应的大臂目标点处有障碍物直接碰撞。
# 解锁瞬间用软启动增益从 0 渐升到 1(下面 SOFTSTART_FRAMES)，防首帧大跳。
ARM_UNLOCK_M = 0.03         # leader 累计位移阈值(米，leader 基座系)，超过则解锁跟随
SOFTSTART_FRAMES = 100      # 解锁后软启动帧数(约2秒@50Hz)，增益 0→1 平滑接手

# 非线性增益：小幅输入低增益(精细微调)、大幅输入高增益(快速大范围移动)。
# 单帧 leader 位移 |d|(米) 的增益 = FINE_GAIN_MIN + (FINE_GAIN_MAX-FINE_GAIN_MIN) * |d|/FINE Knee，
# 在 |d| >= FINE_KNEE 后饱和到 FINE_GAIN_MAX。曲线在小幅段斜率低(精细)、大幅段斜率高(快)。
# 即对 delta 整体乘一个随其模长递增的系数(再乘 scale 已含在 p_leader_mapped 里)。
# 想更精细就调小 FINE_GAIN_MIN(如0.3)；想大幅更快就调大 FINE_GAIN_MAX(如1.5)。
FINE_GAIN_MIN = 0.5         # 极小位移时的增益(精细微调)
FINE_GAIN_MAX = 1.2         # 大位移时的增益(快速移动)
FINE_KNEE = 0.01            # 增益饱和阈值(米)：|d|>=此值时增益到 FINE_GAIN_MAX

# 死区：设 0 关闭。连续喂点是丝滑关键，死区会跳过帧导致 servo 队列空转顿挫。
DEADZONE_POS_M = 0.0
DEADZONE_ROT_RAD = 0.0
DEADZONE_GRIPPER = 1.0     # 0-100 量纲（夹爪仍保留死区，避免抖动误触发吸盘）


def load_calibration():
    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"找不到 {CALIB_PATH}，请先跑 calibrate.py")
    calib = json.loads(CALIB_PATH.read_text())
    calib["R_align"] = np.array(calib["R_align"])
    calib["t"] = np.array(calib["t"])
    calib["R_neutral"] = Rotation.from_rotvec(np.array(calib["neutral_rotvec"])).as_matrix()
    return calib


def leader_to_aubo_target(action, kinematics, motor_names, calib, state):
    """leader 关节 → (p_leader_mapped, R_target, gripper_pos, p_leader_raw)。

    - p_leader_mapped: leader FK 位置经 scale·R_align 映射到 Aubo 系(未加基准偏移)。
      增量跟随时 main 用它减去启动基准得相对位移。
    - R_target: 目标姿态矩阵(相对中性偏差 ZYX，锁侧倾)。
    - p_leader_raw: leader FK 原始位置(leader 基座系)，用于记录基准。
    """
    q = np.array([action[f"{m}.pos"] for m in motor_names], dtype=float)
    T = kinematics.forward_kinematics(q)
    p_leader = T[:3, 3]
    R_leader = Rotation.from_matrix(T[:3, :3]).as_matrix()

    # 位置：方向对齐 + 缩放（不加 t 偏移；增量跟随由 main 处理基准）
    p_leader_mapped = calib["scale"] * (calib["R_align"] @ p_leader)
    # 姿态：表达到 Aubo 基座系，再对中性偏差做 ZYX(锁侧倾)
    R_aligned = calib["R_align"] @ R_leader
    R_target, yaw, pitch = extract_target_rotmat(
        R_aligned, calib["R_neutral"], state.get("last_yaw"), state.get("last_pitch"),
        roll_rad=FIXED_ROLL_RAD,
    )
    state["last_yaw"], state["last_pitch"] = yaw, pitch

    gripper_pos = float(action["gripper.pos"])
    return p_leader_mapped, R_target, gripper_pos, p_leader


def main():
    init_logging()
    logging.getLogger().setLevel(logging.INFO)

    calib = load_calibration()
    motor_names = calib["motor_names"]
    kinematics = RobotKinematics(str(URDF_PATH), TARGET_FRAME, joint_names=motor_names)
    logging.info(f"kinematics.joint_names: {kinematics.joint_names}")
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

    # 速度：对齐官方 example_servo_cartesian.py 的纯时间模式 (acc=0, vel=0, time=SERVO_TIME)
    follower.line_velocity = LINE_VELOCITY        # 0.0
    follower.line_acceleration = LINE_ACCELERATION  # 0.0
    follower.servo_time = SERVO_TIME              # 0.1s，主限速旋钮

    # 首帧安全：用当前 TCP 初始化 safety._last_pos，防第一帧大跳
    obs = follower.get_observation()
    safety._last_pos = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=float)
    # 记录启动姿态：POSITION_ONLY 模式下姿态锁定为此，不跟随 leader
    locked_rotvec = np.array([obs["ee.wx"], obs["ee.wy"], obs["ee.wz"]], dtype=float)
    logging.info(f"当前 TCP: {safety._last_pos}")
    if POSITION_ONLY:
        logging.info(f"POSITION_ONLY=True：姿态锁定为 {locked_rotvec}，只测位置跟随")

    # 手机式增量跟随：维护"当前目标点"，每帧加上 leader 的位移增量。
    # 停手(leader 位移=0) → 目标不变 → servo 立即停，无前瞻追赶摆动。
    # 区别于"绝对位置映射"(目标=base+scale·(leader_now-leader_base))，
    # 后者停手后 servoCartesian 的前瞻平滑仍会追之前的高频目标序列 → 摆动。
    state = {"last_yaw": None, "last_pitch": None}
    action0 = leader.get_action()
    p_leader_mapped_prev, _, _, _ = leader_to_aubo_target(action0, kinematics, motor_names, calib, state)
    p_leader_mapped_prev = p_leader_mapped_prev.copy()
    # 启动时 leader 末端位置(leader 基座系)，用于累计"leader 自启动后移动了多远"以判断是否解锁
    leader_start_pos = p_leader_mapped_prev.copy()
    # 当前目标点初始化为 Aubo 实际 TCP（启动待机期间恒等于此，大臂不动）
    cur_target_pos = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=float)
    logging.info(f"增量跟随已初始化：目标={cur_target_pos}")
    logging.info(f"启动待机中：请缓慢移动小臂累计>{ARM_UNLOCK_M*100:.0f}cm 后大臂才开始跟随")

    last_target = None  # (p_target, R_target_rotvec, gripper) 用于死区
    # 上次实际下发的位置（用于硬性单帧步长 clamp，防漂移、防首帧大跳）
    p_cmd_prev = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=float)
    frame_idx = 0
    unlocked = False            # 是否已解锁跟随(leader 首次产生足够位移后置 True)
    accum_leader_move = 0.0     # leader 自启动后累计位移(米)
    softstart_idx = 0           # 解锁后的软启动帧计数

    print(f"开始 FK 末端遥操 (FPS={FPS}, POSITION_ONLY={POSITION_ONLY})。Ctrl-C 退出。")
    try:
        while True:
            t0 = time.perf_counter()
            action = leader.get_action()
            p_leader_mapped, R_target, gripper_pos, _ = leader_to_aubo_target(
                action, kinematics, motor_names, calib, state
            )
            # 手机式：目标 += leader 本帧位移增量(经 scale·R_align)
            delta = p_leader_mapped - p_leader_mapped_prev
            p_leader_mapped_prev = p_leader_mapped.copy()
            # 滤 leader 静止时的关节读数抖动，防目标累积漂移
            if np.linalg.norm(delta) < 1e-4:  # 0.1mm
                delta = np.zeros(3)

            # 启动待机门控：leader 必须先累计移动 > ARM_UNLOCK_M 才解锁跟随。
            # 解锁前大臂完全冻结在启动 TCP，避免 leader 初始位置对应的目标点处有障碍物直接碰撞。
            if not unlocked:
                # 累计位移 = 当前 leader 位置距起点的欧氏距离(稳健，不受抖动反复增减影响)
                accum_leader_move = float(np.linalg.norm(p_leader_mapped - leader_start_pos))
                if accum_leader_move > ARM_UNLOCK_M:
                    unlocked = True
                    logging.info(f"跟随已解锁(leader 累计移动 {accum_leader_move*100:.1f}cm)，开始软启动")
                delta = np.zeros(3)  # 解锁前不产生任何目标位移
                gain = 0.0
            else:
                # 解锁后软启动：增益从 0 渐升到 1，防首帧大跳
                gain = min(1.0, softstart_idx / SOFTSTART_FRAMES) if SOFTSTART_FRAMES > 0 else 1.0
                softstart_idx += 1
                # 非线性细粒度增益：小幅位移低增益(精细)、大幅位移高增益(快)
                d_norm = float(np.linalg.norm(delta))
                if d_norm > 1e-9:
                    fine = FINE_GAIN_MIN + (FINE_GAIN_MAX - FINE_GAIN_MIN) * min(1.0, d_norm / FINE_KNEE)
                    gain = gain * fine
            # 下降方向(Z负)额外降增益，防小臂误下导致大臂戳桌锁死(温和手段，不破坏 clamp 协调)
            if delta[2] < 0.0:
                delta = delta.copy()
                delta[2] *= DOWN_GAIN
            cur_target_pos = cur_target_pos + delta * gain
            p_aubo = cur_target_pos
            R_target_rv = rotmat_to_rotvec(R_target)

            # 死区：目标相对上次目标变化太小则跳过（含 gripper）
            if last_target is not None:
                dp = float(np.linalg.norm(p_aubo - last_target[0]))
                dr = float(np.linalg.norm(R_target_rv - last_target[1]))
                dg = abs(gripper_pos - last_target[2])
                if dp < DEADZONE_POS_M and dr < DEADZONE_ROT_RAD and dg < DEADZONE_GRIPPER:
                    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
                    frame_idx += 1
                    continue

            # 位置：先边界裁剪，再做"相对上次下发位置"的各向同性单帧步长硬 clamp。
            # 每帧位移不超过 MAX_POS_STEP_M，且以 p_cmd_prev 为基准，clamp 不产生漂移。
            p_desired = np.clip(p_aubo, calib["bounds_min"], calib["bounds_max"])
            step = p_desired - p_cmd_prev
            step_norm = float(np.linalg.norm(step))
            if step_norm > MAX_POS_STEP_M and step_norm > 1e-9:
                step = step * (MAX_POS_STEP_M / step_norm)
            p_cmd = p_cmd_prev + step
            p_cmd_prev = p_cmd.copy()

            # 姿态：POSITION_ONLY 锁定；否则朝目标步进(防 rotvec 跳变)。
            # 姿态步进仍需读实际 TCP 做基准。
            if POSITION_ONLY:
                rotvec_step = locked_rotvec
            else:
                obs_now = follower.get_observation()
                cur_rotvec = np.array([obs_now["ee.wx"], obs_now["ee.wy"], obs_now["ee.wz"]])
                R_current = Rotation.from_rotvec(cur_rotvec).as_matrix()
                rotvec_step = step_toward_orientation(
                    R_current, R_target, MAX_ORI_STEP_RAD, ref_rotvec=cur_rotvec
                )

            cmd = {
                "ee.x": float(p_cmd[0]),
                "ee.y": float(p_cmd[1]),
                "ee.z": float(p_cmd[2]),
                "ee.wx": float(rotvec_step[0]),
                "ee.wy": float(rotvec_step[1]),
                "ee.wz": float(rotvec_step[2]),
                "ee.gripper_pos": gripper_pos,
                "ee_mode": "absolute",
            }
            follower.send_action(cmd)
            last_target = (p_aubo.copy(), R_target_rv, gripper_pos)
            frame_idx += 1

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n遥操停止")
    finally:
        # 关吸盘、关伺服、断开
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
