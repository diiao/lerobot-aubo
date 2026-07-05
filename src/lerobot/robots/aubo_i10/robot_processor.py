#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

import math

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    ObservationProcessorStep,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


def _align_rotvec(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """把 target 旋转用与 reference 表示一致的 rotvec 表达出来。

    servoCartesian 对 rotvec 的具体数值表示不鲁棒：同一旋转有多种等价 rotvec
    （绕旋转轴加任意 2π 整数倍，以及反向轴+补角），其中只有部分表示会被
    IK 接受，其余返回 ret=-5。直接用 as_rotvec() 的输出可能在边界处落到
    被拒绝的表示上。

    本函数把 target 旋转重新表达：枚举 reference 旋转的所有等价 rotvec
    表示中与 "target 旋转对应的 as_rotvec 主值" 等价的候选，选出与
    reference 距离最近的一个。由于 reference 通常是上一帧 servoCartesian
    已接受的 rotvec，对齐后的输出大概率也能被接受，保证连续伺服。

    Args:
        target: 想要的旋转的 rotvec（弧度），任意等价表示均可。
        reference: 参考 rotvec（通常是当前 TCP 的 rotvec，已知可被接受）。

    Returns:
        与 target 旋转等价、且与 reference 表示最接近的 rotvec。
    """
    target = np.asarray(target, dtype=float)
    reference = np.asarray(reference, dtype=float)

    R_target = Rotation.from_rotvec(target)
    R_ref = Rotation.from_rotvec(reference)

    # 目标旋转的"主值" rotvec（norm ≤ π 的标准表示）
    primary = R_target.as_rotvec()

    # 候选1: 目标主值本身
    # 候选2: 目标主值绕自身轴 +2π（等价表示）
    # 候选3: 目标主值绕自身轴 -2π（等价表示）
    candidates = [primary]
    norm = float(np.linalg.norm(primary))
    if norm > 1e-9:
        axis = primary / norm
        candidates.append(primary + 2.0 * np.pi * axis)
        candidates.append(primary - 2.0 * np.pi * axis)

    best = min(candidates, key=lambda c: float(np.linalg.norm(np.asarray(c) - reference)))
    return np.asarray(best, dtype=float)


@ProcessorStepRegistry.register("phone_ee_to_aubo_ee")
@dataclass
class PhoneEEToAuboEE(RobotActionProcessorStep):
    """
    位置速度控制 + 末端位姿计算（支持操纵杆速度模式和弹性位置模式）。

    velocity_mode=True（操纵杆/速度模式，推荐遥操）：
    - 手机相对按下瞬间的位移 → 末端速度（每帧移动量）
    - 手机回到按下位置(中心) → 速度=0 → 机器人停止
    - 手机不动 = 机器人不动（真正的操纵杆行为）
    - 小位移 = 慢速精确，大位移 = 高速快移（鼠标加速，position_acceleration控制曲线）
    - 松手/重按 → 新原点（按下时从当前TCP开始积分）

    velocity_mode=False（弹性/位置模式）：
    - 手机相对按下瞬间的位移 → 末端相对参考TCP的绝对偏移（弹簧连接）

    Attributes:
        end_effector_step_sizes: 速度模式下每单位手机位移的每帧移动量(m)；越大越快。
        velocity_mode: True=速度控制(操纵杆)，False=位置控制(弹性)。
        position_acceleration: 非线性加速系数（鼠标加速）；0=线性，>0=大位移额外加速。
            output = disp * scale * (1 + acceleration * |disp|)
        use_latched_reference: velocity_mode=False时才有效，True=按下时锁存参考。
    """

    end_effector_step_sizes: dict[str, float] = field(default_factory=lambda: {"x": 0.05, "y": 0.05, "z": 0.05})
    velocity_mode: bool = True
    position_acceleration: float = 4.0
    use_latched_reference: bool = True
    position_smoothing: float = 0.0

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)
    _smoothed_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _vel_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def _vel_step(self, v: float, scale: float) -> float:
        """鼠标加速映射：小位移精确，大位移快速。
        output = v * scale * (1 + acceleration * |v|)
        acceleration=0 → 纯线性；acceleration=4 → 10cm位移时速度是线性的1.4倍。
        """
        return v * scale * (1.0 + self.position_acceleration * abs(v))

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        
        if observation is None:
            raise ValueError("Observation is required for computing absolute EE pose")
        
        current_ee_pose = np.array([
            float(observation.get("ee.x", 0.0)),
            float(observation.get("ee.y", 0.0)),
            float(observation.get("ee.z", 0.0)),
            float(observation.get("ee.wx", 0.0)),
            float(observation.get("ee.wy", 0.0)),
            float(observation.get("ee.wz", 0.0)),
        ])
        
        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        rotation_handled_upstream = not any(
            k in action for k in ("target_wx", "target_wy", "target_wz")
        )
        twx = float(action.pop("target_wx", 0.0))
        twy = float(action.pop("target_wy", 0.0))
        twz = float(action.pop("target_wz", 0.0))
        gripper_vel = float(action.pop("gripper_vel"))

        desired = current_ee_pose.copy()

        if self.velocity_mode:
            # ── 速度/操纵杆模式 ──────────────────────────────────────────
            # 按下上升沿：从当前 TCP 重新开始积分（每次按下 = 新原点）
            if enabled:
                if not self._prev_enabled or self._vel_pos is None:
                    self._vel_pos = current_ee_pose[:3].copy()

                # 鼠标加速：小位移精确，大位移快速
                dx = self._vel_step(tx, self.end_effector_step_sizes["x"])
                dy = self._vel_step(ty, self.end_effector_step_sizes["y"])
                dz = self._vel_step(tz, self.end_effector_step_sizes["z"])
                self._vel_pos = self._vel_pos + np.array([dx, dy, dz], dtype=float)

                # Anti-windup：防止积分器在边界处持续累积。
                # 若 _vel_pos 距当前实际 TCP 超过一帧最大步长，将其拉回。
                # 保证松手后重按仍从真实当前位置出发，不会卡在边界死区。
                overshoot = self._vel_pos - current_ee_pose[:3]
                ov_norm = float(np.linalg.norm(overshoot))
                max_ahead = self.end_effector_step_sizes.get("x", 0.05) * 8  # ~8帧缓冲
                if ov_norm > max_ahead:
                    self._vel_pos = current_ee_pose[:3] + overshoot * (max_ahead / ov_norm)

                desired[:3] = self._vel_pos
                self._command_when_disabled = desired.copy()
            else:
                # 未按下：保持上次位置（机器人不动）
                if self._command_when_disabled is None:
                    self._command_when_disabled = current_ee_pose.copy()
                desired = self._command_when_disabled.copy()
        else:
            # ── 弹性/位置模式 ────────────────────────────────────────────
            if enabled:
                ref = current_ee_pose
                if self.use_latched_reference:
                    if not self._prev_enabled or self.reference_ee_pose is None:
                        self.reference_ee_pose = current_ee_pose.copy()
                    ref = self.reference_ee_pose

                delta_p = np.array([
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ], dtype=float)

                desired = ref.copy()
                desired[:3] = ref[:3] + delta_p

                a = float(np.clip(self.position_smoothing, 0.0, 0.99))
                if a > 0.0:
                    if self._smoothed_pos is None or not self._prev_enabled:
                        self._smoothed_pos = desired[:3].copy()
                    else:
                        self._smoothed_pos = a * self._smoothed_pos + (1.0 - a) * desired[:3]
                    desired[:3] = self._smoothed_pos

                if not rotation_handled_upstream:
                    desired[3] = ref[3] + twx * 0.1
                    desired[4] = ref[4] + twy * 0.1
                    desired[5] = ref[5] + twz * 0.1

                self._command_when_disabled = desired.copy()
            else:
                if self._command_when_disabled is None:
                    self._command_when_disabled = current_ee_pose.copy()
                desired = self._command_when_disabled.copy()

        action["ee.x"] = float(desired[0])
        action["ee.y"] = float(desired[1])
        action["ee.z"] = float(desired[2])
        # 旋转：若上游已处理则不覆盖（保留上游写入的 ee.wx/wy/wz）
        if not rotation_handled_upstream:
            action["ee.wx"] = float(desired[3])
            action["ee.wy"] = float(desired[4])
            action["ee.wz"] = float(desired[5])
        action["ee.gripper_vel"] = gripper_vel
        
        self._prev_enabled = enabled
        return action

    def reset(self):
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None
        self._smoothed_pos = None
        self._vel_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("aubo_gripper_velocity_to_position")
@dataclass
class AuboGripperVelocityToPosition(RobotActionProcessorStep):
    """
    Converts gripper velocity command to gripper position for AuboI10.

    Uses threshold-based three-state switching (matching teleoperate.py):
    - gripper_vel > 0.5  → fully open (100.0)  — A button pressed
    - gripper_vel < -0.5 → fully closed (0.0)   — B button pressed
    - else               → neutral (50.0)       — no button pressed

    Attributes:
        open_threshold: Velocity threshold to open gripper.
        close_threshold: Velocity threshold to close gripper.
        clip_min: Minimum gripper position (closed).
        clip_max: Maximum gripper position (open).
        neutral_pos: Gripper position when neither open nor close is commanded.
    """

    open_threshold: float = 0.5
    close_threshold: float = -0.5
    clip_min: float = 0.0
    clip_max: float = 100.0
    neutral_pos: float = 50.0

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("Observation is required for gripper position computation")

        gripper_vel = action.pop("ee.gripper_vel")

        # Three-state gripper control matching teleoperate.py logic:
        # A pressed (vel > threshold)  → open  (100.0)
        # B pressed (vel < -threshold) → close (0.0)
        # Neither pressed              → neutral / hold (50.0)
        if gripper_vel > self.open_threshold:
            gripper_pos = self.clip_max      # Open
        elif gripper_vel < self.close_threshold:
            gripper_pos = self.clip_min      # Close
        else:
            gripper_pos = self.neutral_pos   # Neutral: neither open nor close

        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )
        return features


@ProcessorStepRegistry.register("phone_pos_to_ee_delta")
@dataclass
class PhonePosToEEDelta(RobotActionProcessorStep):
    """
    把手机位置信号 (target_x/y/z) 转成末端位置 delta (ee.x/y/z)。

    手机经 MapPhoneActionToRobotAction 输出的 target_x/y/z 是相对标定位姿的
    位置信号（非绝对机器人坐标）。本步骤乘以 step_size 作为末端位置增量，
    配合 delta 伺服通路 (_send_ee_action_servo) 使用：每帧 target=current+delta。

    须放在 AuboLockVerticalYaw 之后（后者已消费 target_w* 并写入 ee.wx/wy/wz）。

    Attributes:
        step_sizes: x/y/z 方向的 delta 缩放系数（米/单位信号）。
        max_step: 单帧位置 delta 限幅（米），防手机甩动导致末端冲过头。
    """

    step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 0.05, "y": 0.05, "z": 0.05}
    )
    max_step: float = 0.05

    def action(self, action: RobotAction) -> RobotAction:
        tx = float(action.pop("target_x", 0.0))
        ty = float(action.pop("target_y", 0.0))
        tz = float(action.pop("target_z", 0.0))

        dx = float(np.clip(tx * self.step_sizes["x"], -self.max_step, self.max_step))
        dy = float(np.clip(ty * self.step_sizes["y"], -self.max_step, self.max_step))
        dz = float(np.clip(tz * self.step_sizes["z"], -self.max_step, self.max_step))

        action["ee.x"] = dx
        action["ee.y"] = dy
        action["ee.z"] = dz

        # 顺带把 gripper_vel 转成 ee.gripper_vel，供下游 AuboGripperVelocityToPosition 使用
        if "gripper_vel" in action:
            action["ee.gripper_vel"] = float(action.pop("gripper_vel"))
        return action

    def reset(self):
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["target_x", "target_y", "target_z"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)
        for feat in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("aubo_lock_vertical_yaw")
@dataclass
class AuboLockVerticalYaw(RobotActionProcessorStep):
    """
    位置跟随 + J6 直控偏航：把手机的左右旋转映射为 Aubo 最后一个轴(J6)的转动，
    末端位置跟手机移动，pitch/roll 锁死（末端保持按下瞬间的竖直向下姿态）。

    机制（配合机器人端 _send_position_j6yaw）：
    - 按下 hold-to-move 瞬间锁存参考姿态 R_ref（=当前 TCP 姿态，用户已摆成竖直向下）
      和参考 J6（=当前实际 J6）。
    - 每帧输出：固定姿态 ee.wx/ee.wy/ee.wz = R_ref（供 IK 保持竖直，pitch/roll 不动），
      以及 J6 绝对目标 ee.j6_target（=参考 J6 + 手机左右旋转累计，弧度），置 ee_mode="abs_j6yaw"。
    - 机器人端对 (位置 + R_ref) 做逆解得 J1..J5，再把 J6 设为 ee.j6_target。
      → 末端只绕法兰轴（工具竖直时即竖直方向）旋转，绝不俯仰/翻滚。

    为什么直控 J6 而非笛卡尔姿态伺服：末端竖直时绕竖直转本应只动 J6，但笛卡尔
    姿态伺服里 IK 会把偏航分配到 J4/J5，表现为绕基座 Y 俯仰（实测现象）。直控 J6
    彻底规避该问题，也无 rotvec 奇异点烦恼。

    偏航信号：从手机旋转矩阵提取绕 yaw_up_axis 轴的航向角(heading)，逐帧增量累加
    （对手机同时倾斜鲁棒）。方向反了改 yaw_gain 符号；幅度不够调大 yaw_gain。

    本步骤须放在 MapPhoneActionToRobotAction 之后、PhoneEEToAuboEE 之前：
    - 消费 target_wx/target_wy/target_wz（手机旋转命令），阻止下游处理旋转。
    - 输出固定姿态 ee.wx/ee.wy/ee.wz + J6 绝对目标 ee.j6_target，并置 ee_mode="abs_j6yaw"。
    - 位置 (target_x/y/z) 不动，留给 PhoneEEToAuboEE 处理成绝对位置。

    Attributes:
        yaw_gain: 手机航向 → J6 偏航的系数；正负决定旋转方向，绝对值决定幅度。
        yaw_smoothing: 偏航一阶低通系数 [0,1)，0=关闭；越大越丝滑但越滞后。
        yaw_up_axis: 手机标定坐标系里"竖直方向"对应的轴 "x"/"y"/"z"（默认 "z"）。
            手机绕该轴的转动被提取为偏航。若绕竖直转手机末端不转/转错，看日志
            [yaw] 里 hx/hy/hz 哪个随手势变化最大，就把本参数设成对应轴。
        debug_log: 是否每帧输出 yaw 诊断日志（三个候选航向 + 当前偏航角）。
        max_wz_step / correct_gain / max_correct_step: 兼容旧调用签名，现已不使用。
    """

    yaw_gain: float = 1.0
    yaw_smoothing: float = 0.0
    yaw_up_axis: str = "z"
    debug_log: bool = False
    # 以下参数仅为兼容旧调用签名（历史遗留），现已不使用：
    max_wz_step: float = 0.1
    correct_gain: float = 0.0
    max_correct_step: float = 0.05
    # 运行时状态(非 init)：
    _enabled_prev: bool = field(default=False, init=False, repr=False)
    _heading_prev: float | None = field(default=None, init=False, repr=False)
    _yaw_accum: float = field(default=0.0, init=False, repr=False)
    _ref_orient: np.ndarray | None = field(default=None, init=False, repr=False)
    _ref_j6: float | None = field(default=None, init=False, repr=False)
    _yaw_filt: float = field(default=0.0, init=False, repr=False)

    @staticmethod
    def _headings(phone_R: np.ndarray) -> tuple[float, float, float]:
        """从手机旋转矩阵提取绕 x/y/z 三个轴的航向角（弧度）。

        对"主要绕某一个轴"的旋转，用两参 atan2 提取该轴转角，对其他轴的
        小幅倾斜鲁棒（比单取 rotvec 分量更稳）。返回 (hx, hy, hz)。
        """
        hx = math.atan2(phone_R[2, 1], phone_R[1, 1])  # 绕 X
        hy = math.atan2(phone_R[0, 2], phone_R[2, 2])  # 绕 Y
        hz = math.atan2(phone_R[1, 0], phone_R[0, 0])  # 绕 Z
        return hx, hy, hz


    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for AuboLockVerticalYaw")

        # 消费手机的旋转命令（阻止下游 PhoneEEToAuboEE 处理旋转）。
        # 逆推手机原始 rotvec：MapPhoneActionToRobotAction 里
        #   target_wx=-rotvec[1], target_wy=-rotvec[0], target_wz=rotvec[2]
        # 故 phone_rotvec = [-target_wy, -target_wx, target_wz]。
        twx = float(action.pop("target_wx", 0.0))
        twy = float(action.pop("target_wy", 0.0))
        twz = float(action.pop("target_wz", 0.0))
        phone_rotvec = np.array([-twy, -twx, twz], dtype=float)
        phone_R = Rotation.from_rotvec(phone_rotvec).as_matrix()
        hx, hy, hz = self._headings(phone_R)
        heading = {"x": hx, "y": hy, "z": hz}.get(self.yaw_up_axis, hz)

        cur_rotvec = np.array(
            [
                float(observation.get("ee.wx", 0.0)),
                float(observation.get("ee.wy", 0.0)),
                float(observation.get("ee.wz", 0.0)),
            ]
        )

        enabled = bool(action.get("enabled", False))
        # 按下 hold-to-move 上升沿：锁存该瞬间的固定参考姿态与参考 J6(当前实际 J6)，
        # 偏航从 0 开始累计。这样松手→再次按下时以当前 J6 为新原点，不会回弹/跳变。
        if enabled and not self._enabled_prev:
            self._heading_prev = heading
            self._yaw_accum = 0.0
            self._ref_orient = cur_rotvec.copy()
            self._ref_j6 = math.radians(float(observation.get("J6", 0.0)))
            self._yaw_filt = 0.0
        self._enabled_prev = enabled

        # 兜底：尚无参考（未曾按下）时用当前 TCP 姿态 / 当前 J6
        if self._ref_orient is None:
            self._ref_orient = cur_rotvec.copy()
        if self._ref_j6 is None:
            self._ref_j6 = math.radians(float(observation.get("J6", 0.0)))

        if enabled:
            # 手机航向连续累计：heading 经 atan2 会在 ±π 处跳变。逐帧对增量做 [-π,π]
            # 环绕再累加，得到连续无跳变的偏航，允许超过 180° 旋转且边界不反转 360°。
            if self._heading_prev is None:
                self._heading_prev = heading
            d = (heading - self._heading_prev + math.pi) % (2 * math.pi) - math.pi
            self._yaw_accum += d
            self._heading_prev = heading
            phi = self._yaw_accum * self.yaw_gain
            # 一阶低通丝滑（可选）
            a = float(np.clip(self.yaw_smoothing, 0.0, 0.99))
            self._yaw_filt = a * self._yaw_filt + (1.0 - a) * phi
        # 未按下：保持上一次偏航角（松手不乱转），self._yaw_filt 不变
        phi_use = self._yaw_filt

        # J6 绝对目标 = 按下瞬间的 J6 + 手机偏航累计（弧度）。以固定 ref_j6 为基准，
        # 分支稳定、无 ±180° 跳变，再次按下也从当前 J6 平滑续转。
        j6_target = self._ref_j6 + phi_use

        # yaw 诊断日志：手机绕竖直转手机时，看 hx/hy/hz 哪个变化最大 → 设 yaw_up_axis。
        if self.debug_log:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "[yaw] enabled=%d up=%s hx=%.1f hy=%.1f hz=%.1f | yaw=%.1f J6_target=%.1f (deg)",
                int(enabled), self.yaw_up_axis,
                math.degrees(hx), math.degrees(hy), math.degrees(hz),
                math.degrees(phi_use), math.degrees(j6_target),
            )

        # 输出：固定参考姿态 (供 IK 保持竖直/pitch/roll 锁死) + J6 绝对目标，置 abs_j6yaw 模式
        action["ee.wx"] = float(self._ref_orient[0])
        action["ee.wy"] = float(self._ref_orient[1])
        action["ee.wz"] = float(self._ref_orient[2])
        action["ee.j6_target"] = float(j6_target)
        action["ee_mode"] = "abs_j6yaw"
        return action

    def reset(self):
        self._enabled_prev = False
        self._heading_prev = None
        self._yaw_accum = 0.0
        self._ref_orient = None
        self._ref_j6 = None
        self._yaw_filt = 0.0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 把 J6 绝对目标角加入 dataset features，以便录制时保存偏航信息
        features[PipelineFeatureType.ACTION]["ee.j6_target"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )
        return features


@ProcessorStepRegistry.register("aubo_ee_bounds_and_safety")
@dataclass
class AuboEEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.
    
    Ensures that the target end-effector pose remains within a safe operational workspace.
    
    Attributes:
        end_effector_bounds: Dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: Maximum allowed change in position (meters) between steps.
        _last_pos: Internal state storing the last commanded position.
    """
    
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-0.8, -1.2, 0.0], "max": [0.8, 0.0, 0.8]}
    )
    max_ee_step_m: float = 0.05
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)

        self._last_pos = pos

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action

    def reset(self):
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("aubo_ee_to_ee_delta")
@dataclass
class AuboEEToEEDelta(RobotActionProcessorStep):
    """
    Converts absolute end-effector pose to delta commands for robot control.
    
    This is used during replay/evaluation to convert the absolute pose from the policy
    to the delta format expected by the robot's send_action method.
    
    The robot's _send_ee_action_servo expects delta values that it adds to current pose.
    """
    
    convert_rotation: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("Observation is required for computing delta")

        current_ee_pose = np.array([
            float(observation.get("ee.x", 0.0)),
            float(observation.get("ee.y", 0.0)),
            float(observation.get("ee.z", 0.0)),
            float(observation.get("ee.wx", 0.0)),
            float(observation.get("ee.wy", 0.0)),
            float(observation.get("ee.wz", 0.0)),
        ])

        target_x = float(action.pop("ee.x"))
        target_y = float(action.pop("ee.y"))
        target_z = float(action.pop("ee.z"))
        target_wx = float(action.pop("ee.wx"))
        target_wy = float(action.pop("ee.wy"))
        target_wz = float(action.pop("ee.wz"))
        gripper_pos = action.pop("ee.gripper_pos", 50.0)

        action["ee.x"] = target_x - current_ee_pose[0]
        action["ee.y"] = target_y - current_ee_pose[1]
        action["ee.z"] = target_z - current_ee_pose[2]
        if self.convert_rotation:
            # 旋转也是绝对 → delta
            action["ee.wx"] = target_wx - current_ee_pose[3]
            action["ee.wy"] = target_wy - current_ee_pose[4]
            action["ee.wz"] = target_wz - current_ee_pose[5]
        else:
            # 旋转已由上游 (如 AuboLockVerticalYaw) 给出 delta，原样保留
            action["ee.wx"] = target_wx
            action["ee.wy"] = target_wy
            action["ee.wz"] = target_wz
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
