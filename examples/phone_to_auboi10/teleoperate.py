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

"""手机遥操 Aubo I10（位置跟随 + J6 直控偏航）。

控制通路与 record.py 完全一致：
    phone.get_action()
      → MapPhoneActionToRobotAction        # 手机位姿 → target_x/y/z/wx/wy/wz + gripper_vel + enabled
      → AuboLockVerticalYaw                 # 消费 target_w*，输出固定竖直姿态 + J6 绝对目标(ee.j6_target)；置 ee_mode=abs_j6yaw
      → PhoneEEToAuboEE                     # target_x/y/z + 按下瞬间TCP → 绝对 ee.x/y/z (位置)
      → AuboEEBoundsAndSafety               # 位置裁剪 + 单帧步长限制 (控速)
      → AuboGripperVelocityToPosition       # ee.gripper_vel → ee.gripper_pos (三态)
      → robot.send_action()                 # IK 求 J1-J5(保持竖直), J6=ee.j6_target, servoJoint 关节伺服

设计：位置跟手机走；手机左右旋转 → 只驱动最后一个轴 J6(工具竖直时=绕竖直方向转)；
pitch/roll 锁死不动。彻底绕开笛卡尔姿态伺服里 IK 把偏航分配到 J4/J5(绕基座Y俯仰)的问题。
"""

import logging
import time
from pathlib import Path

import numpy as np

from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Config, AuboI10Robot
from lerobot.robots.aubo_i10.robot_processor import (
    AuboEEBoundsAndSafety,
    AuboGripperVelocityToPosition,
    AuboLockVerticalYaw,
    PhoneEEToAuboEE,
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging

FPS = 30


def _fmt_val(v):
    """把单个值格式化为紧凑字符串（数值保留 3 位，图像等跳过）。"""
    # Rotation 对象：打印其 rotvec（弧度），方便排查手机旋转信号落在哪个轴
    if isinstance(v, Rotation):
        try:
            rv = np.asarray(v.as_rotvec()).ravel()
            return "rotvec[" + ",".join(f"{float(x):.3f}" for x in rv.tolist()) + "]"
        except Exception:
            return repr(v)
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.3f}"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(v).ravel()
            if arr.size > 12:
                return f"<array shape={np.asarray(v).shape}>"
            return "[" + ",".join(_fmt_val(x) for x in arr.tolist()) + "]"
        except Exception:
            return repr(v)
    return repr(v)


def _fmt_dict(d):
    """把观测/动作字典格式化为 key=value 的紧凑字符串，跳过图像字段。"""
    if not isinstance(d, dict):
        return repr(d)
    parts = []
    for k, v in d.items():
        # 图像/大数组不展开，只标注类型，避免日志爆炸
        if "image" in str(k).lower() or isinstance(v, np.ndarray) and v.ndim >= 2:
            parts.append(f"{k}=<image>")
            continue
        # 嵌套字典（如 phone.raw_inputs）递归展开
        if isinstance(v, dict):
            parts.append(f"{k}={{{_fmt_dict(v)}}}")
            continue
        parts.append(f"{k}={_fmt_val(v)}")
    return " ".join(parts)


def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "teleoperate.log"
    # 每次运行前清空旧日志，保证本次测试的日志干净、可单独排查。
    log_file.write_text("")
    init_logging(log_file=log_file, console_level="INFO", file_level="DEBUG")
    logging.info("=== 新一轮手机遥操测试开始（日志已覆盖） ===")

    robot_config = AuboI10Config()
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

    robot = AuboI10Robot(robot_config)
    phone = Phone(teleop_config)

    # 手机遥操控制通路（速度模式/操纵杆）
    # - 位置：速度控制，手机偏移量 → 每帧移动速度（鼠标加速：小偏移精确，大偏移快速）
    #         按下=新原点，手机不动=机器人不动，松手重按继续从当前位置
    # - 姿态：J6 直控（手机左右旋转=J6绕竖直轴转），pitch/roll 锁死
    phone_to_robot_ee_pose_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            AuboLockVerticalYaw(
                yaw_gain=-1.0,       # 负号修正方向（手机右转=J6正转）
                yaw_smoothing=0.0,   # 无滤波，yaw即时响应
            ),
            PhoneEEToAuboEE(
                velocity_mode=True,                          # 速度/操纵杆模式
                end_effector_step_sizes={"x": 0.05, "y": 0.05, "z": 0.05},  # 每帧基础速度(m/单位位移)
                position_acceleration=4.0,                   # 鼠标加速系数(大位移额外加速)
            ),
            AuboEEBoundsAndSafety(
                end_effector_bounds={"min": [-0.8, -1.2, -0.3], "max": [1.0, 0.0, 0.8]},
                max_ee_step_m=0.05,  # 每帧最大移动量(m)，也是最高速度限制
            ),
            AuboGripperVelocityToPosition(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()
    phone.connect()

    if not robot.is_connected or not phone.is_connected:
        raise ValueError("Robot or phone is not connected!")

    print("开始手机遥操（末端位姿 delta 模式，姿态锁定 + 手机 yaw）。按 Ctrl+C 退出。")
    # 每次启动前清除上一轮累积状态
    phone_to_robot_ee_pose_processor.reset()

    try:
        while True:
            t0 = time.perf_counter()

            robot_obs = robot.get_observation()
            phone_obs = phone.get_action()

            # 手机动作 → 末端位姿 delta（旋转锁定 + yaw，位置 delta）
            robot_action = phone_to_robot_ee_pose_processor((phone_obs, robot_obs))

            # 统一成 Python 标量，避免下游 SDK 收到 np.float64
            for key, val in robot_action.items():
                if isinstance(val, (np.floating, np.integer)):
                    robot_action[key] = float(val)

            # 把手机输入、机械臂观测、最终下发的动作都写进日志文件（DEBUG 级），
            # 方便排查遥操问题。控制台仍保持 INFO，不会被刷屏。
            logging.debug(
                "[loop] phone_obs=%s", _fmt_dict(phone_obs)
            )
            logging.debug(
                "[loop] robot_obs=%s", _fmt_dict(robot_obs)
            )
            logging.debug(
                "[loop] robot_action=%s", _fmt_dict(robot_action)
            )

            robot.send_action(robot_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止遥操。")
    finally:
        try:
            phone.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
