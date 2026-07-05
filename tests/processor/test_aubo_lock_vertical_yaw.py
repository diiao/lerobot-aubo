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

"""Tests for AuboLockVerticalYaw: 操纵杆速度模式 + 1:1 回归 + 限位/死区/限速."""

import math

import pytest

from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots.aubo_i10.robot_processor import AuboLockVerticalYaw


def _obs(j6_deg: float = 0.0) -> dict:
    """竖直向下 TCP 姿态观测。"""
    return {
        "J6": j6_deg,
        "ee.wx": 0.0,
        "ee.wy": math.pi,
        "ee.wz": 0.0,
    }


def _action(yaw_rad: float, enabled: bool = True) -> dict:
    """构造手机动作：phone_rotvec = [-twy, -twx, twz] = [0, 0, yaw_rad] → 绕 z 转 yaw_rad."""
    return {
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": float(yaw_rad),
        "enabled": enabled,
    }


def _run(step: AuboLockVerticalYaw, action: dict, obs: dict) -> dict:
    """单帧执行：设置 transition（带观测），再调 step.action(action)。"""
    step._current_transition = create_transition(observation=obs)
    return step.action(dict(action))


def _press_then_deflect(step, yaw_rad, n_frames, j6_deg=0.0):
    """按下(yaw=0) 1 帧, 再以 yaw_rad 持续 n_frames。返回每帧的 j6_target(度)。"""
    targets = []
    targets.append(math.degrees(_run(step, _action(0.0, True), _obs(j6_deg))["ee.j6_target"]))
    for _ in range(n_frames):
        targets.append(math.degrees(_run(step, _action(yaw_rad, True), _obs(j6_deg))["ee.j6_target"]))
    return targets


def test_velocity_mode_continuous_rotation():
    """速度模式: 手机持续偏转 → J6_target 持续朝同一方向变化(不会停下)."""
    step = AuboLockVerticalYaw(yaw_velocity_mode=True, yaw_vel_gain=-0.012)
    # 偏转 0.5 rad(≈28.6°), 超出 5° 死区
    targets = _press_then_deflect(step, yaw_rad=0.5, n_frames=5)
    # 第 1 帧(按下, yaw=0) → 0; 之后持续递减(负增益)
    assert targets[0] == pytest.approx(0.0, abs=1e-6)
    for i in range(1, len(targets)):
        assert targets[i] < targets[i - 1]  # 持续朝负方向转
    # 速度恒定(同偏转量) → 等差
    diffs = [targets[i] - targets[i - 1] for i in range(2, len(targets))]
    for d in diffs:
        assert d == pytest.approx(diffs[0], abs=1e-3)


def test_velocity_mode_return_to_neutral_stops():
    """速度模式: 手机回正(落入死区) → J6_target 停止变化."""
    step = AuboLockVerticalYaw(yaw_velocity_mode=True, yaw_vel_gain=-0.012)
    _press_then_deflect(step, yaw_rad=0.5, n_frames=3)  # 先转起来
    held = math.degrees(step._yaw_accum)  # 当前累计(度)
    # 回正到 0(在死区内)
    t1 = math.degrees(_run(step, _action(0.0, True), _obs())["ee.j6_target"])
    t2 = math.degrees(_run(step, _action(0.0, True), _obs())["ee.j6_target"])
    assert t1 == pytest.approx(held, abs=1e-3)  # 不再增长
    assert t2 == pytest.approx(t1, abs=1e-6)  # 持续回正 → 保持不动


def test_velocity_mode_deadzone():
    """速度模式: 偏移在死区内 → 速度 0, J6 不转."""
    step = AuboLockVerticalYaw(
        yaw_velocity_mode=True, yaw_vel_gain=-0.012, yaw_deadzone_deg=5.0
    )
    # 0.04 rad ≈ 2.3° < 5° 死区
    targets = _press_then_deflect(step, yaw_rad=0.04, n_frames=5)
    for t in targets:
        assert t == pytest.approx(0.0, abs=1e-6)


def test_velocity_mode_max_speed_cap():
    """速度模式: 大偏转 → 单帧变化不超过 max_yaw_vel_deg_per_s / fps."""
    step = AuboLockVerticalYaw(
        yaw_velocity_mode=True, yaw_vel_gain=-0.012, max_yaw_vel_deg_per_s=30.0,
        control_fps=30.0,
    )
    max_per_frame = 30.0 / 30.0  # 1°/帧
    _run(step, _action(0.0, True), _obs())  # 按下
    prev = math.degrees(step._yaw_accum)
    for _ in range(10):
        # 偏转 π(180°) 远超触顶阈值
        _run(step, _action(math.pi, True), _obs())
        cur = math.degrees(step._yaw_accum)
        assert abs(cur - prev) <= max_per_frame + 1e-6
        prev = cur


def test_velocity_mode_j6_limit_clamp():
    """速度模式: 持续旋转 → j6_target 钳在 [j6_min, j6_max], 不超限."""
    step = AuboLockVerticalYaw(
        yaw_velocity_mode=True, yaw_vel_gain=-0.012,
        j6_min_rad=-math.radians(10.0), j6_max_rad=math.radians(10.0),
    )
    # 负增益 + 正偏转 → accum 朝负方向, 撞 j6_min=-10°
    _run(step, _action(0.0, True), _obs(0.0))  # 按下建立中性点(yaw=0)
    for _ in range(200):
        out = _run(step, _action(0.5, True), _obs(0.0))
    assert out["ee.j6_target"] >= step.j6_min_rad - 1e-9
    assert math.degrees(out["ee.j6_target"]) == pytest.approx(-10.0, abs=1e-3)


def test_1to1_mode_holds_position():
    """1:1 模式回归: 手机保持某偏转角 → J6 跟到对应角后停住(不持续转)."""
    step = AuboLockVerticalYaw(yaw_velocity_mode=False, yaw_gain=-1.0)
    # 按下(yaw=0)
    _run(step, _action(0.0, True), _obs(0.0))
    # 转到 yaw=0.5 并保持
    t1 = math.degrees(_run(step, _action(0.5, True), _obs(0.0))["ee.j6_target"])
    t2 = math.degrees(_run(step, _action(0.5, True), _obs(0.0))["ee.j6_target"])
    # 1:1: 手机不动 → J6 不动(关键区别于速度模式)
    assert t1 == pytest.approx(t2, abs=1e-6)
    # 且角度 ≈ -0.5 rad * gain(-1) = 0.5 rad ≈ 28.6°... 实际: accum=0.5, phi=0.5*-1=-0.5 → -28.6°
    assert t1 == pytest.approx(math.degrees(-0.5), abs=1e-3)


def test_repress_resets_origin():
    """松手再按: 以当前 J6 为新原点, accum 清零, j6_target 无跳变."""
    step = AuboLockVerticalYaw(yaw_velocity_mode=True, yaw_vel_gain=-0.012)
    _press_then_deflect(step, yaw_rad=0.5, n_frames=3)  # 转一会
    # 松手
    _run(step, _action(0.0, False), _obs(j6_deg=20.0))  # 机器人 J6 现在在 20°
    # 重新按下(yaw=0): 应以 J6=20° 为新原点
    out = _run(step, _action(0.0, True), _obs(j6_deg=20.0))
    assert math.degrees(out["ee.j6_target"]) == pytest.approx(20.0, abs=1e-3)
    assert step._yaw_accum == pytest.approx(0.0, abs=1e-9)


def test_output_fields_present():
    """输出包含固定姿态 + ee.j6_target + ee_mode=abs_j6yaw."""
    step = AuboLockVerticalYaw(yaw_velocity_mode=True)
    out = _run(step, _action(0.0, True), _obs(0.0))
    assert out["ee_mode"] == "abs_j6yaw"
    assert "ee.j6_target" in out
    assert "ee.wx" in out and "ee.wy" in out and "ee.wz" in out
    # target_w* 已被消费
    assert "target_wx" not in out and "target_wy" not in out and "target_wz" not in out
