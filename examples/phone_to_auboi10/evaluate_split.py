#!/usr/bin/env python
"""拆分式推理的客户端，跑在本地工作站。

机器人 + 相机都在本地；策略推理在 GPU 机上（inference_server.py），经 Tailscale 连接。
本脚本：取观测 -> 发给服务器取得当前纯 ACT 动作 ->
先按实测位姿限步裁剪（同 8 月 3 日 AuboEEBoundsAndSafety），
再经过动作离散化、拒绝式安全门和 ee_mode 处理 ->
按授权模式决定是否下发 -> 写评估数据集。

默认是 DRY_RUN=1：连接机器人和相机、请求模型动作并记录，但绝不调用
robot.send_action。只有同时设置 DRY_RUN=0 和
POLICY_EXECUTION_AUTHORIZED=1 才会进入真实策略执行模式。

模型 raw/反归一化/迟滞/DO 证据写入独立 JSONL；默认路径由
EVAL_DATASET_PATH 派生，也可用 INFERENCE_TRACE_PATH 显式指定。trace 不改变
评估数据集里原有的迟滞后 0/100 action 语义，且拒绝覆盖已有文件。

操作同 evaluate.py：-> 开始/结束 episode，← 重录，Esc 全停。

前置：
  1. GPU 机上先启动 inference_server.py（见该文件头注释）。
  2. 本地机器人上电、相机就位（by-id + MJPG，同 record.py）。
  3. 本地有聚合数据集 bamboo_newview_full（取 features；aggregate.py 产物）。
"""

import json
import logging
import math
import os
import pickle
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.bamboo_sorting.contracts import ACTION_FIELD_NAMES
from lerobot.bamboo_sorting.thin_safety_gate import ThinSafetyGate, ThinSafetyGateLimits
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Config, AuboI10Robot
from lerobot.robots.aubo_i10.robot_processor import AuboEEBoundsAndSafety, AuboSetEEMode
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.rotation import Rotation
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

SCRIPT_DIR = Path(__file__).resolve().parent
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "1"))
EXPECTED_CONTROL_FPS = 25
HANDEYE_CAPTURE_FPS = 30
FIXED_CAPTURE_FPS = 25
EPISODE_TIME_SEC = float(os.environ.get("EPISODE_TIME_SEC", "10"))
TASK_DESCRIPTION = "抓取竹条"
TRAINING_DATASET_PATH = os.environ.get(
    "DATASET_PATH", str(SCRIPT_DIR / "datasets" / "bamboo_act_report_full")
)
LOCAL_EVAL_DATASET_PATH = os.environ.get(
    "EVAL_DATASET_PATH", str(SCRIPT_DIR / "datasets" / "bamboo_act_report_eval_run02")
)
# Generic physical safety limit, not a task-specific trajectory rule. 8 mm at
# 25 Hz caps off-distribution action jumps to 0.2 m/s while preserving the
# normal demonstrated motion range. Override only for a documented experiment.
MAX_EE_STEP_M = float(os.environ.get("MAX_EE_STEP_M", "0.008"))
MAX_J6_STEP_RAD = float(os.environ.get("MAX_J6_STEP_RAD", "0.03"))
# Same independent safety layer as 8 mm xyz, but for TCP orientation.
# 40 single-strip demos (41,704 adjacent steps): 41,516 action orientation
# changes are ~0 (locked pose); measured TCP geodesic p99.9=0.74°, continuous
# max=1.09°. 0.03 rad (1.72°) is ~1.6× that continuous max, matching J6, and
# clips the live02 3.5–7°/frame drift. Relock jumps are not continuous motion.
MAX_EE_ROT_STEP_RAD = float(os.environ.get("MAX_EE_ROT_STEP_RAD", "0.03"))
# IK continuity is a configuration-flip detector, not the 8 mm Cartesian cap.
# 0.03 rad (~1.7°) aborts mid-reach; 0.25 rad (~14°) still catches branch jumps.
MAX_IK_JOINT_STEP_RAD = float(os.environ.get("MAX_IK_JOINT_STEP_RAD", "0.25"))
# Split inference JPEG + Tailscale + GPU is far slower than a local 10 Hz VLA
# loop. 100 ms would reject the first frame before any action is sent.
MAX_OBSERVATION_AGE_MS = float(os.environ.get("MAX_OBSERVATION_AGE_MS", "2000"))
MAX_ACTION_CHUNK_AGE_MS = float(os.environ.get("MAX_ACTION_CHUNK_AGE_MS", "2000"))
START_JOINT_TOLERANCE_DEG = float(os.environ.get("START_JOINT_TOLERANCE_DEG", "2.0"))
CONNECT_TIMEOUT_S = float(os.environ.get("CONNECT_TIMEOUT_S", "10"))
INFERENCE_TIMEOUT_S = float(os.environ.get("INFERENCE_TIMEOUT_S", "2"))
MAX_RESPONSE_BYTES = int(os.environ.get("MAX_RESPONSE_BYTES", str(1024 * 1024)))
GRIPPER_ACTION_INDEX = ACTION_FIELD_NAMES.index("ee.gripper_pos")


def env_flag(name: str, default: bool = False) -> bool:
    """Parse one explicit boolean environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 1/0、true/false、yes/no 或 on/off")


DRY_RUN = env_flag("DRY_RUN", default=True)
POLICY_EXECUTION_AUTHORIZED = env_flag("POLICY_EXECUTION_AUTHORIZED", default=False)
AUTOMATIC_RETURN_AUTHORIZED = env_flag("AUTOMATIC_RETURN_AUTHORIZED", default=False)
# Teleop labels lock ee.wx/wy/wz for the whole press and let J6 carry yaw.
# Following the model's drifting rotvec, then clipping it toward measured TCP
# (which already includes J6 yaw), fights that split and leaves closed-loop
# observations out of distribution. Hold the first measured pose instead.
HOLD_LOCKED_EE_POSE = env_flag("HOLD_LOCKED_EE_POSE", default=True)
# Pure-ACT baseline: do not replace the selected gripper output using later
# chunk values.  This task-specific lookahead remains opt-in for a separately
# named ablation only.  The actuator-side 60/20 hysteresis remains unchanged.
USE_GRIPPER_CHUNK_LOOKAHEAD = env_flag("USE_GRIPPER_CHUNK_LOOKAHEAD", default=False)
GRIPPER_CHUNK_CLOSE_SCORE = float(os.environ.get("GRIPPER_CHUNK_CLOSE_SCORE", "20"))
# Pure-ACT baseline: execute the selected ACT motion output.  Selecting the
# lowest-z pose from a future chunk is task-specific and therefore opt-in for
# a separately named ablation; the generic 8 mm safety limiter still applies.
USE_MOTION_CHUNK_LOOKAHEAD = env_flag("USE_MOTION_CHUNK_LOOKAHEAD", default=False)
MOTION_CHUNK_MIN_Z_DROP_M = float(os.environ.get("MOTION_CHUNK_MIN_Z_DROP_M", "0.02"))
# live08 pick/place chatter: max(chunk) flickered 13–23 and 12–59. Feeding
# 100 or the raw -3 skipped the 20–60 hold band. Open only when the whole
# chunk is below approach noise (~5.5). 8 is from live08: false opens were
# 13.3/18.3/12.5; real place release dropped to -1.
GRIPPER_CHUNK_OPEN_SCORE = float(os.environ.get("GRIPPER_CHUNK_OPEN_SCORE", "8"))
GRIPPER_CHUNK_HOLD_SCORE = 40.0

# GPU 机（Tailscale）
SERVER_HOST = os.environ.get("SERVER_HOST", "100.88.143.45")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5555"))

# 相机用稳定的 by-id 路径 + MJPG（handeye 是当前眼在手外相机）
HANDEYE_DEV = "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"
FIXED_DEV = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"

# 起始关节角（度），与 record.py 一致。
# 每轮推理完后可在单独授权时按 r 自动归位到此 + 松吸盘。
START_JOINT_DEG = [-65.29, -5.88, 113.77, 31.07, 90.88, -185.32]
# Derived from all 180,294 labels in bamboo_act_report_extended137_v1.  The
# demonstrated J6 target range is [-4.365706, -1.490041] rad; retain the prior
# approximately 0.10 rad margins.  This is a task safety envelope, not a
# trajectory heuristic.
WORKSPACE_MIN_M = (0.0, -0.85, 0.05)
WORKSPACE_MAX_M = (0.67, -0.25, 0.30)
J6_TARGET_MIN_RAD = float(os.environ.get("J6_TARGET_MIN_RAD", "-4.47"))
J6_TARGET_MAX_RAD = float(os.environ.get("J6_TARGET_MAX_RAD", "-1.39"))


@dataclass
class ActionSafetyState:
    """Track discrete vacuum hysteresis; it never authorizes motion."""

    suction_on: bool = False
    initialized: bool = False

    def initialize_from_observation(self, observation: dict[str, Any]) -> None:
        self.suction_on = float(observation.get("gripper_pos", 0.0)) > 60.0
        self.initialized = True


class JsonlTraceWriter:
    """Append structured inference evidence without changing evaluation action semantics."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def resolve_trace_path(eval_root: Path) -> Path:
    raw = os.environ.get("INFERENCE_TRACE_PATH")
    if raw:
        path = Path(raw).expanduser()
        return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    return eval_root.parent / f"{eval_root.name}_inference_trace.jsonl"


def requested_gripper_do(robot: Any, *, target_suction_on: bool, transition: bool) -> dict[str, bool] | None:
    """Describe intended controller outputs; this is not physical gripper feedback."""
    if not transition:
        return None
    return {
        str(robot.suction_on_pin): bool(target_suction_on),
        str(robot.suction_off_pin): not bool(target_suction_on),
    }


def write_trace(trace_writer: Any | None, record: dict[str, Any]) -> None:
    if trace_writer is not None:
        trace_writer.write(record)


def attempt_trace_record(
    *,
    record_type: str,
    episode_index: int,
    attempt_index: int,
    disposition: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build an episode-attempt lifecycle record for unambiguous rerecord traces."""
    if record_type not in {"attempt_start", "attempt_end"}:
        raise ValueError(f"未知 attempt trace 类型: {record_type}")
    if record_type == "attempt_start" and (disposition is not None or error is not None):
        raise ValueError("attempt_start 不应包含 disposition 或 error")
    if record_type == "attempt_end" and disposition not in {"saved", "rerecorded", "aborted"}:
        raise ValueError(f"未知 attempt disposition: {disposition}")
    return {
        "schema_version": "aubo_act_execution_trace_v1",
        "record_type": record_type,
        "episode_index": int(episode_index),
        "attempt_index": int(attempt_index),
        "attempt_id": f"episode-{episode_index:04d}-attempt-{attempt_index:03d}",
        "disposition": disposition,
        "error": error,
    }


def resolve_local_dataset_path(path_value: str, *, must_exist: bool) -> tuple[str, Path]:
    """Resolve an explicit local LeRobot dataset without falling back to HF cache."""
    root = Path(path_value).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    if must_exist and not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"本地 LeRobot 数据集不存在或不完整: {root}")
    return root.name, root


def validate_runtime_settings() -> None:
    """Validate all settings before opening cameras, robot RPC, or the server."""
    if not DRY_RUN and not POLICY_EXECUTION_AUTHORIZED:
        raise PermissionError(
            "拒绝真实执行：DRY_RUN=0 时还必须显式设置 POLICY_EXECUTION_AUTHORIZED=1"
        )
    if NUM_EPISODES <= 0:
        raise ValueError("NUM_EPISODES 必须大于 0")
    if EPISODE_TIME_SEC <= 0:
        raise ValueError("EPISODE_TIME_SEC 必须大于 0")
    if not 0.001 <= MAX_EE_STEP_M <= 0.05:
        raise ValueError(f"MAX_EE_STEP_M 必须在 [0.001, 0.05] m，当前为 {MAX_EE_STEP_M}")
    if not 0 < MAX_J6_STEP_RAD <= 0.4:
        raise ValueError(f"MAX_J6_STEP_RAD 必须在 (0, 0.4] rad，当前为 {MAX_J6_STEP_RAD}")
    if not math.isfinite(J6_TARGET_MIN_RAD) or not math.isfinite(J6_TARGET_MAX_RAD):
        raise ValueError("J6_TARGET_MIN_RAD 和 J6_TARGET_MAX_RAD 必须是有限数")
    if J6_TARGET_MIN_RAD >= J6_TARGET_MAX_RAD:
        raise ValueError(
            "J6_TARGET_MIN_RAD 必须小于 J6_TARGET_MAX_RAD，"
            f"当前为 [{J6_TARGET_MIN_RAD}, {J6_TARGET_MAX_RAD}]"
        )
    if not 0 < MAX_EE_ROT_STEP_RAD <= 0.4:
        raise ValueError(
            "MAX_EE_ROT_STEP_RAD 必须在 (0, 0.4] rad，"
            f"当前为 {MAX_EE_ROT_STEP_RAD}"
        )
    if not 0 < MAX_IK_JOINT_STEP_RAD <= 0.4:
        raise ValueError(
            "MAX_IK_JOINT_STEP_RAD 必须在 (0, 0.4] rad，"
            f"当前为 {MAX_IK_JOINT_STEP_RAD}"
        )
    if MAX_OBSERVATION_AGE_MS <= 0 or MAX_ACTION_CHUNK_AGE_MS <= 0:
        raise ValueError("观测和动作时效阈值必须大于 0")
    if CONNECT_TIMEOUT_S <= 0 or INFERENCE_TIMEOUT_S <= 0:
        raise ValueError("网络超时必须大于 0")
    if not 1024 <= MAX_RESPONSE_BYTES <= 64 * 1024 * 1024:
        raise ValueError("MAX_RESPONSE_BYTES 必须在 [1024, 67108864]")
    if START_JOINT_TOLERANCE_DEG <= 0:
        raise ValueError("START_JOINT_TOLERANCE_DEG 必须大于 0")
    if not 0 < GRIPPER_CHUNK_CLOSE_SCORE <= 100:
        raise ValueError(
            "GRIPPER_CHUNK_CLOSE_SCORE 必须在 (0, 100]，"
            f"当前为 {GRIPPER_CHUNK_CLOSE_SCORE}"
        )
    if not 0 <= GRIPPER_CHUNK_OPEN_SCORE < GRIPPER_CHUNK_CLOSE_SCORE:
        raise ValueError(
            "GRIPPER_CHUNK_OPEN_SCORE 必须在 [0, GRIPPER_CHUNK_CLOSE_SCORE)，"
            f"当前为 {GRIPPER_CHUNK_OPEN_SCORE}"
        )
    if not 0 < MOTION_CHUNK_MIN_Z_DROP_M <= 0.15:
        raise ValueError(
            "MOTION_CHUNK_MIN_Z_DROP_M 必须在 (0, 0.15] m，"
            f"当前为 {MOTION_CHUNK_MIN_Z_DROP_M}"
        )


def discretize_gripper_command(raw_value: Any, *, suction_on: bool) -> tuple[float, bool]:
    """Map the policy score to the existing physical 0/100 hysteresis semantics."""
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError("ee.gripper_pos 必须是有限数")
    if value > 60.0:
        return 100.0, True
    if value < 20.0:
        return 0.0, False
    return (100.0, True) if suction_on else (0.0, False)


def gripper_chunk_max(chunk: Any) -> float | None:
    """Return max denormalized gripper in an ACT chunk, or None if unusable."""
    if chunk is None:
        return None
    values = [float(value) for value in chunk]
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return max(values)


def apply_motion_chunk_lookahead(
    action: RobotAction,
    chunk_actions: Any,
    *,
    min_z_drop_m: float,
    suction_on: bool,
) -> tuple[RobotAction, bool]:
    """If the ACT chunk plans a descent, command that pose (still rate-limited)."""
    if suction_on or chunk_actions is None:
        return dict(action), False
    rows = [list(row) for row in chunk_actions]
    if not rows or any(len(row) < 4 for row in rows):
        return dict(action), False
    zs = [float(row[3]) for row in rows]
    if not all(math.isfinite(value) for value in zs):
        return dict(action), False
    index = min(range(len(zs)), key=lambda i: zs[i])
    planned_z = zs[index]
    current_z = float(action["ee.z"])
    if current_z - planned_z < min_z_drop_m:
        return dict(action), False
    planned = rows[index]
    updated = dict(action)
    updated["ee.j6_target"] = float(planned[0])
    updated["ee.x"] = float(planned[1])
    updated["ee.y"] = float(planned[2])
    updated["ee.z"] = float(planned[3])
    return updated, True


def apply_gripper_chunk_lookahead(
    action: RobotAction,
    chunk: Any,
    *,
    close_score: float,
    open_score: float,
    hold_score: float = GRIPPER_CHUNK_HOLD_SCORE,
) -> tuple[RobotAction, str]:
    """Map chunk max to close/hold/pass so hysteresis 20–60 can stick."""
    chunk_max = gripper_chunk_max(chunk)
    if chunk_max is None:
        return dict(action), "pass"
    if chunk_max > close_score:
        overridden = dict(action)
        overridden["ee.gripper_pos"] = 100.0
        return overridden, "close"
    if chunk_max < open_score:
        return dict(action), "pass"
    held = dict(action)
    held["ee.gripper_pos"] = float(hold_score)
    return held, "hold"


def normalize_action_for_actuator(
    action: RobotAction, *, suction_on: bool
) -> tuple[RobotAction, bool]:
    """Make the discrete vacuum command explicit; Cartesian/J6 values are untouched."""
    normalized = dict(action)
    gripper, next_suction_on = discretize_gripper_command(
        normalized["ee.gripper_pos"], suction_on=suction_on
    )
    normalized["ee.gripper_pos"] = gripper
    return normalized, next_suction_on


def action_vector(action: RobotAction) -> tuple[float, ...]:
    """Convert the named robot action to ordered ActionSchemaV1 values."""
    return tuple(float(action[name]) for name in ACTION_FIELD_NAMES)


def measured_action_anchors(
    observation: dict[str, Any],
) -> tuple[tuple[float, float, float], float]:
    """Return measured TCP in metres and convert observed J6 degrees to radians."""
    tcp_m = tuple(float(observation[f"ee.{axis}"]) for axis in "xyz")
    j6_rad = math.radians(float(observation["J6"]))
    return tcp_m, j6_rad


def measured_tcp_rotvec(observation: dict[str, Any]) -> tuple[float, float, float]:
    """Return the measured base-frame TCP rotation vector in radians."""
    return tuple(float(observation[f"ee.w{axis}"]) for axis in "xyz")


def rotation_geodesic_angle_rad(
    current_rotvec: tuple[float, float, float] | np.ndarray,
    target_rotvec: tuple[float, float, float] | np.ndarray,
) -> float:
    """Return the geodesic angle between two rotation vectors, in radians."""
    current = Rotation.from_rotvec(np.asarray(current_rotvec, dtype=float))
    target = Rotation.from_rotvec(np.asarray(target_rotvec, dtype=float))
    delta = (target * current.inv()).as_rotvec()
    return float(np.linalg.norm(delta))


def align_rotvec_to_reference(
    target_rotvec: tuple[float, float, float] | np.ndarray,
    reference_rotvec: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Re-express target on the 2π branch nearest the measured/reference rotvec."""
    target = np.asarray(target_rotvec, dtype=float)
    reference = np.asarray(reference_rotvec, dtype=float)
    primary = Rotation.from_rotvec(target).as_rotvec()
    candidates = [primary]
    norm = float(np.linalg.norm(primary))
    if norm > 1e-9:
        axis = primary / norm
        candidates.append(primary + 2.0 * math.pi * axis)
        candidates.append(primary - 2.0 * math.pi * axis)
    return np.asarray(
        min(candidates, key=lambda item: float(np.linalg.norm(np.asarray(item) - reference))),
        dtype=float,
    )


def apply_locked_ee_pose(
    action: RobotAction, locked_rotvec: tuple[float, float, float]
) -> RobotAction:
    """Replace model orientation with the episode-start locked TCP rotvec."""
    locked = dict(action)
    locked["ee.wx"] = float(locked_rotvec[0])
    locked["ee.wy"] = float(locked_rotvec[1])
    locked["ee.wz"] = float(locked_rotvec[2])
    return locked


def clip_rotvec_toward_measured(
    target_rotvec: tuple[float, float, float] | np.ndarray,
    previous_rotvec: tuple[float, float, float] | np.ndarray,
    max_step_rad: float,
) -> np.ndarray:
    """Take at most one geodesic step from the measured pose, then align branches."""
    previous = np.asarray(previous_rotvec, dtype=float)
    target = np.asarray(target_rotvec, dtype=float)
    r_prev = Rotation.from_rotvec(previous)
    r_tgt = Rotation.from_rotvec(target)
    rv_delta = (r_tgt * r_prev.inv()).as_rotvec()
    angle = float(np.linalg.norm(rv_delta))
    limit = max_step_rad * (1.0 - 1e-6)
    if angle > limit and angle > 0.0:
        next_rv = (Rotation.from_rotvec(rv_delta * (limit / angle)) * r_prev).as_rotvec()
    else:
        next_rv = target
    return align_rotvec_to_reference(next_rv, previous)


def clip_absolute_action_to_measured_limits(
    action: RobotAction,
    *,
    previous_tcp_m: tuple[float, float, float],
    previous_j6_rad: float,
    max_ee_step_m: float,
    max_j6_step_rad: float,
    previous_rotvec: tuple[float, float, float] | None = None,
    max_ee_rot_step_rad: float | None = None,
) -> RobotAction:
    """Clip one absolute command toward the measured pose instead of rejecting it."""
    clipped = dict(action)
    target = np.array(
        [float(clipped["ee.x"]), float(clipped["ee.y"]), float(clipped["ee.z"])],
        dtype=float,
    )
    previous = np.asarray(previous_tcp_m, dtype=float)
    delta = target - previous
    dist = float(np.linalg.norm(delta))
    ee_limit = max_ee_step_m * (1.0 - 1e-6)
    if dist > ee_limit and dist > 0.0:
        target = previous + delta * (ee_limit / dist)
        clipped["ee.x"] = float(target[0])
        clipped["ee.y"] = float(target[1])
        clipped["ee.z"] = float(target[2])
    j6 = float(clipped["ee.j6_target"])
    j6_limit = max_j6_step_rad * (1.0 - 1e-6)
    delta_j6 = j6 - previous_j6_rad
    if abs(delta_j6) > j6_limit:
        clipped["ee.j6_target"] = float(previous_j6_rad + math.copysign(j6_limit, delta_j6))
    if previous_rotvec is not None:
        if max_ee_rot_step_rad is None:
            raise ValueError("max_ee_rot_step_rad is required when clipping orientation")
        clipped_rotvec = clip_rotvec_toward_measured(
            (float(clipped["ee.wx"]), float(clipped["ee.wy"]), float(clipped["ee.wz"])),
            previous_rotvec,
            max_ee_rot_step_rad,
        )
        clipped["ee.wx"] = float(clipped_rotvec[0])
        clipped["ee.wy"] = float(clipped_rotvec[1])
        clipped["ee.wz"] = float(clipped_rotvec[2])
    return clipped


def assert_task_action_envelope(vector: tuple[float, ...]) -> None:
    """Reject slow J6 drift outside the demonstrated task envelope."""
    j6_target = vector[0]
    if not J6_TARGET_MIN_RAD <= j6_target <= J6_TARGET_MAX_RAD:
        raise RuntimeError(
            f"J6 目标 {j6_target:.4f} rad 超出任务安全范围 "
            f"[{J6_TARGET_MIN_RAD:.2f}, {J6_TARGET_MAX_RAD:.2f}]"
        )


def assert_start_pose(joints_deg: list[float] | tuple[float, ...]) -> None:
    """Reject out-of-distribution starts before resetting or executing the policy."""
    if len(joints_deg) != len(START_JOINT_DEG):
        raise RuntimeError("无法验证起始位：关节数量不是 6")
    errors = [
        abs(float(actual) - expected)
        for actual, expected in zip(joints_deg, START_JOINT_DEG, strict=True)
    ]
    if any(not math.isfinite(error) for error in errors):
        raise RuntimeError("无法验证起始位：关节状态包含非有限数")
    worst = max(errors)
    if worst > START_JOINT_TOLERANCE_DEG:
        detail = ", ".join(f"J{i + 1}={error:.2f}°" for i, error in enumerate(errors))
        raise RuntimeError(
            f"起始位检查失败（最大偏差 {worst:.2f}° > "
            f"{START_JOINT_TOLERANCE_DEG:.2f}°）：{detail}"
        )


def read_current_joints_deg(robot: AuboI10Robot) -> list[float]:
    joints_rad = robot.robot_interface.getRobotState().getJointPositions()
    return [math.degrees(float(value)) for value in joints_rad]


def diagnose_inverse_kinematics(
    robot: AuboI10Robot, action: tuple[float, ...]
) -> dict[str, Any]:
    """Return IK errno, solution, and joint deltas without sending motion."""
    trace: dict[str, Any] = {
        "passed": False,
        "errno": None,
        "solution_rad": None,
        "seed_rad": None,
        "commanded_joints_rad": None,
        "joint_delta_rad": None,
        "joint_delta_deg": None,
        "max_abs_delta_rad": None,
        "max_abs_delta_deg": None,
        "limit_rad": MAX_IK_JOINT_STEP_RAD,
        "failed_joints": [],
        "error": None,
    }
    try:
        seed = [float(value) for value in robot.robot_interface.getRobotState().getJointPositions()]
        solution, errno = robot.robot_interface.getRobotAlgorithm().inverseKinematics(
            seed, list(action[1:7])
        )
    except Exception as exc:
        trace["error"] = str(exc)
        return trace
    trace["errno"] = int(errno)
    trace["seed_rad"] = seed
    if int(errno) != 0 or len(solution) != 6:
        trace["solution_rad"] = [float(value) for value in solution]
        return trace
    solution_rad = [float(value) for value in solution]
    commanded = list(solution_rad)
    commanded[5] = float(action[0])
    trace["solution_rad"] = solution_rad
    trace["commanded_joints_rad"] = commanded
    if not all(math.isfinite(value) for value in commanded):
        trace["error"] = "non-finite IK solution"
        return trace
    deltas = [target - current for target, current in zip(commanded, seed, strict=True)]
    trace["joint_delta_rad"] = deltas
    trace["joint_delta_deg"] = [math.degrees(value) for value in deltas]
    trace["max_abs_delta_rad"] = max(abs(value) for value in deltas)
    trace["max_abs_delta_deg"] = math.degrees(trace["max_abs_delta_rad"])
    trace["failed_joints"] = [
        f"J{index + 1}"
        for index, value in enumerate(deltas)
        if abs(value) > MAX_IK_JOINT_STEP_RAD
    ]
    trace["passed"] = not trace["failed_joints"]
    return trace


def make_ik_checker(robot: AuboI10Robot):
    """Check IK and reject discontinuous joint solutions before action send."""

    def check(action: tuple[float, ...]) -> bool:
        check.last_trace = diagnose_inverse_kinematics(robot, action)
        return bool(check.last_trace["passed"])

    check.last_trace = None
    return check


def return_to_start(robot):
    """关伺服后用 moveJoint 回到起始关节位 + 松吸盘。重置阶段按 r 触发。"""
    logging.info("归位：moveJoint 到起始位...")
    motion = robot.robot_interface.getMotionControl()
    if motion.isServoModeEnabled():
        motion.setServoMode(False)
        time.sleep(0.5)
    target_rad = [math.radians(d) for d in START_JOINT_DEG]
    motion.setSpeedFraction(0.5)
    motion.moveJoint(target_rad, 80 * (math.pi / 180), 60 * (math.pi / 180), 0, 0)
    exec_id = motion.getExecId()
    cnt = 0
    while exec_id == -1:
        if cnt > 100:
            break
        time.sleep(0.05)
        cnt += 1
        exec_id = motion.getExecId()
    while motion.getExecId() != -1:
        time.sleep(0.05)
    try:
        if not robot.suction_release():
            logging.error("归位完成，但吸盘释放失败")
        else:
            logging.info("归位完成，吸盘已释放")
    except Exception as e:
        logging.error(f"吸盘释放失败: {e}")


# ----------------------------- 网络协议（与服务端一致）-----------------------------
def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock, *, max_bytes: int = MAX_RESPONSE_BYTES):
    header = recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > max_bytes:
        raise RuntimeError(f"服务器响应长度非法: {length} bytes（上限 {max_bytes}）")
    payload = recv_exactly(sock, length)
    if payload is None:
        return None
    return pickle.loads(payload)


def encode_obs(obs_frame: dict, task: str, robot_type: str) -> dict:
    """observation_frame(numpy) -> 可 pickle 的传输 dict。图像 JPEG 压缩。"""
    msg = {}
    for k, v in obs_frame.items():
        if "image" in k:
            ok, jpg = cv2.imencode(".jpg", v, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError(f"JPEG 编码失败: {k}")
            msg[k] = jpg.tobytes()
        else:
            msg[k] = np.asarray(v, dtype=np.float32)
    msg["task"] = task
    msg["robot_type"] = robot_type
    return msg


# ----------------------------- 控制循环（替代 record_loop）-----------------------------
def run_episode(
    robot,
    events,
    fps,
    sock,
    dataset,
    control_time_s,
    single_task,
    ee_mode_processor,
    robot_observation_processor,
    safety_gate: ThinSafetyGate,
    execute_actions: bool,
    trace_writer: Any | None = None,
    episode_index: int = 0,
    attempt_index: int = 0,
):
    action_queue = deque()  # 保留协议兼容性；当前服务端每次返回一步动作
    safety_state = ActionSafetyState()
    ik_checker = make_ik_checker(robot)
    locked_rotvec: tuple[float, float, float] | None = None
    dt = 1.0 / fps
    timestamp = 0.0
    step_index = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # 1. 观测
        obs = robot.get_observation()
        observation_timestamp_s = time.monotonic()
        if int(timestamp * fps + 1e-6) % int(fps) == 0:
            try:
                joints = [float(obs[f"J{i}"]) for i in range(1, 7)]
                logging.info(
                    "pose t=%.1fs J=[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f] "
                    "TCP=[%.4f, %.4f, %.4f] gripper=%.0f",
                    timestamp,
                    *joints,
                    float(obs["ee.x"]),
                    float(obs["ee.y"]),
                    float(obs["ee.z"]),
                    float(obs.get("gripper_pos", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                pass
        if not safety_state.initialized:
            safety_state.initialize_from_observation(obs)
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # 2. 队列空 -> 向服务器请求动作（ensemble 模式返回 1 个，queue 模式返回多个）
        if len(action_queue) == 0:
            msg = {
                "cmd": "predict",
                "obs": encode_obs(observation_frame, single_task, robot.robot_type),
            }
            try:
                send_msg(sock, msg)
                resp = recv_msg(sock)
            except (OSError, TimeoutError) as exc:
                if execute_actions:
                    robot.disable_servo_mode()
                raise RuntimeError("推理服务器超时或连接中断，已停止本轮") from exc
            if resp is None or "error" in resp:
                if execute_actions:
                    robot.disable_servo_mode()
                raise RuntimeError(f"服务器返回错误: {resp}")
            if "actions" not in resp:
                if execute_actions:
                    robot.disable_servo_mode()
                raise RuntimeError(f"服务器响应缺少 actions: {resp}")
            actions = np.atleast_2d(np.asarray(resp["actions"], dtype=np.float32))
            if actions.ndim != 2 or actions.shape[1] != len(ACTION_FIELD_NAMES):
                if execute_actions:
                    robot.disable_servo_mode()
                raise RuntimeError(
                    f"服务器动作形状错误: {actions.shape}，应为 (N, {len(ACTION_FIELD_NAMES)})"
                )
            chunk_created_s = time.monotonic()
            server_trace = resp.get("trace")
            for row_index, row in enumerate(actions):
                act_tensor = torch.from_numpy(row).unsqueeze(0)  # (1, action_dim)
                action_queue.append(
                    (
                        make_robot_action(act_tensor, dataset.features),
                        chunk_created_s,
                        server_trace if row_index == 0 else None,
                    )
                )

        # 3. 取一个动作 -> 离散吸盘 -> 按实测位姿限步裁剪 -> 安全门 -> 按模式决定是否下发
        raw_act, chunk_created_s, server_trace = action_queue.popleft()
        previous_suction_on = safety_state.suction_on
        selected_gripper = float(raw_act["ee.gripper_pos"])
        chunk_actions = (server_trace or {}).get("predicted_chunk_denormalized_action")
        motion_lookahead_overrode = False
        if USE_MOTION_CHUNK_LOOKAHEAD:
            raw_act, motion_lookahead_overrode = apply_motion_chunk_lookahead(
                raw_act,
                chunk_actions,
                min_z_drop_m=MOTION_CHUNK_MIN_Z_DROP_M,
                suction_on=previous_suction_on,
            )
        chunk_gripper = (
            (server_trace or {}).get("predicted_chunk_denormalized_gripper")
        )
        chunk_max = None
        if chunk_gripper is not None:
            chunk_values = [float(value) for value in chunk_gripper]
            if chunk_values:
                chunk_max = max(chunk_values)
        lookahead_mode = "pass"
        if USE_GRIPPER_CHUNK_LOOKAHEAD:
            raw_act, lookahead_mode = apply_gripper_chunk_lookahead(
                raw_act,
                chunk_gripper,
                close_score=GRIPPER_CHUNK_CLOSE_SCORE,
                open_score=GRIPPER_CHUNK_OPEN_SCORE,
            )
        lookahead_overrode = lookahead_mode == "close"
        act, next_suction_on = normalize_action_for_actuator(
            raw_act, suction_on=previous_suction_on
        )
        transition_requested = next_suction_on != previous_suction_on
        trace_record: dict[str, Any] = {
            "schema_version": "aubo_act_execution_trace_v1",
            "record_type": "step",
            "episode_index": int(episode_index),
            "attempt_index": int(attempt_index),
            "attempt_id": f"episode-{episode_index:04d}-attempt-{attempt_index:03d}",
            "step_index": int(step_index),
            "control_timestamp_s": float(timestamp),
            "observation_monotonic_s": float(observation_timestamp_s),
            "chunk_created_monotonic_s": float(chunk_created_s),
            "execution_mode": "authorized_execution" if execute_actions else "dry_run",
            "server": server_trace,
            "gripper": {
                "selected_normalized": (
                    float(server_trace["selected_normalized_action"][GRIPPER_ACTION_INDEX])
                    if server_trace and server_trace.get("selected_normalized_action")
                    else None
                ),
                "raw_denormalized": selected_gripper,
                "chunk_max_denormalized": chunk_max,
                "chunk_lookahead_enabled": USE_GRIPPER_CHUNK_LOOKAHEAD,
                "chunk_lookahead_mode": lookahead_mode,
                "chunk_lookahead_overrode": lookahead_overrode,
                "thresholded_command": float(act["ee.gripper_pos"]),
                "hysteresis_state_before": bool(previous_suction_on),
                "hysteresis_state_candidate": bool(next_suction_on),
                "transition_requested": bool(transition_requested),
                "requested_do": requested_gripper_do(
                    robot,
                    target_suction_on=next_suction_on,
                    transition=transition_requested,
                ),
            },
            "safety": {"passed": None, "reasons": []},
            "pose_clip": None,
            "motion_chunk": {
                "enabled": USE_MOTION_CHUNK_LOOKAHEAD,
                "overrode": motion_lookahead_overrode,
                "min_z_drop_m": MOTION_CHUNK_MIN_Z_DROP_M,
                "chunk_min_z": (
                    min(float(row[3]) for row in chunk_actions)
                    if chunk_actions
                    else None
                ),
                "commanded_z": float(raw_act["ee.z"]),
            },
            "ik": None,
            "actuator": {
                "attempted": False,
                "success": None,
                "gripper_io": None,
                "commanded_state_before": bool(
                    getattr(robot, "is_suction_on", previous_suction_on)
                ),
                "commanded_state_after": bool(
                    getattr(robot, "is_suction_on", previous_suction_on)
                ),
            },
            "outcome": "pending",
        }
        measured_tcp_m, measured_j6_rad = measured_action_anchors(obs)
        measured_rotvec = measured_tcp_rotvec(obs)
        raw_rotvec = (
            float(act["ee.wx"]),
            float(act["ee.wy"]),
            float(act["ee.wz"]),
        )
        if HOLD_LOCKED_EE_POSE:
            if locked_rotvec is None:
                locked_rotvec = measured_rotvec
            act = apply_locked_ee_pose(act, locked_rotvec)
        ik_checker.last_trace = None
        clip_rotvec = None if HOLD_LOCKED_EE_POSE else measured_rotvec
        act = clip_absolute_action_to_measured_limits(
            act,
            previous_tcp_m=measured_tcp_m,
            previous_j6_rad=measured_j6_rad,
            max_ee_step_m=MAX_EE_STEP_M,
            max_j6_step_rad=MAX_J6_STEP_RAD,
            previous_rotvec=clip_rotvec,
            max_ee_rot_step_rad=None if clip_rotvec is None else MAX_EE_ROT_STEP_RAD,
        )
        sent_rotvec = (
            float(act["ee.wx"]),
            float(act["ee.wy"]),
            float(act["ee.wz"]),
        )
        trace_record["pose_clip"] = {
            "hold_locked_ee_pose": HOLD_LOCKED_EE_POSE,
            "locked_rotvec": list(locked_rotvec) if locked_rotvec is not None else None,
            "measured_rotvec": list(measured_rotvec),
            "raw_rotvec": list(raw_rotvec),
            "sent_rotvec": list(sent_rotvec),
            "geodesic_to_measured_rad": rotation_geodesic_angle_rad(
                measured_rotvec, raw_rotvec
            ),
            "sent_geodesic_to_measured_rad": rotation_geodesic_angle_rad(
                measured_rotvec, sent_rotvec
            ),
            "max_ee_rot_step_rad": MAX_EE_ROT_STEP_RAD,
        }
        robot_action = ee_mode_processor((act, obs))
        vector = action_vector(robot_action)
        try:
            assert_task_action_envelope(vector)
            gated = safety_gate.evaluate(
                [vector],
                now_monotonic_s=time.monotonic(),
                observation_sync_timestamp_s=observation_timestamp_s,
                chunk_created_monotonic_s=chunk_created_s,
                previous_tcp_m=measured_tcp_m,
                previous_j6_rad=measured_j6_rad,
                ik_checker=ik_checker,
            )
        except Exception as exc:
            trace_record["safety"] = {"passed": False, "reasons": [str(exc)]}
            trace_record["ik"] = getattr(ik_checker, "last_trace", None)
            trace_record["outcome"] = "pre_send_rejected"
            write_trace(trace_writer, trace_record)
            if execute_actions:
                robot.disable_servo_mode()
            raise
        trace_record["safety"] = {
            "passed": bool(gated.passed),
            "reasons": list(gated.reasons),
        }
        trace_record["ik"] = getattr(ik_checker, "last_trace", None)
        if not gated.passed:
            trace_record["outcome"] = "thin_safety_gate_rejected"
            write_trace(trace_writer, trace_record)
            if execute_actions:
                robot.disable_servo_mode()
            raise RuntimeError(
                "ThinSafetyGate 拒绝动作，已停止本轮: " + ", ".join(gated.reasons)
            )
        if execute_actions:
            trace_record["actuator"]["attempted"] = True
            try:
                robot.send_action(robot_action)
            except Exception as exc:
                trace_record["actuator"]["success"] = False
                trace_record["actuator"]["gripper_io"] = getattr(
                    robot, "last_gripper_command_trace", None
                )
                trace_record["actuator"]["commanded_state_after"] = bool(
                    getattr(robot, "is_suction_on", previous_suction_on)
                )
                trace_record["actuator"]["error"] = str(exc)
                trace_record["outcome"] = "actuator_error"
                write_trace(trace_writer, trace_record)
                robot.disable_servo_mode()
                raise RuntimeError("动作或夹爪 DO 下发失败，已停止本轮并关闭伺服") from exc

            commanded_state_after = bool(getattr(robot, "is_suction_on", next_suction_on))
            if commanded_state_after != next_suction_on:
                trace_record["actuator"]["success"] = False
                trace_record["actuator"]["gripper_io"] = getattr(
                    robot, "last_gripper_command_trace", None
                )
                trace_record["actuator"]["commanded_state_after"] = commanded_state_after
                trace_record["actuator"]["error"] = "software commanded state mismatch"
                trace_record["outcome"] = "actuator_state_mismatch"
                write_trace(trace_writer, trace_record)
                robot.disable_servo_mode()
                raise RuntimeError("动作下发后夹爪软件状态不一致，已停止本轮")
            safety_state.suction_on = commanded_state_after
            trace_record["actuator"]["success"] = True
            trace_record["actuator"]["gripper_io"] = getattr(
                robot, "last_gripper_command_trace", None
            )
            trace_record["actuator"]["commanded_state_after"] = commanded_state_after
            motion = robot.robot_interface.getMotionControl()
            if not robot.is_servo_mode_enabled or not motion.isServoModeEnabled():
                trace_record["actuator"]["success"] = False
                trace_record["actuator"]["error"] = "servo mode not enabled after action"
                trace_record["outcome"] = "servo_mode_lost"
                write_trace(trace_writer, trace_record)
                robot.disable_servo_mode()
                raise RuntimeError("动作下发后伺服模式未保持启用，已停止本轮")
        else:
            # Dry-run hysteresis is a simulation; no robot commanded state is changed.
            safety_state.suction_on = next_suction_on
            logging.info(
                "DRY_RUN 动作通过但未下发: xyz=[%.4f, %.4f, %.4f], J6=%.4f, gripper=%.0f",
                vector[1],
                vector[2],
                vector[3],
                vector[0],
                vector[7],
            )

        trace_record["outcome"] = "action_sent" if execute_actions else "dry_run_not_sent"
        write_trace(trace_writer, trace_record)

        # 4. 写评估数据集（obs + action + task，便于回放）
        action_frame = build_dataset_frame(dataset.features, act, prefix=ACTION)
        dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
        step_index += 1

        # 5. 维持帧率
        timestamp = time.perf_counter() - start_episode_t
        elapsed = time.perf_counter() - start_loop_t
        if elapsed < dt:
            precise_sleep(dt - elapsed)
        else:
            logging.warning(
                "推理控制循环低于目标频率: %.1f Hz（目标 %d Hz）",
                1.0 / elapsed,
                fps,
            )


def wait_for_key(events: dict, prompt: str = "按 -> (右箭头键) 继续") -> bool:
    events["exit_early"] = False
    print("\n" + "=" * 50)
    print(prompt)
    print("按 Esc 终止")
    print("=" * 50)
    while not events["exit_early"] and not events["stop_recording"]:
        time.sleep(0.05)
    if events["stop_recording"]:
        return False
    events["exit_early"] = False
    return True


def wait_for_key_with_return(
    events: dict, robot, prompt: str, *, allow_automatic_return: bool
) -> bool:
    """Wait for a key; automatic moveJoint remains a separate authorization."""
    events["exit_early"] = False
    events["return_to_start"] = False
    print("\n" + "=" * 50)
    print(prompt)
    if allow_automatic_return:
        print("  r  -> 归位到起始位置（moveJoint + 松吸盘）")
    else:
        print("  r  -> 已禁用（需另设 AUTOMATIC_RETURN_AUTHORIZED=1）")
    print("  -> -> 继续（开始下一轮 / 重置结束）")
    print("  Esc -> 终止整个评估")
    print("=" * 50)
    while not events["exit_early"] and not events["stop_recording"]:
        if events.get("return_to_start"):
            events["return_to_start"] = False
            if allow_automatic_return:
                return_to_start(robot)
                print("已归位，按 -> 继续")
            else:
                logging.warning("已忽略 r：自动归位未被单独授权")
                print("自动归位未授权；请用示教器安全归位后按 -> 继续")
        time.sleep(0.05)
    if events["stop_recording"]:
        return False
    events["exit_early"] = False
    return True


def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    init_logging(log_file=str(log_dir / "evaluate_split.log"))

    validate_runtime_settings()
    execute_actions = not DRY_RUN
    mode_name = "AUTHORIZED_EXECUTION" if execute_actions else "DRY_RUN_NO_ACTION_SEND"
    logging.warning("运行模式: %s", mode_name)
    logging.info(
        "拒绝阈值: workspace=%s..%s, xyz_step=%.3f m, J6_step=%.3f rad, "
        "J6_range=[%.3f, %.3f] rad, "
        "rot_step=%.3f rad, IK_joint_step=%.3f rad, observation_age=%.1f ms, "
        "hold_locked_ee_pose=%s, gripper_chunk_lookahead=%s, "
        "gripper_chunk_close_score=%.1f, gripper_chunk_open_score=%.1f, "
        "motion_chunk_lookahead=%s, motion_chunk_min_z_drop=%.3f m",
        WORKSPACE_MIN_M,
        WORKSPACE_MAX_M,
        MAX_EE_STEP_M,
        MAX_J6_STEP_RAD,
        J6_TARGET_MIN_RAD,
        J6_TARGET_MAX_RAD,
        MAX_EE_ROT_STEP_RAD,
        MAX_IK_JOINT_STEP_RAD,
        MAX_OBSERVATION_AGE_MS,
        HOLD_LOCKED_EE_POSE,
        USE_GRIPPER_CHUNK_LOOKAHEAD,
        GRIPPER_CHUNK_CLOSE_SCORE,
        GRIPPER_CHUNK_OPEN_SCORE,
        USE_MOTION_CHUNK_LOOKAHEAD,
        MOTION_CHUNK_MIN_Z_DROP_M,
    )

    # 1. 以训练数据集的 FPS 为唯一控制时间基准。
    training_repo_id, training_root = resolve_local_dataset_path(
        TRAINING_DATASET_PATH, must_exist=True
    )
    training_metadata = LeRobotDatasetMetadata(training_repo_id, root=training_root)
    control_fps = int(training_metadata.fps)
    if control_fps != EXPECTED_CONTROL_FPS:
        raise ValueError(
            f"新视角纯 ACT 数据应为 {EXPECTED_CONTROL_FPS} FPS，"
            f"但 {training_root} 是 {control_fps} FPS"
        )
    logging.info("训练数据契约来源: %s", training_root)

    # 2. 相机 + 机器人（本地直连）
    camera_config = {
        "handeye": OpenCVCameraConfig(
            index_or_path=HANDEYE_DEV,
            width=640,
            height=480,
            fps=HANDEYE_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
        "fixed": OpenCVCameraConfig(
            index_or_path=FIXED_DEV,
            width=640,
            height=480,
            fps=FIXED_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
    }
    robot = AuboI10Robot(
        AuboI10Config(cameras=camera_config, control_fps=control_fps)
    )

    # 3. 先按 8 月 3 日方式限步/工作空间裁剪，再让 ThinSafetyGate 拒绝裁剪后仍非法的命令。
    safety_gate = ThinSafetyGate(
        ThinSafetyGateLimits(
            workspace_min_m=WORKSPACE_MIN_M,
            workspace_max_m=WORKSPACE_MAX_M,
            max_ee_step_m=MAX_EE_STEP_M,
            max_ee_speed_mps=MAX_EE_STEP_M * control_fps,
            max_j6_step_rad=MAX_J6_STEP_RAD,
            max_observation_age_ms=MAX_OBSERVATION_AGE_MS,
            max_chunk_age_ms=MAX_ACTION_CHUNK_AGE_MS,
            control_fps=control_fps,
            require_previous_tcp=True,
            require_ik=True,
        )
    )
    ee_mode_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            AuboEEBoundsAndSafety(
                end_effector_bounds={
                    "min": list(WORKSPACE_MIN_M),
                    "max": list(WORKSPACE_MAX_M),
                },
                max_ee_step_m=MAX_EE_STEP_M,
            ),
            AuboSetEEMode(ee_mode="abs_j6yaw"),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # 4. 用训练数据集 features 创建评估数据集。评估结果不得被静默覆盖；
    # 已存在时要求调用者通过 EVAL_DATASET_PATH 选择一个新名称。
    eval_repo_id, eval_root = resolve_local_dataset_path(
        LOCAL_EVAL_DATASET_PATH, must_exist=False
    )
    if eval_root.exists():
        raise FileExistsError(
            f"评估数据集已存在: {eval_root}；请设置新的 EVAL_DATASET_PATH"
        )
    trace_path = resolve_trace_path(eval_root)
    if trace_path.exists():
        raise FileExistsError(
            f"推理 trace 已存在: {trace_path}；请设置新的 EVAL_DATASET_PATH 或 INFERENCE_TRACE_PATH"
        )

    # 5. 连接机器人 + 服务器。连接失败也必须进入统一清理路径。
    listener = None
    sock = None
    dataset = None
    trace_writer = None
    episode_idx = 0
    attempt_idx = 0
    try:
        robot.connect()
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")
        listener, events = init_keyboard_listener()

        logging.info("连接推理服务器 %s:%d ...", SERVER_HOST, SERVER_PORT)
        sock = socket.create_connection(
            (SERVER_HOST, SERVER_PORT), timeout=CONNECT_TIMEOUT_S
        )
        sock.settimeout(INFERENCE_TIMEOUT_S)
        logging.info("服务器已连接，单次响应超时 %.2f s", INFERENCE_TIMEOUT_S)

        print("\n" + "=" * 50)
        print("拆分式评估操作指南:")
        print(f"  当前模式: {mode_name}")
        if not execute_actions:
            print("  安全保证: 本进程不会调用 robot.send_action")
        print("  -> (右箭头): 开始/结束当前 episode")
        print("  ← (左箭键): 结束并重录当前 episode")
        if execute_actions and AUTOMATIC_RETURN_AUTHORIZED:
            print("  r:          归位到起始位置（已单独授权）")
        else:
            print("  r:          自动归位禁用")
        print("  Esc:        终止整个评估")
        print("=" * 50)

        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"准备执行 episode {episode_idx + 1} / {NUM_EPISODES}")
            if execute_actions:
                robot.disable_servo_mode()
            if not wait_for_key_with_return(
                events,
                robot,
                f"按 -> 开始执行 episode {episode_idx + 1} / {NUM_EPISODES}",
                allow_automatic_return=(
                    execute_actions and AUTOMATIC_RETURN_AUTHORIZED
                ),
            ):
                break

            # 每轮必须从训练分布的统一起始关节位开始。
            assert_start_pose(read_current_joints_deg(robot))
            if execute_actions and not robot.suction_release():
                raise RuntimeError("无法确认吸盘已释放，拒绝开始真实执行")

            # 每轮开始：重置服务器策略队列 + 本地处理器。
            try:
                send_msg(sock, {"cmd": "reset"})
                resp = recv_msg(sock)
            except (OSError, TimeoutError) as exc:
                raise RuntimeError("推理服务器 reset 超时或连接中断") from exc
            if resp is None or "error" in resp:
                raise RuntimeError(f"服务器 reset 失败: {resp}")
            ee_mode_processor.reset()

            # Delay creation until hardware, start pose, and server reset all pass;
            # failed preflight therefore leaves no empty dataset blocking a retry.
            if dataset is None:
                dataset = LeRobotDataset.create(
                    repo_id=eval_repo_id,
                    root=eval_root,
                    fps=control_fps,
                    features=training_metadata.features,
                    robot_type=robot.name,
                    use_videos=True,
                    image_writer_threads=4,
                )
                trace_writer = JsonlTraceWriter(trace_path)

            log_say(f"开始推理 episode {episode_idx + 1} / {NUM_EPISODES}")
            print("推理中... 按 -> 结束本轮, 按 ← 重录, 按 Esc 终止")
            write_trace(
                trace_writer,
                attempt_trace_record(
                    record_type="attempt_start",
                    episode_index=episode_idx,
                    attempt_index=attempt_idx,
                ),
            )
            attempt_closed = False
            try:
                run_episode(
                    robot=robot,
                    events=events,
                    fps=control_fps,
                    sock=sock,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    ee_mode_processor=ee_mode_processor,
                    robot_observation_processor=robot_observation_processor,
                    safety_gate=safety_gate,
                    execute_actions=execute_actions,
                    trace_writer=trace_writer,
                    episode_index=episode_idx,
                    attempt_index=attempt_idx,
                )

                if execute_actions:
                    robot.disable_servo_mode()

                if events["rerecord_episode"]:
                    dataset.clear_episode_buffer()
                    write_trace(
                        trace_writer,
                        attempt_trace_record(
                            record_type="attempt_end",
                            episode_index=episode_idx,
                            attempt_index=attempt_idx,
                            disposition="rerecorded",
                        ),
                    )
                    attempt_closed = True
                    log_say("重新执行本轮 episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    attempt_idx += 1
                    continue

                dataset.save_episode()
                write_trace(
                    trace_writer,
                    attempt_trace_record(
                        record_type="attempt_end",
                        episode_index=episode_idx,
                        attempt_index=attempt_idx,
                        disposition="saved",
                    ),
                )
                attempt_closed = True
                log_say(f"Episode {episode_idx + 1} 已保存")
                episode_idx += 1
                attempt_idx = 0
            except Exception as exc:
                if not attempt_closed:
                    write_trace(
                        trace_writer,
                        attempt_trace_record(
                            record_type="attempt_end",
                            episode_index=episode_idx,
                            attempt_index=attempt_idx,
                            disposition="aborted",
                            error=str(exc),
                        ),
                    )
                raise

            if episode_idx >= NUM_EPISODES or events["stop_recording"]:
                break

            if execute_actions:
                robot.disable_servo_mode()
            log_say("请重置环境；自动归位需要独立授权")

    finally:
        log_say("评估结束")
        if execute_actions and robot.is_connected:
            robot.disable_servo_mode()
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if robot.is_connected:
            robot.disconnect()
        if listener:
            listener.stop()
        if trace_writer is not None:
            trace_writer.close()
        if dataset is not None:
            dataset.finalize()
            print(f"\n评估数据已保存至: {eval_root}")
            print(f"结构化推理 trace 已保存至: {trace_path}")
        else:
            print("\n预检未通过或用户在开始前退出；未创建评估数据集")
        print(f"共执行 {episode_idx} 个 episodes")


if __name__ == "__main__":
    main()
