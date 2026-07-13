#!/usr/bin/env python
"""拆分式推理的客户端，跑在本地工作站。

机器人 + 相机都在本地；策略推理在 GPU 机上（inference_server.py），经 Tailscale 连接。
本脚本：取观测 -> 队列空时把观测发给服务器拿回一块 n_action_steps 个动作 ->
每帧从队列取一个动作 -> 加 ee_mode -> 下发给机器人 -> 写评估数据集。

操作同 evaluate.py：-> 开始/结束 episode，← 重录，Esc 全停。

前置：
  1. GPU 机上先启动 inference_server.py（见该文件头注释）。
  2. 本地机器人上电、相机就位（by-id + MJPG，同 record.py）。
  3. 本地有聚合数据集 phone_auboi10_full（取 features；aggregate.py 产物）。
"""

import logging
import pickle
import socket
import struct
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

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
from lerobot.robots.aubo_i10.robot_processor import AuboSetEEMode
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "抓取竹条"
TRAINING_DATASET_PATH = "./datasets/phone_auboi10_full"  # 取 features（本地）
LOCAL_EVAL_DATASET_PATH = "./datasets/phone_auboi10_split_eval"

# GPU 机（Tailscale）
SERVER_HOST = "100.88.143.45"
SERVER_PORT = 5555

# 相机用稳定的 by-id 路径 + MJPG（本机直连相机，同 record.py）
HANDEYE_DEV = "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"
FIXED_DEV = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"


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


def recv_msg(sock):
    header = recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
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
):
    action_queue = deque()  # 兼容 queue 模式：服务器返回多动作时缓存
    dt = 1.0 / fps
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # 1. 观测
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # 2. 队列空 -> 向服务器请求动作（ensemble 模式返回 1 个，queue 模式返回多个）
        if len(action_queue) == 0:
            msg = {
                "cmd": "predict",
                "obs": encode_obs(observation_frame, single_task, robot.robot_type),
            }
            send_msg(sock, msg)
            resp = recv_msg(sock)
            if resp is None or "error" in resp:
                raise RuntimeError(f"服务器返回错误: {resp}")
            actions = np.atleast_2d(np.asarray(resp["actions"], dtype=np.float32))
            for row in actions:
                act_tensor = torch.from_numpy(row).unsqueeze(0)  # (1, action_dim)
                action_queue.append(make_robot_action(act_tensor, dataset.features))

        # 3. 取一个动作 -> 加 ee_mode -> 下发
        act = action_queue.popleft()
        robot_action = ee_mode_processor((act, obs))
        robot.send_action(robot_action)

        # 4. 写评估数据集（obs + action + task，便于回放）
        action_frame = build_dataset_frame(dataset.features, act, prefix=ACTION)
        dataset.add_frame({**observation_frame, **action_frame, "task": single_task})

        # 5. 维持帧率
        timestamp = time.perf_counter() - start_episode_t
        elapsed = time.perf_counter() - start_loop_t
        if elapsed < dt:
            precise_sleep(dt - elapsed)


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


def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    init_logging(log_file=str(log_dir / "evaluate_split.log"))

    # 1. 相机 + 机器人（本地直连）
    camera_config = {
        "handeye": OpenCVCameraConfig(index_or_path=HANDEYE_DEV, width=640, height=480, fps=FPS, fourcc="MJPG"),
        "fixed": OpenCVCameraConfig(index_or_path=FIXED_DEV, width=640, height=480, fps=FPS, fourcc="MJPG"),
    }
    robot = AuboI10Robot(AuboI10Config(cameras=camera_config))

    # 2. ee_mode 处理器（策略输出不含 ee_mode 字符串，补回去走 abs_j6yaw，同 evaluate.py）
    ee_mode_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[AuboSetEEMode(ee_mode="abs_j6yaw")],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # 3. 训练数据集 features（建评估数据集 + make_robot_action 用）
    training_metadata = LeRobotDatasetMetadata(TRAINING_DATASET_PATH)
    # 自动清掉上次没干净退出留下的评估数据集外壳，避免 FileExistsError
    import shutil
    from lerobot.utils.constants import HF_LEROBOT_HOME
    eval_root = HF_LEROBOT_HOME / LOCAL_EVAL_DATASET_PATH
    if eval_root.exists():
        print(f"清掉旧的评估数据集外壳: {eval_root}")
        shutil.rmtree(eval_root)
    dataset = LeRobotDataset.create(
        repo_id=LOCAL_EVAL_DATASET_PATH,
        fps=FPS,
        features=training_metadata.features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # 4. 连接机器人 + 服务器
    robot.connect()
    listener, events = init_keyboard_listener()

    logging.info(f"连接推理服务器 {SERVER_HOST}:{SERVER_PORT} ...")
    sock = socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=30)
    sock.settimeout(None)
    logging.info("服务器已连接")

    episode_idx = 0
    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("\n" + "=" * 50)
        print("拆分式评估操作指南:")
        print("  -> (右箭头): 开始/结束当前 episode")
        print("  ← (左箭键): 结束并重录当前 episode")
        print("  Esc:        终止整个评估")
        print("=" * 50)

        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"准备执行 episode {episode_idx + 1} / {NUM_EPISODES}")
            if not wait_for_key(events, f"按 -> 开始执行 episode {episode_idx + 1} / {NUM_EPISODES}"):
                break

            # 每轮开始：重置服务器策略队列 + 本地处理器
            send_msg(sock, {"cmd": "reset"})
            resp = recv_msg(sock)
            if resp is None or "error" in resp:
                raise RuntimeError(f"服务器 reset 失败: {resp}")
            ee_mode_processor.reset()

            log_say(f"开始推理 episode {episode_idx + 1} / {NUM_EPISODES}")
            print("推理中... 按 -> 结束本轮, 按 ← 重录, 按 Esc 终止")

            run_episode(
                robot=robot,
                events=events,
                fps=FPS,
                sock=sock,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                ee_mode_processor=ee_mode_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if events["rerecord_episode"]:
                log_say("重新执行本轮 episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            log_say(f"Episode {episode_idx + 1} 已保存")
            episode_idx += 1

            if episode_idx >= NUM_EPISODES or events["stop_recording"]:
                break

            robot.disable_servo_mode()
            log_say("请重置环境（可使用示教器移动机器人）")
            if not wait_for_key(events, "重置环境完成后，按 -> 开始下一轮推理"):
                break

    finally:
        log_say("评估结束")
        try:
            sock.close()
        except Exception:
            pass
        robot.disconnect()
        if listener:
            listener.stop()
        dataset.finalize()
        print(f"\n评估数据已保存至: {LOCAL_EVAL_DATASET_PATH}")
        print(f"共执行 {episode_idx} 个 episodes")


if __name__ == "__main__":
    main()
