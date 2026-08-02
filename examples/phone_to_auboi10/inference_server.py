#!/usr/bin/env python
"""拆分式推理的服务端，跑在 GPU 机上。

本机（工作站）跑 evaluate_split.py 做控制循环，通过 Tailscale 把观测发给本服务，
本服务加载 ACT 策略做纯模型推理，每次返回当前要执行的一步动作。

协议：TCP + 4 字节大端长度前缀 + pickle（信任的 tailnet，pickle 可接受）。
客户端每帧发一条消息：
  - {"cmd": "reset"}                      -> 服务端清空策略队列，回 {"ok": True}
  - {"cmd": "predict", "obs": {...}}      -> 推理，回 {"actions": np.ndarray(8)}
obs 字段：observation.state (np), observation.images.* (JPEG bytes), task, robot_type

启动：
  ssh gpu 'cd ~/program/lerobot-aubo/examples/phone_to_auboi10 && \
    HF_HUB_OFFLINE=1 nohup ../../.venv/bin/python inference_server.py > logs/inf_server.log 2>&1 &'
"""

import logging
import os
import pickle
import socket
import struct
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import init_logging

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/bamboo_newview_act/best")
HOST = "0.0.0.0"  # 监听所有接口；客户端经 Tailscale IP (100.88.143.45) 连入
PORT = 5555
DEFAULT_TEMPORAL_ENSEMBLE_COEFF = 0.01


def get_temporal_ensemble_coeff() -> float | None:
    """Return the configured ACT smoothing coefficient.

    Temporal ensembling is the standard ACT mechanism for reconciling
    overlapping action chunks.  It is enabled by default for physical-robot
    inference, while an explicit ``TEMPORAL_ENSEMBLE_COEFF=0`` keeps the old
    low-latency queue mode available for controlled comparisons.
    """
    raw_value = os.environ.get("TEMPORAL_ENSEMBLE_COEFF")
    if raw_value is None:
        return DEFAULT_TEMPORAL_ENSEMBLE_COEFF
    value = raw_value.strip().lower()
    if value in ("", "0", "false", "no", "none", "off"):
        return None
    coeff = float(value)
    if coeff <= 0:
        raise ValueError("TEMPORAL_ENSEMBLE_COEFF 必须大于 0，或显式设为 0 关闭")
    return coeff


def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock):
    header = recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = recv_exactly(sock, length)
    if payload is None:
        return None
    return pickle.loads(payload)


def recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class InferenceServer:
    def __init__(self):
        logging.info(f"加载策略与处理器: {MODEL_PATH}")
        self.policy = ACTPolicy.from_pretrained(MODEL_PATH)
        self.policy.eval()
        self.device = torch.device(self.policy.config.device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=MODEL_PATH,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        # Temporal ensembling: every tick predicts an overlapping action chunk,
        # then fuses them online. This avoids hard boundaries between chunks.
        coeff = get_temporal_ensemble_coeff()
        if coeff is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            self.policy.config.temporal_ensemble_coeff = coeff
            self.policy.config.n_action_steps = 1  # ensemble 要求 n_action_steps=1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(coeff, self.policy.config.chunk_size)
            self.policy.reset()
            self.ensemble = True
            logging.info(f"temporal ensembling 开启: coeff={coeff}")
        else:
            self.ensemble = False
        self.n_action_steps = self.policy.config.n_action_steps

        self.task = "抓取竹条"
        self.robot_type = "aubo_i10"
        logging.info(
            f"纯 ACT 就绪: device={self.device}, n_action_steps={self.n_action_steps}, "
            f"ensemble={self.ensemble}"
        )

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict(self, obs_raw: dict) -> np.ndarray:
        """obs_raw: {observation.state: np, observation.images.*: jpg bytes, task, robot_type}"""
        task = obs_raw.pop("task", "")
        robot_type = obs_raw.pop("robot_type", "")

        # 重建 numpy 观测：JPEG bytes -> HWC uint8
        obs_np = {}
        for name, value in obs_raw.items():
            if isinstance(value, (bytes, bytearray)):
                arr = np.frombuffer(value, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC uint8
                if img is None:
                    raise ValueError(f"解码图像失败: {name}")
                obs_np[name] = img
            else:
                obs_np[name] = np.asarray(value)

        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                obs_np, self.device, task, robot_type
            )
            batch = self.preprocessor(batch)
            # select_action 一次返回 1 个动作：
            # - ensemble 模式：每帧推理 + 在线 ensemble，返回平滑动作
            # - queue 模式：首次推理填队列，之后按 n_action_steps 消耗（队列在 policy 内部管理）
            a = self.policy.select_action(batch)
            a = self.postprocessor(a)
        return a.squeeze(0).cpu().numpy()  # (action_dim,)

    def handle_client(self, conn):
        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                cmd = msg.get("cmd")
                try:
                    if cmd == "reset":
                        self.reset()
                        send_msg(conn, {"ok": True})
                    elif cmd == "predict":
                        actions = self.predict(msg["obs"])
                        send_msg(conn, {"actions": actions})
                    else:
                        send_msg(conn, {"error": f"unknown cmd: {cmd}"})
                except Exception as e:
                    logging.exception("处理请求失败")
                    send_msg(conn, {"error": str(e)})

    def serve_forever(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        logging.info(f"监听 {HOST}:{PORT}")
        while True:
            conn, addr = srv.accept()
            logging.info(f"客户端连入: {addr}")
            try:
                self.handle_client(conn)
            except Exception:
                logging.exception("连接异常")
            logging.info(f"客户端断开: {addr}")


def main():
    Path("logs").mkdir(exist_ok=True)
    init_logging(log_file="logs/inf_server.log")
    server = InferenceServer()
    server.serve_forever()


if __name__ == "__main__":
    main()
