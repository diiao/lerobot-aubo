#!/usr/bin/env python
"""拆分式推理的服务端，跑在 GPU 机上。

本机（工作站）跑 evaluate_split.py 做控制循环，通过 Tailscale 把观测发给本服务，
本服务加载 ACT 策略做推理，返回一整块 n_action_steps 个动作。

协议：TCP + 4 字节大端长度前缀 + pickle（信任的 tailnet，pickle 可接受）。
客户端每帧发一条消息：
  - {"cmd": "reset"}                      -> 服务端清空策略队列，回 {"ok": True}
  - {"cmd": "predict", "obs": {...}}      -> 推理，回 {"actions": np.ndarray(n,8)}
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

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import init_logging

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/phone_auboi10")
TRAINING_DATASET_PATH = "./datasets/bamboo_full_shift"  # 取 stats
HOST = "0.0.0.0"  # 监听所有接口；客户端经 Tailscale IP (100.88.143.45) 连入
PORT = 5555

# --- 启发式夹爪触发 (绕过模型记忆) ---
# 诊断结论: 模型把"触发吸盘"绑死在训练帧像素上, eval 帧位姿到位(xyz<2cm)但像素有
# 结构差异导致不触发。state 可靠, 所以用 EE 位姿状态机直接决定 gripper, 模型只管 EE 轨迹。
# 阈值从训练 fire(0->100)/release(100->0) 转换帧的 EE 位姿统计得出。
# env HEURISTIC_GRIPPER=0 可关闭 (回退到纯模型)。
HEURISTIC_GRIPPER = os.environ.get("HEURISTIC_GRIPPER", "1").strip().lower() not in ("", "0", "false", "no")
FIRE_XY = (0.592, -0.407)   # 训练 fire 帧 xy 均值
FIRE_XY_R = 0.07            # xy 半径 (m)
FIRE_Z_MAX = 0.125          # fire z max 0.101 + margin; eval z-min 0.106~0.117 全覆盖
RELEASE_X_MAX = 0.30        # release x ~0.15 (排除 1 个 0.606 outlier)
RELEASE_Y_MAX = -0.50       # release y < -0.50
RELEASE_Z_MAX = 0.17        # release z max 0.151 + margin

# --- 脚本化最终抓取 (Path A) ---
# 诊断结论: 启发式能触发吸盘, 但模型轨迹不精确--eval 里 z 只降到 0.118 (物体在 0.10),
# 且到物体 xy 时已经提起 (z~0.18), cup 从没同时到达 "物体 xy + 低 z"。根因同 gripper:
# 小数据背诵, 关键的 2cm 下扎被平均掉。解法: 模型只负责 "带到 fire 区", 进区后脚本接管
# xyz 精确下降->吸->提起->交还模型做 "移到落点"。
# abs_j6yaw 下 action[1:4] 是绝对 EE xyz (见 aubo_i10._send_position_j6yaw), 直接覆盖即可;
# rotvec(4:7)/j6(0) 保持模型输出 (诊断显示朝向 OK, 只有 xyz 偏)。
# env SCRIPTED_GRASP=0 可关闭 (回退到纯启发式 gripper)。
SCRIPTED_GRASP = os.environ.get("SCRIPTED_GRASP", "1").strip().lower() not in ("", "0", "false", "no")
GRASP_XYZ = tuple(float(v) for v in os.environ.get("GRASP_XYZ", "0.592,-0.407,0.098").split(","))  # 训练 fire 帧位姿均值 = 物体位姿
LIFT_XYZ = tuple(float(v) for v in os.environ.get("LIFT_XYZ", "0.592,-0.407,0.20").split(","))      # 提起高度 (清桌面)
DESCEND_FRAMES = int(os.environ.get("DESCEND_FRAMES", "15"))  # ~0.5s 降到物体
HOLD_FRAMES = int(os.environ.get("HOLD_FRAMES", "10"))        # ~0.3s 吸牢
LIFT_FRAMES = int(os.environ.get("LIFT_FRAMES", "15"))        # ~0.5s 提起


def _lerp3(p0, p1, t):
    t = max(0.0, min(1.0, t))
    return (p0[0] + (p1[0] - p0[0]) * t,
            p0[1] + (p1[1] - p0[1]) * t,
            p0[2] + (p1[2] - p0[2]) * t)


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

        metadata = LeRobotDatasetMetadata(TRAINING_DATASET_PATH)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=MODEL_PATH,
            dataset_stats=metadata.stats,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.n_action_steps = self.policy.config.n_action_steps

        # Temporal ensembling: 推理时启用（不用重训）。coeff!=None 时 select_action 走 ensemble
        # 路径：每帧推理 + 在线 ensemble，返回 1 个平滑动作（解决 chunk 边界跳变）。
        # 设了 TEMPORAL_ENSEMBLE_COEFF (默认 0.01) 就开；留空则走原 queue 模式。
        coeff = os.environ.get("TEMPORAL_ENSEMBLE_COEFF", "0.01").strip()
        if coeff:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            c = float(coeff)
            self.policy.config.temporal_ensemble_coeff = c
            self.policy.config.n_action_steps = 1  # ensemble 要求 n_action_steps=1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(c, self.policy.config.chunk_size)
            self.policy.reset()
            self.ensemble = True
            logging.info(f"temporal ensembling 开启: coeff={c}")
        else:
            self.ensemble = False

        self.task = "抓取竹条"
        self.robot_type = "aubo_i10"
        self.holding = False  # 启发式 latch: once 抓住, 持续到 release 区
        # Path A 脚本化抓取状态机: approach -> descend -> hold -> lift -> holding -> (release) -> approach
        self.phase = "approach"
        self.phase_frame = 0
        self.start_xyz = None  # descend 起点实际 EE xyz
        logging.info(
            f"就绪: device={self.device}, n_action_steps={self.n_action_steps}, ensemble={self.ensemble}, "
            f"heuristic_gripper={HEURISTIC_GRIPPER}, scripted_grasp={SCRIPTED_GRASP}"
        )

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.holding = False
        self.phase = "approach"
        self.phase_frame = 0
        self.start_xyz = None

    def _heuristic_gripper(self, state: np.ndarray) -> float:
        """EE 位姿状态机决定 gripper (0=off / 100=suction on)。
        state: 13-dim, ee.xyz 在 [6,7,8]。模型只管 EE 轨迹, gripper 由此接管。"""
        x, y, z = float(state[6]), float(state[7]), float(state[8])
        in_fire = (z < FIRE_Z_MAX
                   and abs(x - FIRE_XY[0]) < FIRE_XY_R
                   and abs(y - FIRE_XY[1]) < FIRE_XY_R)
        in_release = (z < RELEASE_Z_MAX
                      and x < RELEASE_X_MAX
                      and y < RELEASE_Y_MAX)
        if not self.holding and in_fire:
            self.holding = True
            return 100.0
        if self.holding and in_release:
            self.holding = False
            return 0.0
        if self.holding:
            return 100.0  # 提/移/持
        return 0.0  # 接近中

    def _scripted_override(self, a: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Path A 脚本化最终抓取。
        模型把 EE 带进 fire 区后, 脚本接管 xyz 精确下降->吸->提起->交还模型移到落点。
        a: 8-dim action (abs_j6yaw: [j6_t, x, y, z, wx, wy, wz, gripper])。
        state: 13-dim, ee.xyz 在 [6,7,8]。rotvec/j6 不动 (朝向 OK), 只覆盖 xyz + gripper。"""
        x, y, z = float(state[6]), float(state[7]), float(state[8])
        in_fire = (z < FIRE_Z_MAX
                   and abs(x - FIRE_XY[0]) < FIRE_XY_R
                   and abs(y - FIRE_XY[1]) < FIRE_XY_R)
        in_release = (z < RELEASE_Z_MAX
                      and x < RELEASE_X_MAX
                      and y < RELEASE_Y_MAX)

        # --- 脚本接管阶段: 覆盖 xyz + gripper=100 ---
        if self.phase == "descend":
            xyz = _lerp3(self.start_xyz, GRASP_XYZ, self.phase_frame / DESCEND_FRAMES)
            a[1], a[2], a[3] = xyz
            a[7] = 100.0
            self.phase_frame += 1
            if self.phase_frame >= DESCEND_FRAMES:
                self.phase = "hold"; self.phase_frame = 0
                logging.info("[scripted] descend->hold")
            return a
        if self.phase == "hold":
            a[1], a[2], a[3] = GRASP_XYZ
            a[7] = 100.0
            self.phase_frame += 1
            if self.phase_frame >= HOLD_FRAMES:
                self.phase = "lift"; self.phase_frame = 0
                logging.info("[scripted] hold->lift")
            return a
        if self.phase == "lift":
            xyz = _lerp3(GRASP_XYZ, LIFT_XYZ, self.phase_frame / LIFT_FRAMES)
            a[1], a[2], a[3] = xyz
            a[7] = 100.0
            self.phase_frame += 1
            if self.phase_frame >= LIFT_FRAMES:
                self.phase = "holding"; self.holding = True
                logging.info("[scripted] lift->holding (交还模型)")
            return a

        # --- approach / holding: 模型控制 xyz ---
        if self.phase == "approach" and in_fire and not self.holding:
            self.phase = "descend"; self.phase_frame = 0
            self.start_xyz = (x, y, z)
            logging.info(f"[scripted] approach->descend @ ({x:.3f},{y:.3f},{z:.3f})")
            a[1], a[2], a[3] = self.start_xyz  # t=0 = 起点, 无跳变
            a[7] = 100.0
            return a

        # holding: 模型移到落点 -> release
        if self.holding and in_release:
            self.holding = False
            self.phase = "approach"
            a[7] = 0.0
            logging.info("[scripted] release @ drop zone -> approach")
            return a
        a[7] = 100.0 if self.holding else 0.0
        return a

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

        # 注意: prepare_observation_for_inference 会原地把 obs_np 各 key 重绑成 CUDA tensor
        # 并 unsqueeze 加 batch 维 (state 变 (1,13)), 所以启发式用的 numpy state 必须在
        # 调它之前取出来 (它只重绑 dict key, 不改原 numpy 数组)。
        state_np = obs_np["observation.state"] if HEURISTIC_GRIPPER else None

        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                obs_np, self.device, task, robot_type
            )
            batch = self.preprocessor(batch)
            # select_action 一次返回 1 个动作：
            # - ensemble 模式：每帧推理 + 在线 ensemble，返回平滑动作
            # - queue 模式：首次推理填队列后 pop，每 10 帧推理一次（队列在 policy 内部管理）
            a = self.policy.select_action(batch)
            a = self.postprocessor(a)
        a = a.squeeze(0).cpu().numpy()  # (action_dim,)
        if HEURISTIC_GRIPPER:
            if SCRIPTED_GRASP:
                # Path A: 脚本化抓取接管 xyz+gripper, 绕过模型轨迹不精确
                a = self._scripted_override(a, state_np)
            else:
                # 纯启发式: 只用 EE 位姿状态机决定 gripper (action[7])
                a[7] = self._heuristic_gripper(state_np)
        return a

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
