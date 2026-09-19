#!/usr/bin/env python
"""拆分式推理的服务端，跑在 GPU 机上。

本机（工作站）跑 evaluate_split.py 做控制循环，通过 Tailscale 把观测发给本服务，
本服务加载 ACT 策略做纯模型推理，每次返回当前要执行的一步动作。

协议：TCP + 4 字节大端长度前缀 + pickle（信任的 tailnet，pickle 可接受）。
客户端每帧发一条消息：
  - {"cmd": "reset"}                      -> 服务端清空策略队列，回 {"ok": True}
  - {"cmd": "predict", "obs": {...}}      -> 推理，回 actions + 可选结构化 trace
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
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.bamboo_sorting.act_input_contract import (
    ActStateInputContract,
    STATE_INPUT_CONTRACT_FILENAME,
    STATE_KEY,
    apply_state_input_contract,
    load_state_input_contract,
)
from lerobot.bamboo_sorting.contracts import ACTION_FIELD_NAMES
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import init_logging

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/bamboo_newview_act/best")
HOST = "0.0.0.0"  # 监听所有接口；客户端经 Tailscale IP (100.88.143.45) 连入
PORT = int(os.environ.get("SERVER_PORT", "5555"))
DEFAULT_TEMPORAL_ENSEMBLE_COEFF = 0.01
DEFAULT_ENSEMBLE_GRIPPER_MODE = "ensemble"
GRIPPER_ACTION_INDEX = ACTION_FIELD_NAMES.index("ee.gripper_pos")


def resolve_checkpoint_state_contract(
    model_path: Path,
    expected_state_shape: tuple[int, ...],
) -> ActStateInputContract | None:
    """Require an explicit contract for every non-legacy ACT state shape."""

    contract_path = model_path / STATE_INPUT_CONTRACT_FILENAME
    if contract_path.is_file():
        contract = load_state_input_contract(model_path)
        if expected_state_shape != (contract.model_width,):
            raise ValueError(
                "checkpoint state contract does not match policy config: "
                f"contract={contract.model_width}, config={expected_state_shape}"
            )
        return contract
    if expected_state_shape == (13,):
        return None
    raise FileNotFoundError(
        f"non-legacy {expected_state_shape} checkpoint is missing required {contract_path}"
    )


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


def get_ensemble_gripper_mode() -> str:
    """Choose whether temporal aggregation also averages the discrete gripper dimension."""
    value = os.environ.get(
        "ENSEMBLE_GRIPPER_MODE", DEFAULT_ENSEMBLE_GRIPPER_MODE
    ).strip().lower()
    if value not in {"ensemble", "latest"}:
        raise ValueError("ENSEMBLE_GRIPPER_MODE 必须是 ensemble 或 latest")
    return value


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
        self.sample_latent = os.environ.get("ACT_LATENT_SAMPLE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.latent_search_k = int(os.environ.get("ACT_LATENT_SEARCH", "0"))
        if self.latent_search_k < 0:
            raise ValueError("ACT_LATENT_SEARCH must be >= 0")
        self.policy.model.sample_latent_at_inference = self.sample_latent
        self._latent_search_committed = False
        self._latent_search_attempted = False
        self._last_search_gripper = None
        self.device = torch.device(self.policy.config.device)
        expected_state_shape = tuple(self.policy.config.input_features[STATE_KEY].shape)
        self.state_input_contract = resolve_checkpoint_state_contract(
            Path(MODEL_PATH),
            expected_state_shape,
        )
        if self.state_input_contract is None:
            logging.warning(
                "legacy 13D checkpoint has no %s; accepting full state without ablation",
                STATE_INPUT_CONTRACT_FILENAME,
            )

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=MODEL_PATH,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        # Temporal ensembling: every tick predicts an overlapping action chunk,
        # then fuses them online. This avoids hard boundaries between chunks.
        coeff = get_temporal_ensemble_coeff()
        self.ensemble_gripper_mode = get_ensemble_gripper_mode()
        if coeff is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            self.policy.config.temporal_ensemble_coeff = coeff
            self.policy.config.n_action_steps = 1  # ensemble 要求 n_action_steps=1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(coeff, self.policy.config.chunk_size)
            self.policy.reset()
            self.ensemble = True
            logging.info(
                "temporal ensembling 开启: coeff=%s, gripper_mode=%s",
                coeff,
                self.ensemble_gripper_mode,
            )
        else:
            self.ensemble = False
            raw_n_action_steps = os.environ.get("N_ACTION_STEPS")
            if raw_n_action_steps:
                n_action_steps = int(raw_n_action_steps)
                chunk_size = int(self.policy.config.chunk_size)
                if n_action_steps < 1 or n_action_steps > chunk_size:
                    raise ValueError(
                        f"N_ACTION_STEPS must be in [1, {chunk_size}], got {n_action_steps}"
                    )
                self.policy.config.n_action_steps = n_action_steps
                logging.info("queue n_action_steps override=%s chunk_size=%s", n_action_steps, chunk_size)
        self.n_action_steps = self.policy.config.n_action_steps
        self.chunk_blend_steps = int(os.environ.get("CHUNK_BLEND_STEPS", "0"))
        if self.chunk_blend_steps < 0:
            raise ValueError("CHUNK_BLEND_STEPS must be >= 0")
        if self.chunk_blend_steps > self.n_action_steps:
            raise ValueError("CHUNK_BLEND_STEPS cannot exceed n_action_steps")
        self._last_selected = None
        self.prediction_sequence = 0
        self.queue_action_index = 0

        self.task = "抓取竹条"
        self.robot_type = "aubo_i10"
        logging.info(
            f"纯 ACT 就绪: device={self.device}, n_action_steps={self.n_action_steps}, "
            f"ensemble={self.ensemble}, "
            f"chunk_blend_steps={self.chunk_blend_steps}, "
            f"latent_sample={self.sample_latent}, "
            f"latent_search_k={self.latent_search_k}, "
            f"ensemble_gripper_mode={self.ensemble_gripper_mode}, "
            f"state_contract={self.state_input_contract.variant if self.state_input_contract else 'legacy_full'}"
        )

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.prediction_sequence = 0
        self.queue_action_index = 0
        self._last_selected = None
        self._latent_search_committed = False
        self._latent_search_attempted = False
        self._last_search_gripper = None

    def _select_action_with_evidence(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Mirror ACT selection while retaining the pre-ensemble/pre-queue model chunk."""
        self.policy.eval()
        predicted_chunk = None
        if self.policy.config.temporal_ensemble_coeff is not None:
            predicted_chunk = self.policy.predict_action_chunk(batch)
            selected = self.policy.temporal_ensembler.update(predicted_chunk)
            if getattr(self, "ensemble_gripper_mode", "ensemble") == "latest":
                selected = selected.clone()
                selected[:, GRIPPER_ACTION_INDEX] = predicted_chunk[
                    :, 0, GRIPPER_ACTION_INDEX
                ]
            return selected, predicted_chunk, 0

        if len(self.policy._action_queue) == 0:
            predicted_chunk = self.policy.predict_action_chunk(batch)[
                :, : self.policy.config.n_action_steps
            ]
            queued = predicted_chunk.transpose(0, 1).clone()
            if (
                self.chunk_blend_steps > 0
                and self._last_selected is not None
                and queued.shape[0] > 0
            ):
                blend = min(self.chunk_blend_steps, int(queued.shape[0]))
                last = self._last_selected
                for index in range(blend):
                    alpha = (index + 1) / (blend + 1)
                    queued[index] = (1.0 - alpha) * last + alpha * queued[index]
            self.policy._action_queue.extend(queued)
            self.queue_action_index = 0
        selected_index = self.queue_action_index
        selected = self.policy._action_queue.popleft()
        self.queue_action_index += 1
        self._last_selected = selected.detach().clone()
        return selected, predicted_chunk, selected_index

    def _to_numpy(self, value) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return self._to_numpy(value.get("action", next(iter(value.values()))))
        try:
            return np.asarray(value)
        except (TypeError, RuntimeError, ValueError):
            if hasattr(value, "action"):
                return self._to_numpy(value.action)
            raise

    def _postprocess_numpy(self, action: torch.Tensor) -> np.ndarray:
        processed = self.postprocessor(self._to_numpy(action) if not torch.is_tensor(action) else action.clone())
        if isinstance(processed, dict):
            processed = processed.get("action", next(iter(processed.values())))
        return np.asarray(self._to_numpy(processed), dtype=np.float32)

    def predict_with_trace(self, obs_raw: dict) -> tuple[np.ndarray, dict]:
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

        if self.state_input_contract is not None:
            obs_np = apply_state_input_contract(obs_np, self.state_input_contract)
        else:
            state = np.asarray(obs_np[STATE_KEY])
            if state.shape != (13,):
                raise ValueError(f"legacy ACT checkpoint expects 13D state; got {state.shape}")

        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                obs_np, self.device, task, robot_type
            )
            batch = self.preprocessor(batch)
            selected_normalized, predicted_chunk_normalized, selected_index = (
                self._select_action_with_evidence(batch)
            )
            selected_denormalized = self._postprocess_numpy(selected_normalized)
            predicted_chunk_denormalized = (
                self._postprocess_numpy(predicted_chunk_normalized)
                if predicted_chunk_normalized is not None
                else None
            )
            state_vec = np.asarray(self._to_numpy(obs_np[STATE_KEY]), dtype=np.float64).reshape(-1)
            if self.state_input_contract is not None:
                z_index = self.state_input_contract.model_names.index("ee.z")
                x_index = self.state_input_contract.model_names.index("ee.x")
            else:
                z_index, x_index = 8, 6
            current_xyz = state_vec[x_index : x_index + 3]
            current_z = float(state_vec[z_index])
            if (
                self.sample_latent
                and self.latent_search_k > 0
                and not self._latent_search_committed
                and not self._latent_search_attempted
                and predicted_chunk_denormalized is not None
                and current_z <= 0.12
            ):
                current_g = float(
                    np.max(predicted_chunk_denormalized[0, :, GRIPPER_ACTION_INDEX])
                )
                self._last_search_gripper = current_g
                self._latent_search_attempted = True
                if current_g < 20.0:
                    best_g = current_g
                    best_z = getattr(self.policy.model, "_infer_z", None)
                    best_chunk = predicted_chunk_normalized
                    original_z = best_z
                    for _ in range(self.latent_search_k):
                        self.policy.model._infer_z = None
                        candidate = self.policy.predict_action_chunk(batch)
                        candidate_denorm = self._postprocess_numpy(candidate)
                        score = float(
                            np.max(np.asarray(candidate_denorm)[..., GRIPPER_ACTION_INDEX])
                        )
                        if score > best_g:
                            best_g = score
                            best_z = self.policy.model._infer_z
                            best_chunk = candidate
                    self._last_search_gripper = best_g
                    if best_g >= 60.0 and best_z is not None:
                        self.policy.model._infer_z = best_z
                        self._latent_search_committed = True
                        self._last_search_gripper = best_g
                        if self.ensemble:
                            self.policy.temporal_ensembler.reset()
                            selected_normalized = self.policy.temporal_ensembler.update(
                                best_chunk
                            )
                            if self.ensemble_gripper_mode == "latest":
                                selected_normalized = selected_normalized.clone()
                                selected_normalized[:, GRIPPER_ACTION_INDEX] = best_chunk[
                                    :, 0, GRIPPER_ACTION_INDEX
                                ]
                        else:
                            selected_normalized = best_chunk[:, 0]
                        predicted_chunk_normalized = best_chunk
                        selected_denormalized = self._postprocess_numpy(
                            selected_normalized
                        )
                        predicted_chunk_denormalized = self._postprocess_numpy(
                            predicted_chunk_normalized
                        )
                    else:
                        self.policy.model._infer_z = original_z

        selected_normalized_np = self._to_numpy(selected_normalized)
        selected_denormalized = np.asarray(self._to_numpy(selected_denormalized), dtype=np.float32)
        if predicted_chunk_denormalized is not None:
            predicted_chunk_denormalized = np.asarray(
                self._to_numpy(predicted_chunk_denormalized), dtype=np.float32
            )
        trace = {
            "schema_version": "aubo_act_server_trace_v1",
            "prediction_sequence": self.prediction_sequence,
            "model_inference_performed": predicted_chunk_normalized is not None,
            "temporal_ensemble_enabled": bool(self.ensemble),
            "ensemble_gripper_mode": self.ensemble_gripper_mode,
            "latent_search_committed": bool(self._latent_search_committed),
            "latent_search_gripper": self._last_search_gripper,
            "queue_action_index": int(selected_index),
            "selected_normalized_action": selected_normalized_np.squeeze(0).tolist(),
            "selected_denormalized_action": selected_denormalized.squeeze(0).tolist(),
            "predicted_chunk_normalized_gripper": (
                predicted_chunk_normalized[0, :, GRIPPER_ACTION_INDEX].detach().cpu().tolist()
                if predicted_chunk_normalized is not None
                else None
            ),
            "predicted_chunk_denormalized_gripper": (
                predicted_chunk_denormalized[0, :, GRIPPER_ACTION_INDEX].tolist()
                if predicted_chunk_denormalized is not None
                else None
            ),
            "predicted_chunk_denormalized_action": (
                predicted_chunk_denormalized[0].tolist()
                if predicted_chunk_denormalized is not None
                else None
            ),
        }
        self.prediction_sequence += 1
        return selected_denormalized.squeeze(0), trace

    def predict(self, obs_raw: dict) -> np.ndarray:
        """Backward-compatible local API returning only the denormalized action."""
        action, _trace = self.predict_with_trace(obs_raw)
        return action

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
                        actions, trace = self.predict_with_trace(msg["obs"])
                        # Existing clients keep reading ``actions``; ``trace`` is additive.
                        send_msg(conn, {"actions": actions, "trace": trace})
                    else:
                        send_msg(conn, {"error": f"unknown cmd: {cmd}"})
                except Exception as e:
                    logging.exception("处理请求失败")
                    send_msg(conn, {"error": f"{e}\n{traceback.format_exc()}"})

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
