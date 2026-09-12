from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.bamboo_sorting.act_input_contract import (
    DROP_GRIPPER_STATE_VARIANT,
    build_state_input_contract,
    write_state_input_contract,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "phone_to_auboi10"
    / "inference_server.py"
)
SPEC = importlib.util.spec_from_file_location("inference_server_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
inference_server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inference_server)


def _features() -> dict:
    return {
        "observation.state": {
            "shape": [4],
            "names": ["J1", "ee.x", "ee.wz", "gripper_pos"],
        }
    }


def test_legacy_13d_checkpoint_remains_compatible(tmp_path: Path) -> None:
    assert inference_server.resolve_checkpoint_state_contract(tmp_path, (13,)) is None


def test_nonlegacy_checkpoint_without_contract_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing required"):
        inference_server.resolve_checkpoint_state_contract(tmp_path, (12,))


def test_checkpoint_contract_must_match_policy_shape(tmp_path: Path) -> None:
    contract = build_state_input_contract(_features(), DROP_GRIPPER_STATE_VARIANT)
    write_state_input_contract(tmp_path, contract)
    assert inference_server.resolve_checkpoint_state_contract(tmp_path, (3,)) == contract
    with pytest.raises(ValueError, match="does not match"):
        inference_server.resolve_checkpoint_state_contract(tmp_path, (4,))


def test_predict_protocol_adds_trace_without_changing_actions(monkeypatch) -> None:
    server = object.__new__(inference_server.InferenceServer)
    action = np.arange(8, dtype=np.float32)
    evidence = {"schema_version": "aubo_act_server_trace_v1"}
    server.predict_with_trace = lambda _obs: (action, evidence)

    messages = iter([{"cmd": "predict", "obs": {}}, None])
    sent = []
    monkeypatch.setattr(inference_server, "recv_msg", lambda _conn: next(messages))
    monkeypatch.setattr(inference_server, "send_msg", lambda _conn, value: sent.append(value))

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    server.handle_client(Connection())

    np.testing.assert_array_equal(sent[0]["actions"], action)
    assert sent[0]["trace"] == evidence


def test_queue_selection_retains_chunk_only_when_model_runs() -> None:
    server = object.__new__(inference_server.InferenceServer)
    chunk = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)

    class Policy:
        config = type("Config", (), {"temporal_ensemble_coeff": None, "n_action_steps": 2})()
        _action_queue = deque([], maxlen=2)

        def eval(self):
            return self

        def predict_action_chunk(self, _batch):
            return chunk

    server.policy = Policy()
    server.queue_action_index = 0

    first, first_chunk, first_index = server._select_action_with_evidence({})
    second, second_chunk, second_index = server._select_action_with_evidence({})

    torch.testing.assert_close(first, chunk[:, 0])
    torch.testing.assert_close(second, chunk[:, 1])
    torch.testing.assert_close(first_chunk, chunk[:, :2])
    assert second_chunk is None
    assert (first_index, second_index) == (0, 1)


def test_ensemble_selection_retains_pre_ensemble_chunk() -> None:
    server = object.__new__(inference_server.InferenceServer)
    chunk = torch.arange(3 * 8, dtype=torch.float32).reshape(1, 3, 8)

    class Ensembler:
        def update(self, actions):
            return actions[:, 0] / 2

    class Policy:
        config = type("Config", (), {"temporal_ensemble_coeff": 0.01})()
        temporal_ensembler = Ensembler()

        def eval(self):
            return self

        def predict_action_chunk(self, _batch):
            return chunk

    server.policy = Policy()

    selected, predicted, selected_index = server._select_action_with_evidence({})

    torch.testing.assert_close(selected, chunk[:, 0] / 2)
    torch.testing.assert_close(predicted, chunk)
    assert selected_index == 0
