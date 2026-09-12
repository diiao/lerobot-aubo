from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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
