from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.bamboo_sorting.act_input_contract import (
    DROP_GRIPPER_STATE_VARIANT,
    build_state_input_contract,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "phone_to_auboi10"
    / "audit_act_offline.py"
)
SPEC = importlib.util.spec_from_file_location("audit_act_offline_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_temporal_ensemble_matches_act_ensembler() -> None:
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    rng = np.random.default_rng(0)
    chunks = rng.normal(size=(17, 25, 8)).astype(np.float32)
    got = audit.deployed_temporal_ensemble_actions(chunks, 0.01)
    ensembler = ACTTemporalEnsembler(0.01, 25)
    reference = []
    for step in range(len(chunks)):
        action = ensembler.update(torch.from_numpy(chunks[step : step + 1]))
        reference.append(action.squeeze(0).numpy())
    np.testing.assert_allclose(got, np.stack(reference), atol=1e-6)
    queue = audit.deployed_queue_actions(chunks, 4)
    assert not np.allclose(got, queue, atol=1e-3)


def test_finds_all_on_and_off_transitions() -> None:
    episodes = []
    for _ in range(13):
        values = np.zeros(80, dtype=np.float32)
        values[10:25] = 100.0
        values[40:55] = 100.0
        episodes.append(values)
    events = [event for series in episodes for event in audit.find_threshold_transitions(series)]
    ons = [event for event in events if event["kind"] == "on"]
    offs = [event for event in events if event["kind"] == "off"]
    assert len(ons) == 26
    assert len(offs) == 26
    assert {event["ordinal"] for event in ons} == {0, 1}
    assert {event["ordinal"] for event in offs} == {0, 1}


def test_geodesic_treats_plus_minus_pi_as_equivalent() -> None:
    plus = np.array([[0.0, 0.0, np.pi]], dtype=np.float64)
    minus = np.array([[0.0, 0.0, -np.pi]], dtype=np.float64)
    angle = audit.geodesic_angle_deg(plus, minus)
    assert angle.shape == (1,)
    assert float(angle[0]) < 1e-5
    identity = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    two_pi = np.array([[0.0, 0.0, 2.0 * np.pi]], dtype=np.float64)
    assert float(audit.geodesic_angle_deg(identity, two_pi)[0]) < 1e-5


def test_forced_zero_does_not_mutate_original_batch() -> None:
    state = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 100.0]])
    batch = {
        "observation.state": state,
        "observation.images.handeye": torch.zeros(1, 3, 4, 4),
        "observation.images.fixed": torch.zeros(1, 3, 4, 4),
        "task": ["grab"],
    }
    original = state.clone()
    features = {
        "observation.state": None,
        "observation.images.handeye": None,
        "observation.images.fixed": None,
    }
    masked = audit.make_inference_observation(batch, features, state_gripper_index=12)
    assert torch.equal(batch["observation.state"], original)
    assert float(original[0, 12]) == 100.0
    assert float(masked["observation.state"][0, 12]) == 0.0
    assert float(batch["observation.state"][0, 12]) == 100.0


def test_drop_gripper_contract_builds_12d_audit_observation() -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [13],
            "names": [f"state_{index}" for index in range(12)] + ["gripper_pos"],
        }
    }
    contract = build_state_input_contract(features, DROP_GRIPPER_STATE_VARIANT)
    original = {
        "observation.state": torch.arange(26, dtype=torch.float32).reshape(2, 13),
    }
    result = audit.make_inference_observation(
        original,
        {"observation.state": object()},
        state_contract=contract,
    )
    assert result["observation.state"].shape == (2, 12)
    assert original["observation.state"].shape == (2, 13)


def test_make_report_dir_refuses_to_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "audit_run02_best_val"
    existing.mkdir()
    (existing / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        audit.make_report_dir(str(existing))
