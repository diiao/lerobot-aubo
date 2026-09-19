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

EVALUATE_SCRIPT_PATH = SCRIPT_PATH.with_name("evaluate_split.py")
EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "evaluate_split_for_audit_consistency_test", EVALUATE_SCRIPT_PATH
)
assert EVALUATE_SPEC is not None and EVALUATE_SPEC.loader is not None
evaluate_split = importlib.util.module_from_spec(EVALUATE_SPEC)
EVALUATE_SPEC.loader.exec_module(evaluate_split)


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


def test_hybrid_keeps_ensemble_motion_and_selected_gripper() -> None:
    ensemble = np.arange(24, dtype=np.float32).reshape(3, 8)
    source = ensemble + 100.0

    hybrid = audit.deployed_hybrid_actions(ensemble, source, gripper_index=7)

    np.testing.assert_array_equal(hybrid[:, :7], ensemble[:, :7])
    np.testing.assert_array_equal(hybrid[:, 7], source[:, 7])
    np.testing.assert_array_equal(ensemble, np.arange(24, dtype=np.float32).reshape(3, 8))


def test_gripper_hysteresis_holds_middle_band() -> None:
    values = np.array([0.0, 61.0, 60.0, 35.0, 20.0, 19.0], dtype=np.float32)

    states = audit.apply_gripper_hysteresis(values)
    events = audit.find_threshold_transitions(values)

    np.testing.assert_array_equal(states, [False, True, True, True, True, False])
    assert [(event["kind"], event["frame"]) for event in events] == [
        ("on", 1),
        ("off", 5),
    ]


def test_action_continuity_uses_adjacent_commands() -> None:
    names = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.j6_target"]
    actions = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.003, 0.004, 0.0, 0.0, 0.0, np.pi / 2, 0.1],
            [0.003, 0.004, 0.012, 0.0, 0.0, np.pi, 0.3],
        ],
        dtype=np.float64,
    )

    values = audit.action_continuity_values(actions, names)

    np.testing.assert_allclose(values["xyz_step_l2_m"], [0.005, 0.012])
    np.testing.assert_allclose(values["j6_step_abs_rad"], [0.1, 0.2])
    np.testing.assert_allclose(values["rotation_step_geodesic_deg"], [90.0, 90.0])


def test_episode_fixed_latent_is_stable_across_batch_boundaries() -> None:
    cache: dict[int, torch.Tensor] = {}

    first = audit.episode_fixed_latent_batch(
        torch.tensor([69, 69, 74]),
        8,
        seed=42,
        device=torch.device("cpu"),
        cache=cache,
    )
    second = audit.episode_fixed_latent_batch(
        torch.tensor([74, 69]),
        8,
        seed=42,
        device=torch.device("cpu"),
        cache=cache,
    )

    torch.testing.assert_close(first[0], first[1])
    torch.testing.assert_close(first[0], second[1])
    torch.testing.assert_close(first[2], second[0])
    assert not torch.equal(first[0], first[2])


def test_episode_fixed_latent_changes_with_seed() -> None:
    episode_indices = torch.tensor([69])

    seed_1 = audit.episode_fixed_latent_batch(
        episode_indices,
        8,
        seed=1,
        device=torch.device("cpu"),
        cache={},
    )
    seed_2 = audit.episode_fixed_latent_batch(
        episode_indices,
        8,
        seed=2,
        device=torch.device("cpu"),
        cache={},
    )

    assert not torch.equal(seed_1, seed_2)


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


def test_transition_matching_ignores_early_spurious_pulse() -> None:
    targets = [
        {"kind": "on", "frame": 100, "ordinal": 0},
        {"kind": "off", "frame": 200, "ordinal": 0},
    ]
    predictions = [
        {"kind": "on", "frame": 78, "ordinal": 0},
        {"kind": "off", "frame": 80, "ordinal": 0},
        {"kind": "on", "frame": 105, "ordinal": 1},
        {"kind": "off", "frame": 206, "ordinal": 1},
    ]

    matched, unmatched = audit.match_transition_events(targets, predictions)

    assert matched == {("on", 0): 105, ("off", 0): 206}
    assert unmatched == {"on": 1, "off": 1}


def test_target_event_summary_uses_selected_episode_labels_not_fixed_count() -> None:
    events = [
        {"episode_index": 4, "target_frame": 10, "frame": 10, "kind": "on", "ordinal": 0},
        {"episode_index": 4, "target_frame": 30, "frame": 30, "kind": "off", "ordinal": 0},
        {"episode_index": 9, "target_frame": 12, "frame": 12, "kind": "on", "ordinal": 0},
        {"episode_index": 9, "target_frame": 32, "frame": 32, "kind": "off", "ordinal": 0},
    ]

    summary = audit.summarize_target_suction_events(events)

    assert summary["target_on_count"] == 2
    assert summary["target_off_count"] == 2
    assert "found_on" not in summary
    assert "all_on_found" not in summary
    assert [event["episode_index"] for event in summary["on_events"]] == [4, 9]


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


def test_local_dataset_loaders_pass_repo_id_and_root_explicitly(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "local_dataset"
    (dataset_path / "meta").mkdir(parents=True)
    (dataset_path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        audit,
        "LeRobotDatasetMetadata",
        lambda **kwargs: calls.append(("metadata", kwargs)) or "metadata",
    )
    monkeypatch.setattr(
        audit,
        "LeRobotDataset",
        lambda **kwargs: calls.append(("dataset", kwargs)) or "dataset",
    )

    assert audit.load_local_dataset_metadata(dataset_path) == "metadata"
    assert (
        audit.load_local_dataset(
            dataset_path,
            episodes=[1, 2],
            delta_timestamps={"action": [0.0]},
        )
        == "dataset"
    )
    assert calls[0] == (
        "metadata",
        {"repo_id": "local_dataset", "root": dataset_path.resolve()},
    )
    assert calls[1] == (
        "dataset",
        {
            "repo_id": "local_dataset",
            "root": dataset_path.resolve(),
            "episodes": [1, 2],
            "delta_timestamps": {"action": [0.0]},
        },
    )


def test_current_client_guard_replay_separates_raw_reject_from_post_clip_pass() -> None:
    action_names = list(audit.ACTION_FIELD_NAMES)
    state_names = [
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
        "J6",
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "gripper_pos",
    ]
    selected = np.asarray(
        [[-2.8, 0.20, -0.70, 0.15, 0.1, 0.2, 0.3, 0.0]], dtype=np.float32
    )
    chunks = np.repeat(selected[:, None, :], 25, axis=1)
    state = np.asarray(
        [[0.0, 0.0, 0.0, 0.0, 0.0, -185.0, 0.10, -0.70, 0.15, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    rows = audit.replay_current_client_guard(
        selected,
        chunks,
        state,
        action_names=action_names,
        state_names=state_names,
        fps=25,
    )
    summary = audit.summarize_client_guard_replay(rows)

    assert rows[0]["raw_gate_reasons"] == ["step:0", "speed:0", "j6_step:0"]
    assert rows[0]["final_gate_reasons"] == []
    assert summary["raw_gate_reject_count"] == 1
    assert summary["post_transform_gate_pass_count"] == 1
    assert summary["transform_counts"]["measured_xyz_step_clip"] == 1
    assert summary["transform_counts"]["measured_j6_step_clip"] == 1
    assert summary["limitations"]["ik_replayed"] is False


def test_client_guard_replay_defaults_match_evaluate_split() -> None:
    assert audit.CLIENT_USE_GRIPPER_CHUNK_LOOKAHEAD is evaluate_split.USE_GRIPPER_CHUNK_LOOKAHEAD
    assert audit.CLIENT_USE_MOTION_CHUNK_LOOKAHEAD is evaluate_split.USE_MOTION_CHUNK_LOOKAHEAD
    assert audit.CLIENT_WORKSPACE_MIN_M == evaluate_split.WORKSPACE_MIN_M
    assert audit.CLIENT_WORKSPACE_MAX_M == evaluate_split.WORKSPACE_MAX_M
    assert audit.CLIENT_MAX_EE_STEP_M == evaluate_split.MAX_EE_STEP_M
    assert audit.CLIENT_MAX_J6_STEP_RAD == evaluate_split.MAX_J6_STEP_RAD
    assert audit.CLIENT_J6_TARGET_MIN_RAD == evaluate_split.J6_TARGET_MIN_RAD
    assert audit.CLIENT_J6_TARGET_MAX_RAD == evaluate_split.J6_TARGET_MAX_RAD


def test_current_client_guard_replay_records_chunk_lookahead_overrides() -> None:
    action_names = list(audit.ACTION_FIELD_NAMES)
    state_names = [
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
        "J6",
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "gripper_pos",
    ]
    selected = np.asarray(
        [[-3.2, 0.10, -0.70, 0.18, 0.0, 0.0, 0.0, -3.0]], dtype=np.float32
    )
    chunks = np.repeat(selected[:, None, :], 25, axis=1)
    chunks[0, -1, 0:4] = [-3.2, 0.11, -0.69, 0.09]
    chunks[0, -1, 7] = 25.0
    state = np.asarray(
        [[0.0, 0.0, 0.0, 0.0, 0.0, -185.0, 0.10, -0.70, 0.18, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    rows = audit.replay_current_client_guard(
        selected,
        chunks,
        state,
        action_names=action_names,
        state_names=state_names,
        fps=25,
        use_gripper_chunk_lookahead=True,
        use_motion_chunk_lookahead=True,
    )

    assert rows[0]["motion_chunk_lookahead_overrode"] is True
    assert rows[0]["gripper_chunk_lookahead_mode"] == "close"
    assert rows[0]["suction_on"] is True
