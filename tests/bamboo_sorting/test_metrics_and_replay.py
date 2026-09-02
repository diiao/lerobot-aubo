from __future__ import annotations

import numpy as np
import pytest

from lerobot.bamboo_sorting import ExperimentAllocation, NpzSceneSensor, wilson_interval


def test_experiment_budget_is_150():
    assert ExperimentAllocation().total == 150


def test_wilson_interval_known_example():
    result = wilson_interval(15, 20)
    assert result.proportion == 0.75
    assert result.lower_95 == pytest.approx(0.5313, abs=1e-4)
    assert result.upper_95 == pytest.approx(0.8881, abs=1e-4)


def test_npz_replay_reads_scene_once(tmp_path):
    path = tmp_path / "frame.npz"
    np.savez(
        path,
        points_xyz=np.array([[0.1, 0.2, 0.3]]),
        timestamp_s=np.array(12.5),
        frame_id=np.array("base"),
    )
    sensor = NpzSceneSensor([path])
    frame = sensor.capture()
    assert frame.timestamp_s == 12.5
    assert frame.metadata["replay_path"] == str(path)
    with pytest.raises(StopIteration):
        sensor.capture()
