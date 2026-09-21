import numpy as np
import pytest

from lerobot.bamboo_sorting.smolvla_action_decoder import decode_smolvla_gripper
from lerobot.bamboo_sorting.thin_safety_gate import ThinSafetyGate


def test_decoding_retains_raw_scores_and_all_seven_motion_coordinates():
    raw = [[0., .1, -.5, .2, .01, .02, .03, x] for x in [-25., 49.9, 50., 133.]]
    result = decode_smolvla_gripper(raw)
    assert result.raw_actions == tuple(tuple(r) for r in raw)
    assert [r[-1] for r in result.actions] == [0., 0., 100., 100.]
    assert all(r[:7] == a[:7] for r, a in zip(result.raw_actions, result.actions))
    assert raw[0][-1] == -25.


def test_decoding_cannot_hide_a_position_jump_from_the_gate():
    result = decode_smolvla_gripper([[0., .2, -.5, .2, 0., 0., 0., 92.]])
    verdict = ThinSafetyGate().evaluate(result.actions, now_monotonic_s=0,
        observation_sync_timestamp_s=0, chunk_created_monotonic_s=0,
        previous_tcp_m=(0., -.5, .2), previous_j6_rad=0)
    assert not verdict.passed
    assert 'step:0' in verdict.reasons
    assert verdict.actions == result.actions


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, np.bool_(False), '50', None])
def test_invalid_raw_scores_are_rejected(bad):
    with pytest.raises(ValueError):
        decode_smolvla_gripper([[0., .1, -.5, .2, 0., 0., 0., bad]])


def test_empty_or_wrong_width_chunk_is_rejected():
    for raw in [[], [[0.] * 7]]:
        with pytest.raises(ValueError):
            decode_smolvla_gripper(raw)
