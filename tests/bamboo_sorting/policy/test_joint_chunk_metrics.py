import numpy as np
import pytest

from lerobot.bamboo_sorting.policy.joint_chunk_metrics import (
    EvaluationGeometry, action_chunk_metrics, evaluation_phase, grouped_chunk_metrics, overlapping_chunk_metrics,
)


def test_future_window_catches_error_hidden_by_first_row_and_excludes_padding():
    states = np.zeros((2,7))
    truth = np.zeros((2,50,7))
    prediction = truth.copy()
    prediction[:,8:20,:6] = 1
    padding = np.zeros((2,50),dtype=bool)
    padding[:,20:] = True
    prediction[:,20:] = 999999
    stats = {'action':{'q01':[0]*7,'q99':[100]*7}}
    metrics = action_chunk_metrics(states,truth,prediction,padding,stats)
    assert metrics['execution_window_joint_mae_deg'] == 1
    assert metrics['full_chunk_joint_mae_deg'] == pytest.approx(12/20)
    assert metrics['execution_window_valid_steps'] == 24
    assert not metrics['physical_success_verified'] and not metrics['target_obedience_verified']


def test_close_events_distinguish_timing_misses_and_spurious_closures():
    states = np.zeros((3,7)); truth = np.zeros((3,50,7)); prediction = truth.copy()
    truth[:2,12:,6] = 100
    prediction[0,15:,6] = 100
    prediction[2,10:,6] = 100
    metrics = action_chunk_metrics(states,truth,prediction,np.zeros((3,50),dtype=bool),
                                  {'action':{'q01':[0]*7,'q99':[100]*7}})
    assert metrics['close_events'] == dict(expected=2,missing=1,false_events=1,
        mean_absolute_timing_error_steps=3., timing_error_with_misses_steps=26.5,
        eligible_windows=3, failure_rate=pytest.approx(2/3), extra_switches_per_window=pytest.approx(1/3),
        event_errors_per_window=1., within_2_steps=0)


def test_nominal_geometry_and_onset_lift_have_correct_axes_and_units():
    joints = [dict(origin=np.eye(4).tolist(),axis=[0,1,0]) for _ in range(6)]
    geometry = EvaluationGeometry(dict(purpose='offline evaluation only',joints=joints,tool_offset_m=[1,0,0]))
    states = np.zeros((1,7)); truth = np.zeros((1,50,7)); pred = truth.copy(); pred[:,:,0] = 90
    metrics = action_chunk_metrics(states,truth,pred,np.zeros((1,50),dtype=bool),
        {'action':{'q01':[0]*7,'q99':[100]*7}},geometry=geometry,motion_onset=[True])
    assert metrics['execution_window_tcp_error_mm'] == pytest.approx(2**.5*1000)
    assert metrics['onset_predicted_lift_mm'] == pytest.approx(-1000)
    assert metrics['onset_expert_lift_mm'] == 0 and metrics['onset_observations'] == 1


def test_orientation_detects_wrong_pose_at_the_same_tcp_position():
    joints = [dict(origin=np.eye(4).tolist(), axis=[0, 0, 1]) for _ in range(6)]
    geometry = EvaluationGeometry(dict(purpose='offline evaluation only', joints=joints, tool_offset_m=[0, 0, 0]))
    truth = np.zeros((1, 50, 7)); pred = truth.copy(); pred[:, :, 5] = 90
    metrics = action_chunk_metrics(np.zeros((1, 7)), truth, pred, np.zeros((1, 50), dtype=bool),
        {'action': {'q01': [0]*7, 'q99': [100]*7}}, geometry=geometry)
    assert metrics['execution_window_tcp_error_mm'] == 0
    assert metrics['execution_window_orientation_error_deg'] == pytest.approx(90)
    assert metrics['execution_window_joint_mae_by_dimension_deg'] == [0, 0, 0, 0, 0, 90]


def test_missing_hard_events_cannot_artificially_improve_timing_score():
    states = np.zeros((2, 7)); truth = np.zeros((2, 50, 7)); truth[:, 10:, 6] = 100
    first = truth.copy(); first[0, :20, 6] = 0
    second = first.copy(); second[0, :, 6] = 0
    args = (np.zeros((2, 50), dtype=bool), {'action': {'q01': [0]*7, 'q99': [100]*7}})
    before = action_chunk_metrics(states, truth, first, *args)['close_events']
    after = action_chunk_metrics(states, truth, second, *args)['close_events']
    assert after['mean_absolute_timing_error_steps'] < before['mean_absolute_timing_error_steps']
    assert after['timing_error_with_misses_steps'] > before['timing_error_with_misses_steps']
    assert after['failure_rate'] > before['failure_rate']


def test_phase_and_episode_balancing_resists_long_easy_segments():
    joints = [dict(origin=np.eye(4).tolist(), axis=[0, 1, 0]) for _ in range(6)]
    geometry = EvaluationGeometry(dict(purpose='offline evaluation only', joints=joints, tool_offset_m=[1, 0, 0]))
    n = 111
    states = np.zeros((n, 7)); truth = np.zeros((n, 50, 7)); pred = truth.copy()
    pred[100:110, :, 0] = 1; pred[110, :, 0] = 3
    phases = ['transport']*100 + ['preclose']*11
    observations = dict(source_index=[0]*n, episode_index=[0]*110+[1],
                        frame_index=list(range(110))+[0], phase=phases)
    metrics = grouped_chunk_metrics(states, truth, pred, np.zeros((n, 50), dtype=bool),
        {'action': {'q01': [0]*7, 'q99': [100]*7}}, observations, geometry=geometry)
    # Episode 0 has ten preclose observations, episode 1 has one; equal episode weights.
    expected = (2*np.sin(np.deg2rad(.5)) + 2*np.sin(np.deg2rad(1.5)))/2*1000
    assert metrics['offline_selection']['primary'] == pytest.approx(expected)
    assert metrics['phase_metrics']['preclose']['episodes'] == 2
    assert metrics['phase_metrics']['transport']['tcp_error_mm'] == 0
    assert metrics['offline_selection']['guards']['precontact_joint_mae_deg'] == pytest.approx(2/6)


def test_overlap_aligns_timestamps_and_excludes_padding_and_other_episodes():
    pred = np.zeros((3, 50, 7)); pred[0, :, :6] = np.arange(50)[:, None]
    pred[1, :, :6] = (8+np.arange(50))[:, None]; pred[2] = 9999
    padding = np.zeros((3, 50), dtype=bool); padding[0, 40:] = True; padding[1, 32:] = True
    pred[1, 32:] = 9999
    observations = dict(source_index=[0, 0, 0], episode_index=[0, 0, 1],
                        frame_index=[0, 8, 0], phase=['approach']*3)
    result = overlapping_chunk_metrics(pred, padding, observations)
    assert result['pairs'] == 1 and result['joint_mae_deg'] == 0
    pred[1, :32, 5] += 6
    result = overlapping_chunk_metrics(pred, padding, observations)
    assert result['joint_mae_deg'] == 1


def test_phase_labels_use_expert_event_times():
    events = dict(onset=10, close=100, release=200)
    assert [evaluation_phase(f, events) for f in (0, 10, 35, 75, 100, 101, 175)] == [
        'initial_hold', 'onset', 'approach', 'preclose', 'preclose', 'transport', 'release']


def test_short_close_pulse_is_visible_even_with_the_correct_first_close_time():
    truth = np.zeros((1, 50, 7)); truth[:, 10:, 6] = 100
    pred = truth.copy(); pred[:, 11:, 6] = 0
    metrics = action_chunk_metrics(np.zeros((1, 7)), truth, pred, np.zeros((1, 50), dtype=bool),
        {'action': {'q01': [0]*7, 'q99': [100]*7}})
    assert metrics['close_events']['mean_absolute_timing_error_steps'] == 0
    assert metrics['close_events']['extra_switches_per_window'] == 1


def test_learning_events_is_not_vetoed_by_a_never_closing_zero_switch_baseline():
    from lerobot.bamboo_sorting.policy.pi05_joint_training import selection_comparison

    geometry = EvaluationGeometry(dict(purpose='offline evaluation only', tool_offset_m=[1, 0, 0],
        joints=[dict(origin=np.eye(4).tolist(), axis=[0, 0, 1]) for _ in range(6)]))
    states = np.zeros((4, 7)); truth = np.zeros((4, 50, 7)); truth[:, :, 0] = 10
    for i in range(4):
        truth[i, 10-i:, 6] = 100
    baseline = np.zeros_like(truth)
    learned = truth.copy(); learned[:, :, 0] = 9
    learned[0, 11:, 6] = 0  # One short pulse among four windows; first close is on time.
    observations = dict(source_index=[0]*4, episode_index=[0]*4, frame_index=list(range(4)), phase=['preclose']*4)
    stats = {'action': {'q01': [0]*7, 'q99': [100]*7}}
    def metrics(pred):
        return grouped_chunk_metrics(states, truth, pred, np.zeros((4, 50), dtype=bool), stats, observations, geometry=geometry)
    before, after = metrics(baseline), metrics(learned)
    assert before['phase_metrics']['preclose']['close_failure_rate'] == 1
    assert before['phase_metrics']['preclose']['close_extra_switches_per_window'] == 0
    assert before['offline_selection']['guards']['close_event_errors_per_window'] == 1
    assert after['offline_selection']['guards']['close_event_errors_per_window'] == .25
    assert selection_comparison(after, before)['improved']
    perfect = truth.copy(); perfect[:, :, 0] = 9.5
    # Once event behavior is correct, newly introduced pulses are still rejected.
    assert not selection_comparison(after, metrics(perfect))['improved']
