"""Pixel-provenance and identity-reference checks for the offline prototype."""
import numpy as np
import pytest

from lerobot.bamboo_sorting.joint_target_separation import PairedStripSeparation


def seeds():
    a, b = (np.zeros((480, 640), bool) for _ in range(2))
    a[200:400, 490:502] = True
    b[200:400, 400:412] = True
    return {'A': a, 'B': b}


def merged(masks):
    return np.logical_or.reduce(list(masks.values())).astype(np.uint16), [{'mask_id': 1, 'class_id': 0}]


def test_merged_instance_separates_without_added_or_shared_pixels():
    original = seeds()
    result, evidence = PairedStripSeparation(original).step(0, *merged(original))
    for target in original:
        np.testing.assert_array_equal(result[target], original[target])
    assert not np.any(result['A'] & result['B'])
    assert evidence['execution_eligible'] is False


def test_missing_target_does_not_rename_competitor_or_propagate_seed():
    original = seeds()
    tracker = PairedStripSeparation(original)
    only_a = {'A': original['A']}
    for frame in range(3):
        result, evidence = tracker.step(frame, *merged(only_a))
        assert not result['B'].any()
        np.testing.assert_array_equal(result['A'], original['A'])
        assert evidence['targets']['B']['status'] == 'no_assigned_foreground'
    result, _ = tracker.step(3, *merged(original))
    np.testing.assert_array_equal(result['B'], original['B'])


def test_occlusion_does_not_move_reference_to_visible_endpoint():
    original = seeds()
    tracker = PairedStripSeparation(original)
    before = tracker.references['B']['center'].copy()
    partial = {key: value.copy() for key, value in original.items()}
    partial['B'][:340] = False
    result, _ = tracker.step(0, *merged(partial))
    np.testing.assert_array_equal(result['B'], partial['B'])
    # Occlusion moves the visible centroid 70 px along the strip. A robust
    # raster line fit need not reproduce the seed center to machine precision,
    # but must not follow that endpoint displacement.
    shift = tracker.references['B']['center'] - before
    assert abs(float(shift @ np.array([0., 1.]))) < 1.
    assert abs(shift[0]) < tracker.references['B']['half_width']


def test_independent_small_motion_and_instance_id_changes():
    original = seeds()
    tracker = PairedStripSeparation(original)
    for frame in range(8):
        current = {'A': original['A'], 'B': np.roll(original['B'], frame*3, axis=1)}
        ids = (7, 2) if frame % 2 else (2, 7)
        instances = np.zeros((480, 640), np.uint16)
        for target, mask_id in zip(current, ids):
            instances[current[target]] = mask_id
        result, _ = tracker.step(frame, instances, [{'mask_id': i, 'class_id': 0} for i in reversed(ids)])
        for target in current:
            np.testing.assert_array_equal(result[target], current[target])


def test_empty_frame_returns_empty_masks_and_rejects_discontinuous_index():
    tracker = PairedStripSeparation(seeds())
    result, _ = tracker.step(0, np.zeros((480, 640), np.uint16), [])
    assert all(not mask.any() for mask in result.values())
    with pytest.raises(ValueError, match='consecutive'):
        tracker.step(2, *merged(seeds()))


def test_invalid_seed_and_candidate_map_rejected():
    original = seeds()
    with pytest.raises(ValueError, match='overlap'):
        PairedStripSeparation({'A': original['A'], 'B': original['A']})
    tracker = PairedStripSeparation(original)
    with pytest.raises(ValueError, match='absent'):
        tracker.step(0, original['A'].astype(np.uint16), [])
