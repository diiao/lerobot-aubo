"""Pixel-provenance and identity-reference checks for the offline prototype."""
import numpy as np
import pytest

from lerobot.bamboo_sorting.perception.joint_target_separation import PairedStripSeparation
from lerobot.bamboo_sorting.perception.joint_target_separation import MultiStripSeparation, visible_seed_masks
from tests.bamboo_sorting.perception_fakes import crossing_seeds, instance_map


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


def test_multi_crossing_preserves_current_instances_and_occlusion():
    original = crossing_seeds()
    tracker = MultiStripSeparation(original)
    result, evidence = tracker.step(0, *instance_map(original))
    for name in original:
        np.testing.assert_array_equal(result[name], original[name])
    assert not evidence['merged_instances']
    current = {t: np.roll(m, 2, axis=1) for t, m in original.items()}
    # Current instance IDs and ordering must not define persistent identity.
    result, _ = tracker.step(1, *instance_map(dict(reversed(list(current.items())))))
    for name in current:
        np.testing.assert_array_equal(result[name], current[name])


def test_multi_merged_crossing_uses_visible_seed_ownership_and_flags_review():
    original = crossing_seeds()
    result, evidence = MultiStripSeparation(original).step(0, *merged(original))
    for name in original:
        np.testing.assert_array_equal(result[name], original[name])
    assert evidence['merged_instances'][0]['requires_crossing_review']
    assert not evidence['human_reviewed']


def test_multi_disappearance_and_unknown_foreground_do_not_invent_identity():
    original = crossing_seeds()
    current = {t: m for t, m in original.items() if t != 's-001'}
    unknown = np.zeros((480, 640), bool)
    unknown[20:80, 20:40] = True
    current['unknown'] = unknown
    result, evidence = MultiStripSeparation(original).step(0, *instance_map(current))
    assert not result['s-001'].any()
    assert evidence['unassigned_foreground_pixels'] == int(unknown.sum())
    assert not np.any(np.logical_or.reduce(list(result.values())) & unknown)


def test_multi_occluded_drafts_do_not_shrink_identity_references():
    original = crossing_seeds()
    tracker = MultiStripSeparation(original)
    partial = {t: m.copy() for t, m in original.items()}
    for mask in partial.values():
        mask[200:] = False
    tracker.step(0, *instance_map(partial))
    for name in original:
        np.testing.assert_array_equal(tracker.references[name], original[name])
    # Reappearance must recover using current pixels, without a learned tip.
    result, _ = tracker.step(1, *merged(original))
    for name in original:
        np.testing.assert_array_equal(result[name], original[name])


def test_multi_single_supported_instance_cannot_absorb_distant_foreground():
    original = crossing_seeds()
    unknown = np.zeros((480, 640), bool)
    unknown[20:80, 20:40] = True
    current = {'merged': original['s-003'] | unknown}
    tracker = MultiStripSeparation(original)
    result, evidence = tracker.step(0, *instance_map(current))
    np.testing.assert_array_equal(result['s-003'], original['s-003'])
    assert evidence['unassigned_foreground_pixels'] == int(unknown.sum())
    # No current detections means no output, despite the retained anchors.
    result, _ = tracker.step(1, *instance_map({}))
    assert not any(m.any() for m in result.values())


def test_multi_dominant_strip_does_not_erase_distant_visible_fragment():
    original = crossing_seeds()
    fragment = original['s-003'].copy()
    fragment[115:] = False
    result, evidence = MultiStripSeparation(original).step(
        0, *instance_map({'merged': original['s-001'] | fragment}))
    np.testing.assert_array_equal(result['s-003'], fragment)
    np.testing.assert_array_equal(result['s-001'], original['s-001'])
    assert evidence['merged_instances']


@pytest.mark.parametrize('radius', [8, 16, 24])
@pytest.mark.parametrize('minor_start', [315, 320])
def test_multi_wider_search_cannot_give_a_nearby_minor_identity_to_the_majority(radius, minor_start):
    a, b = (np.zeros((480, 640), bool) for _ in range(2))
    a[200:400, 300:310] = True
    b[200:400, minor_start:minor_start + 10] = True
    fragment = b.copy()
    fragment[210:] = False
    original = {'A': a, 'B': b}
    # A dominates the merged instance, but the visible B tip is still B.
    # A wider search radius must not turn split ownership into wholesale A.
    result, evidence = MultiStripSeparation(original, motion_radius_px=radius).step(
        0, *merged({'A': a, 'B': fragment}))
    np.testing.assert_array_equal(result['A'], a)
    np.testing.assert_array_equal(result['B'], fragment)
    assert evidence['merged_instances']


def test_multi_wider_search_can_recover_unassigned_foreground_without_erasing_a_seed():
    a, b = (np.zeros((480, 640), bool) for _ in range(2))
    a[200:400, 300:310] = True
    b[200:400, 400:410] = True
    current = np.roll(a, -12, axis=1)
    instance, segments = instance_map({'A': current, 'B': b})
    narrow, _ = MultiStripSeparation({'A': a, 'B': b}).step(0, instance, segments)
    wide, _ = MultiStripSeparation({'A': a, 'B': b}, motion_radius_px=16).step(0, instance, segments)
    assert narrow['A'].sum() < current.sum()
    np.testing.assert_array_equal(wide['A'], current)
    np.testing.assert_array_equal(wide['B'], b)


def test_visible_polygon_and_candidate_seeds_have_same_pixel_contract():
    masks = crossing_seeds()
    instances, _ = instance_map(masks)
    selected = visible_seed_masks([{'target_id': 'a', 'instance_ids': [7]},
                                  {'target_id': 'b', 'instance_ids': [14]}], instances)
    np.testing.assert_array_equal(selected['a'], masks['s-001'])
    with pytest.raises(ValueError, match='overlap'):
        visible_seed_masks([{'target_id': 'a', 'instance_ids': [7]},
                            {'target_id': 'b', 'instance_ids': [7]}], instances)
