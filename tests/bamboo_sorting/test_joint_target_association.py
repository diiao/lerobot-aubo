import numpy as np
import pytest

from lerobot.bamboo_sorting.joint_target_association import TargetAssociation, TargetReconfirmation


def frame(*rectangles):
    labels = np.zeros((480, 640), np.uint16)
    segments = []
    for mask_id, x, y, width, height in rectangles:
        labels[y:y + height, x:x + width] = mask_id
        segments.append({"mask_id": mask_id, "class_id": 0, "score": 0.9})
    return labels, segments


def test_association_follows_geometry_when_frame_local_ids_swap():
    tracker = TargetAssociation("strip_001")
    tracker.select(600, *frame((1, 100, 200, 100, 20)), instance_id=1)
    current, segments = frame((9, 103, 200, 100, 20), (1, 400, 200, 100, 20))
    segments[0]["class_id"] = 1  # A classifier change does not change object identity.
    mask, result = tracker.step(601, current, segments)
    assert result["status"] == "matched" and result["target_id"] == "strip_001"
    assert result["mask_id"] == 9 and result["class_id"] == 1
    assert np.array_equal(mask, current == 9)
    assert not mask[200, 100]  # Never render the stale mask.


def test_partial_visible_mask_is_not_completed_from_old_shape():
    tracker = TargetAssociation("strip_001")
    tracker.select(0, *frame((1, 100, 200, 100, 20)), instance_id=1)
    current, segments = frame((7, 100, 200, 50, 20))
    mask, result = tracker.step(1, current, segments)
    assert result["status"] == "matched"
    assert np.array_equal(mask, current == 7) and not mask[:, 150:].any()


def test_near_equal_candidates_are_ambiguous_instead_of_silently_switching():
    tracker = TargetAssociation("strip_001", alignment="rigid")
    tracker.select(0, *frame((1, 100, 200, 100, 24)), instance_id=1)
    mask, result = tracker.step(1, *frame((8, 100, 200, 100, 11), (9, 100, 213, 100, 11)))
    assert mask is None and result["reason"] == "ambiguous_candidates"
    mask, result = tracker.step(2, *frame((1, 100, 200, 100, 24)))
    assert mask is None and result["reason"] == "explicit_selection_required"


@pytest.mark.parametrize("next_index,rectangles,reason", [
    (1, (), "no_compatible_candidate"),
    (1, ((1, 400, 200, 100, 20),), "no_compatible_candidate"),
    (2, ((1, 100, 200, 100, 20),), "nonconsecutive_frame"),
])
def test_loss_never_falls_back_to_same_pixel_id_or_last_mask(next_index, rectangles, reason):
    tracker = TargetAssociation("strip_001")
    tracker.select(0, *frame((1, 100, 200, 100, 20)), instance_id=1)
    mask, result = tracker.step(next_index, *frame(*rectangles))
    assert mask is None and result["reason"] == reason and result["mask_id"] is None


def test_rigid_alignment_handles_turn_but_returns_only_new_prediction():
    import cv2

    tracker = TargetAssociation("strip_001", alignment="rigid")
    initial, segments = frame((1, 180, 220, 180, 12))
    tracker.select(0, initial, segments, instance_id=1)
    transform = cv2.getRotationMatrix2D((270, 226), 22, 1.0)
    transform[:, 2] += [0, 26]
    current = cv2.warpAffine(initial, transform, (640, 480), flags=cv2.INTER_NEAREST)
    current[current > 0] = 9
    selected, result = tracker.step(1, current, [{"mask_id": 9, "class_id": 0}])
    assert result["status"] == "matched"
    assert np.array_equal(selected, current == 9)


def reconfirmation_seed():
    tracker = TargetReconfirmation("strip_001")
    tracker.select(0, *frame((1, 100, 200, 100, 24)), instance_id=1, source_timestamp=1.0)
    return tracker


def test_reconfirmation_requires_two_source_images_and_returns_only_current_mask():
    tracker = reconfirmation_seed()
    selected, result = tracker.step(1, *frame(), source_timestamp=1.04)
    assert selected is None and result["status"] == "unobserved"
    selected, result = tracker.step(2, *frame((7, 101, 200, 100, 24)), source_timestamp=1.08)
    assert selected is None and result["status"] == "confirming" and result["confirmations"] == 1
    current, segments = frame((9, 102, 200, 100, 24))
    selected, result = tracker.step(3, current, segments, source_timestamp=1.12)
    assert result["status"] == "reconfirmed" and result["target_id"] == "strip_001"
    assert result["confirmations"] == 2 and result["gap_frames"] == 3
    assert np.array_equal(selected, current == 9)


def test_repeated_source_timestamp_does_not_confirm_identity():
    tracker = reconfirmation_seed()
    tracker.step(1, *frame(), source_timestamp=1.04)
    tracker.step(2, *frame((7, 100, 200, 100, 24)), source_timestamp=1.08)
    selected, result = tracker.step(3, *frame((7, 100, 200, 100, 24)), source_timestamp=1.08)
    assert selected is None and result["confirmations"] == 1
    selected, result = tracker.step(4, *frame((8, 101, 200, 100, 24)), source_timestamp=1.12)
    assert result["status"] == "reconfirmed" and selected is not None


def test_reconfirmation_window_expires_and_requires_selection():
    tracker = reconfirmation_seed()
    for i in range(1, 6):
        selected, result = tracker.step(i, *frame(), source_timestamp=1 + i * 0.04)
        assert selected is None and result["status"] == "unobserved"
    selected, result = tracker.step(6, *frame((1, 100, 200, 100, 24)), source_timestamp=1.24)
    assert selected is None and result["reason"] == "reconfirmation_timeout"
    selected, result = tracker.step(7, *frame((1, 100, 200, 100, 24)), source_timestamp=1.28)
    assert selected is None and result["reason"] == "explicit_selection_required"


@pytest.mark.parametrize("rectangles", [
    ((1, 140, 200, 100, 24),),  # Outside the recovery anchor gate despite same ID.
    ((8, 100, 200, 100, 11), (9, 100, 213, 100, 11)),  # Ambiguous candidates.
])
def test_reconfirmation_rejects_distractor_or_ambiguity(rectangles):
    tracker = reconfirmation_seed()
    tracker.step(1, *frame(), source_timestamp=1.04)
    selected, result = tracker.step(2, *frame(*rectangles), source_timestamp=1.08)
    assert selected is None and result["status"] == "lost"
    selected, result = tracker.step(3, *frame((1, 100, 200, 100, 24)), source_timestamp=1.12)
    assert selected is None and result["reason"] == "explicit_selection_required"


def test_another_empty_detection_resets_pending_confirmation():
    tracker = reconfirmation_seed()
    tracker.step(1, *frame(), source_timestamp=1.04)
    tracker.step(2, *frame((1, 100, 200, 100, 24)), source_timestamp=1.08)
    selected, result = tracker.step(3, *frame(), source_timestamp=1.12)
    assert selected is None and result["confirmations"] == 0
    selected, result = tracker.step(4, *frame((1, 100, 200, 100, 24)), source_timestamp=1.16)
    assert selected is None and result["confirmations"] == 1


@pytest.mark.parametrize("next_index,timestamp", [(2, 1.08), (1, 0.99)])
def test_reconfirmation_cannot_bridge_missing_frames_or_reversed_camera_time(next_index, timestamp):
    tracker = reconfirmation_seed()
    selected, result = tracker.step(next_index, *frame((1, 100, 200, 100, 24)), source_timestamp=timestamp)
    assert selected is None and result["reason"] == "frame_or_source_time_discontinuity"
