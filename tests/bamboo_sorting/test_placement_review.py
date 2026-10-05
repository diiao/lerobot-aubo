import copy
import json

import cv2
import numpy as np
import pytest

from lerobot.bamboo_sorting.placement_review import build_review, polygon_mask, save_review
from lerobot.bamboo_sorting.placement_sequence import PlacementSequence


@pytest.fixture
def example(tmp_path):
    source = tmp_path / "source"
    sequence = PlacementSequence(source, placement_id="pile", split="validation", configuration={})
    frame = np.zeros((80, 120, 3), np.uint8)
    sequence.append(frame)
    points = [
        [[10, 30], [109, 30], [109, 39], [10, 39]],
        [[50, 10], [59, 10], [59, 69], [50, 69]],
        [[30, 45], [119, 45], [119, 54], [30, 54]],
    ]
    for polygon in points:
        frame[polygon_mask(polygon, frame.shape)] = [210, 170, 105]
        sequence.append(frame)
    annotations = {"placement_id": "pile", "image_size": [120, 80],
                   "annotation_source": "test_draft", "reviewed_by": "",
                   "placement_from_above_confirmed": True, "older_strips_stationary_confirmed": True,
                   "placement_confirmation_source": "synthetic fixture",
                   "objects": [{"id": f"strip_{i:03d}", "frame_index": i, "polygon": p,
                                "reviewed": False, "truncated": False} for i, p in enumerate(points, 1)]}
    return source, annotations


def test_propagation_fills_same_color_gap_and_preserves_source(example, tmp_path):
    source, annotations = example
    original = (source / "sequence.json").read_bytes()
    output = tmp_path / "review"
    result = build_review(source, annotations, output)
    assert result["placement_id"] == "pile" and result["split"] == "validation"
    labels = json.loads((output / "frame_002_labels.json").read_text())
    assert labels["top_ids_candidate"] == ["strip_002"]
    assert labels["annotation_status"] == "polygon_draft" and not labels["training_ready"]
    assert not labels["mask_review_complete"] and not labels["full_frame_mask_eligible"]
    mask1 = cv2.imread(str(output / "frame_002_strip_001_visible.png"), 0)
    mask2 = cv2.imread(str(output / "frame_002_strip_002_visible.png"), 0)
    assert np.count_nonzero(mask1) == 900 and np.count_nonzero(mask2) == 600
    assert mask2[30:40, 50:60].all() and not mask1[30:40, 50:60].any()
    assert (source / "sequence.json").read_bytes() == original
    assert (source / "frame_002.png").read_bytes() == (output / "frame_002.png").read_bytes()
    assert '__REVIEW_DATA__' not in (output / "review.html").read_text()
    with pytest.raises(FileExistsError):
        build_review(source, annotations, output)


@pytest.mark.parametrize("reviewer", [None, "", "fixture-reviewer"])
def test_reviewed_truncated_frame_is_not_complete_training_label(example, tmp_path, reviewer):
    source, annotations = example
    if reviewer is None:
        annotations.pop("reviewed_by")
    else:
        annotations["reviewed_by"] = reviewer
    for item in annotations["objects"]:
        item["reviewed"] = True
    build_review(source, annotations, tmp_path / "review")
    normal = json.loads((tmp_path / "review/frame_002_labels.json").read_text())
    clipped = json.loads((tmp_path / "review/frame_003_labels.json").read_text())
    assert normal["full_frame_mask_eligible"] and not normal["training_ready"]
    assert clipped["mask_review_complete"] and not clipped["full_frame_mask_eligible"]
    assert clipped["instances"][-1]["truncated"]  # Detected even if checkbox says false.
    assert json.loads((tmp_path / "review/polygons.json").read_text()) == annotations


def test_unconfirmed_placement_does_not_propagate_old_masks(example, tmp_path):
    source, annotations = example
    annotations["older_strips_stationary_confirmed"] = False
    build_review(source, annotations, tmp_path / "review")
    labels = json.loads((tmp_path / "review/frame_002_labels.json").read_text())
    assert labels["top_ids_candidate"] is None
    assert labels["relations_from_order_and_overlap"] is None
    assert labels["instances"][0]["visible_mask"] is None
    assert labels["instances"][1]["visible_mask"]


@pytest.mark.parametrize("change", ["group", "size", "order", "review_flag", "outside", "crossing"])
def test_invalid_annotation_does_not_create_output(example, tmp_path, change):
    source, draft = example
    annotations = copy.deepcopy(draft)
    if change == "group":
        annotations["placement_id"] = "another-pile"
    elif change == "size":
        annotations["image_size"] = [1920, 1080]
    elif change == "order":
        annotations["objects"].reverse()
    elif change == "review_flag":
        annotations["objects"][0]["reviewed"] = "true"
    elif change == "outside":
        annotations["objects"][0]["polygon"][0] = [-1, 30]
    else:
        annotations["objects"][0]["polygon"] = [[10, 10], [80, 50], [10, 40], [50, 10]]
    with pytest.raises(ValueError):
        build_review(source, annotations, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_save_review_updates_one_directory_and_index(example):
    source, annotations = example
    original = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    save_review(source, annotations)
    root = source.parent / "review"
    output = root / source.name
    assert "待复核 0/3" in (root / "index.html").read_text()
    for item in annotations["objects"]:
        item["reviewed"] = True
    annotations["objects"][0]["polygon"][0][0] = 12
    save_review(source, annotations)
    assert json.loads((output / "polygons.json").read_text()) == annotations
    assert json.loads((output / "frame_002_labels.json").read_text())["full_frame_mask_eligible"]
    assert "轮廓已确认 3/3" in (root / "index.html").read_text()
    assert sorted(p.name for p in root.iterdir()) == ["index.html", source.name]
    assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == original


def test_failed_review_update_preserves_previous_result(example, monkeypatch):
    from lerobot.bamboo_sorting import placement_review

    source, annotations = example
    save_review(source, annotations)
    output = source.parent / "review" / source.name
    previous = {p.name: p.read_bytes() for p in output.iterdir()}
    def fail_write(*args, **kwargs):
        raise OSError("simulated derived-image write failure")
    monkeypatch.setattr(placement_review, "write_image", fail_write)
    with pytest.raises(OSError, match="simulated"):
        save_review(source, annotations)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == previous
    assert not list(output.parent.glob(".review-update-*"))


def test_save_review_rejects_unrelated_or_source_directory(example, tmp_path):
    source, annotations = example
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="not a review"):
        save_review(source, annotations, unrelated)
    with pytest.raises(ValueError, match="separate"):
        save_review(source, annotations, source)
    output = tmp_path / "existing-review"
    save_review(source, annotations, output)
    wrong = copy.deepcopy(annotations)
    wrong["placement_id"] = "another-pile"
    with pytest.raises(ValueError, match="another source or placement"):
        save_review(source, wrong, output)
    assert (unrelated / "keep.txt").read_text() == "keep"


def test_failed_review_directory_swap_restores_previous(example, monkeypatch):
    from pathlib import Path

    source, annotations = example
    save_review(source, annotations)
    output = source.parent / "review" / source.name
    original = (output / "polygons.json").read_bytes()
    rename = Path.rename
    def fail_publish(path, target):
        if path.name == "result":
            raise OSError("simulated publish failure")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        save_review(source, annotations)
    assert (output / "polygons.json").read_bytes() == original
    assert not list(output.parent.glob(".review-update-*"))
