import numpy as np
import pytest

from lerobot.bamboo_sorting.joint_target import (
    TARGET_FRAME_SCHEMA, TARGET_IMAGE_KEY, prepare_target_images,
    selected_instance_mask, target_contract_record,
)


def inputs():
    images = {k: np.full((480, 640, 3), 37, np.uint8) for k in ("global_rgb", "grasp_rgb")}
    mask = np.zeros((480, 640), bool)
    mask[10:20, 10:80] = True
    mask[10:20, 100:130] = True
    annotation = dict(schema=TARGET_FRAME_SCHEMA, frame_id="frame_a", target_id="strip_a", status="visible", reviewed=True)
    return images, mask, annotation


def test_visible_fragments_keep_gap_and_original_inputs():
    images, mask, annotation = inputs()
    originals = {k: v.copy() for k, v in images.items()}
    result = prepare_target_images(images, mask, annotation, frame_id="frame_a", target_id="strip_a")
    marked = result["images"][TARGET_IMAGE_KEY]
    assert np.array_equal(marked[10, 10], [255, 0, 255])
    assert np.array_equal(marked[10, 100], [255, 0, 255])
    assert np.all(marked[10:20, 80:100] == 37)  # No amodal bridge across the occluder.
    assert np.all(marked[12:18, 12:78] == 37)
    for k in images:
        assert np.array_equal(images[k], originals[k])
    assert np.array_equal(result["images"]["grasp_rgb"], originals["grasp_rgb"])
    assert "global_rgb" not in result["images"] and result["preview_only"] is False


@pytest.mark.parametrize("field,value,error", [
    ("frame_id", "old_frame", "another frame"),
    ("target_id", "other_strip", "identity changed"),
    ("status", "occluded", "target unavailable"),
    ("status", "lost", "target unavailable"),
    ("reviewed", False, "requires review"),
])
def test_no_silent_reuse_or_target_switch(field, value, error):
    images, mask, annotation = inputs()
    annotation[field] = value
    with pytest.raises(ValueError, match=error):
        prepare_target_images(images, mask, annotation, frame_id="frame_a", target_id="strip_a")


def test_draft_preview_and_empty_mask_are_distinct():
    images, mask, annotation = inputs()
    annotation["reviewed"] = False
    result = prepare_target_images(images, mask, annotation, frame_id="frame_a", target_id="strip_a", allow_draft=True)
    assert result["preview_only"] is True
    with pytest.raises(ValueError, match="nonempty"):
        prepare_target_images(images, np.zeros_like(mask), annotation, frame_id="frame_a", target_id="strip_a", allow_draft=True)


def test_instance_ids_are_not_class_ids_and_selection_changes_pixels():
    images, mask, annotation = inputs()
    instance_map = np.zeros((480, 640), np.uint16)
    instance_map[mask] = 7
    instance_map[100:110, 100:140] = 12
    segments = [{"id": 7, "label_id": 0}, {"id": 12, "label_id": 0}]
    a = selected_instance_mask(instance_map, segments, 7)
    b = selected_instance_mask(instance_map, segments, 12)
    assert np.array_equal(a, mask) and not np.any(a & b)
    outputs = [prepare_target_images(images, m, annotation, frame_id="frame_a", target_id="strip_a")["images"][TARGET_IMAGE_KEY] for m in (a, b)]
    assert not np.array_equal(*outputs)  # Input changes; does not prove a model follows it.
    with pytest.raises(ValueError, match="IDs"):
        selected_instance_mask(instance_map, segments, 0)


def test_legacy_adapter_rejects_target_contract_before_loading_a_model():
    from lerobot.bamboo_sorting.smolvla_joint_adapter import SmolVLAJointOfflineAdapter
    with pytest.raises(ValueError, match="contract is required"):
        SmolVLAJointOfflineAdapter(None, None, None, contract=target_contract_record())


def test_review_page_uses_requested_saved_frame_and_preserves_sources(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import runpy

    import cv2

    opened = []

    class SavedVideo:
        def __init__(self, path):
            self.path, self.index, self.released = path, None, False
            opened.append(self)

        def set(self, prop, value):
            assert prop == cv2.CAP_PROP_POS_FRAMES
            self.index = value

        def read(self):
            assert self.index == 25
            return True, np.full((480, 640, 3), 37, np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", SavedVideo)
    script = Path(__file__).resolve().parents[2] / "examples/phone_to_auboi10/review_joint_target.py"
    build_page = runpy.run_path(str(script))["build_page"]
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps({
        "source_run": "saved_run", "target_id": "strip_a", "frames": [{
            "schema": TARGET_FRAME_SCHEMA, "frame_id": "saved_run/global_rgb/25",
            "target_id": "strip_a", "status": "visible", "reviewed": False,
            "video_frame_index": 25, "label": "saved frame",
            "visible_polygons": [[[10, 10], [30, 10], [30, 20], [10, 20]]],
        }],
    }))
    original = source.read_bytes()
    output = tmp_path / "index.html"
    result = build_page(source, output)
    assert result["frames"] == result["draft_frames"] == 1
    assert result["training_or_inference_run"] is False
    assert "saved_run/global_rgb/25" in output.read_text()
    assert "data:image/png;base64," in output.read_text()
    assert len(opened) == 2 and all(cap.released for cap in opened)
    assert {Path(cap.path).name for cap in opened} == {"global_rgb.mp4", "grasp_rgb.mp4"}
    assert source.read_bytes() == original
    page = output.read_bytes()
    with pytest.raises(FileExistsError):
        build_page(source, output)
    assert output.read_bytes() == page and len(opened) == 2
    confirmed = json.loads(source.read_text())
    confirmed["frames"][0]["reviewed"] = True
    source.write_text(json.dumps(confirmed))
    (tmp_path / "tracking.json").write_text(json.dumps({
        "checkpoints_before_human_reset": [{"time_s": 40, "iou": 0, "track_available": False}],
    }))
    updated = build_page(source, output, update=True)
    assert updated["draft_frames"] == 0
    assert 'src="tracking.mp4"' in output.read_text()
    assert "已丢失" in output.read_text()
    assert json.loads(source.read_text()) == confirmed


def test_flow_moves_visible_fragments_without_bridging_the_gap():
    import cv2
    from lerobot.bamboo_sorting.joint_target_tracking import TargetMaskFlow, mask_overlap

    rng = np.random.default_rng(42)
    gray = rng.integers(40, 230, (96, 128), dtype=np.uint8)
    # Resolvable texture, rather than independent single-pixel noise that aliases
    # in the optical-flow image pyramid.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    mask = np.zeros((96, 128), bool)
    mask[30:50, 20:50] = True
    mask[30:50, 65:100] = True
    move = np.float32([[1, 0, 3], [0, 1, 2]])
    shifted = cv2.warpAffine(rgb, move, (128, 96))
    expected = cv2.warpAffine(mask.astype(np.uint8), move, (128, 96)).astype(bool)
    tracker = TargetMaskFlow()
    tracker.seed(rgb, mask)
    candidate, detail = tracker.step(shifted)
    assert detail["status"] == "draft" and detail["reviewed"] is False
    assert mask_overlap(candidate, expected)["iou"] > 0.9
    assert not candidate[32:52, 55:65].any()


def test_flow_disappearance_requires_explicit_reseed():
    from lerobot.bamboo_sorting.joint_target_tracking import TargetMaskFlow, mask_overlap

    rgb = np.full((96, 128, 3), 200, np.uint8)
    mask = np.zeros((96, 128), bool)
    mask[20:50, 20:60] = True
    tracker = TargetMaskFlow()
    tracker.seed(rgb, mask)
    candidate, detail = tracker.step(np.zeros_like(rgb))
    assert candidate is None and detail["status"] == "lost"
    candidate, detail = tracker.step(rgb)
    assert candidate is None and detail["reason"] == "explicit_seed_required"
    assert mask_overlap(candidate, mask) == {"iou": 0.0, "recall": 0.0, "track_available": False}
    tracker.seed(rgb, mask)
    candidate, detail = tracker.step(rgb)
    assert mask_overlap(candidate, mask)["iou"] == 1.0


def test_segmentation_png_ids_and_oracle_overlap_are_not_class_selection():
    from pathlib import Path
    import runpy
    from lerobot.bamboo_sorting.joint_target import compare_target_instances

    script = Path(__file__).resolve().parents[2] / "examples/phone_to_auboi10/predict_strip_images.py"
    pack = runpy.run_path(str(script))["instance_png"]
    segmentation = np.array([[-1, 0, 0], [9, 9, -1]])
    png, segments = pack(segmentation, [{"id": 0, "label_id": 1, "score": 0.7},
                                       {"id": 9, "label_id": 0, "score": 0.9}])
    assert png.tolist() == [[0, 1, 1], [2, 2, 0]]
    assert png.dtype == np.uint16 and segments[0]["class_id"] == 1
    reference = np.array([[False, True, True], [False, False, False]])
    result = compare_target_instances(png, segments, reference)
    assert result["best"]["mask_id"] == 1 and result["best"]["class_id"] == 1
    assert result["best"]["iou"] == result["best"]["recall"] == result["best"]["precision"] == 1.0
    assert compare_target_instances(np.zeros_like(png), [], reference)["best"] is None


def test_segmentation_review_does_not_invent_truth_for_unreviewed_frames(tmp_path):
    import cv2
    import json
    from pathlib import Path
    import runpy

    root = tmp_path / "segmentation"
    (root / "result").mkdir(parents=True)
    cv2.imwrite(str(root / "raw.png"), np.full((480, 640, 3), 37, np.uint8))
    mask = np.zeros((480, 640), np.uint16)
    mask[10:20, 10:30] = 7
    cv2.imwrite(str(root / "result/mask.png"), mask)
    rows = [{"id": f"frame_{i}", "video_frame_index": i, "video_time_s": i / 25,
             "image": "raw.png", "instance_map": "mask.png",
             "segments": [{"mask_id": 7, "class_id": 0, "score": 0.9, "pixels": 200}]}
            for i in (0, 25)]
    (root / "result/predictions.json").write_text(json.dumps({"samples": rows}))
    (root / "comparison.json").write_text(json.dumps({"samples": [
        {"id": "frame_0", "comparison": {"best": {"iou": 0.8, "recall": 0.9}}},
        {"id": "frame_25", "comparison": None},
    ]}))
    spec = {"frames": [{"video_frame_index": 0, "reviewed": True,
                        "visible_polygons": [[[10, 10], [30, 10], [30, 20], [10, 20]]]}]}
    script = Path(__file__).resolve().parents[2] / "examples/phone_to_auboi10/review_joint_target.py"
    render = runpy.run_path(str(script))["segmentation_section"]
    page = render(tmp_path, spec, lambda _: "data:image/png;base64,preview")
    assert "共2帧" in page and "1个已确认关键帧" in page
    assert "本帧尚未人工标注，不计算IoU" in page
    assert "实例7：top_strip" in page
