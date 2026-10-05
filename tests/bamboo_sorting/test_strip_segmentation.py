import copy
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.bamboo_sorting.placement_review import build_review, polygon_mask
from lerobot.bamboo_sorting.placement_sequence import PlacementSequence
from lerobot.bamboo_sorting.strip_segmentation_data import (
    audit_dataset, export_reviewed_piles, read_json, write_json,
)


@pytest.fixture
def selection(tmp_path):
    root = tmp_path / "capture"
    points = [[[8, 25], [85, 25], [85, 32], [8, 32]],
              [[42, 8], [49, 8], [49, 55], [42, 55]],
              [[15, 40], [80, 40], [80, 47], [15, 47]]]
    entries = []
    for name, split in (("pile_a", "train"), ("pile_b", "validation")):
        source = root / name
        sequence = PlacementSequence(source, placement_id=name, split=split, configuration={})
        frame = np.zeros((64, 96, 3), np.uint8)
        frame[:] = [10, 20, 100]  # distinguish channel order
        sequence.append(frame)
        for polygon in points:
            frame[polygon_mask(polygon, frame.shape)] = [210, 170, 105]
            sequence.append(frame)
        sequence.finish("completed")
        annotations = {"placement_id": name, "image_size": [96, 64],
                       "placement_from_above_confirmed": True, "older_strips_stationary_confirmed": True,
                       "placement_confirmation_source": "synthetic fixture",
                       "objects": [{"id": f"strip_{i:03d}", "frame_index": i, "polygon": p,
                                    "reviewed": True, "truncated": False} for i, p in enumerate(points, 1)]}
        build_review(source, annotations, root / "review" / name)
        entries.append({"placement_id": name, "split": split})
    path = tmp_path / "selection.json"
    write_json(path, {"review_root": "capture/review", "placements": entries,
                      "image_size": [96, 64], "strip_dimensions_mm": {"length": 400, "width": 25, "thickness": 8}})
    return path


def test_export_raw_rgb_visible_regions_classes_and_empty(selection, tmp_path):
    output = tmp_path / "export"
    report = export_reviewed_piles(selection, output)
    assert report["splits"]["train"] == {"groups": 1, "images": 4, "empty_images": 1,
                                         "instances": 6, "top_strip": 3, "covered_strip": 3}
    records = read_json(output / "dataset.json")["samples"]
    assert records[0]["segments"] == [] and records[0]["empty_baseline"]
    second = records[2]
    assert [x["class_id"] for x in second["segments"]] == [1, 0]
    mask = cv2.imread(str(output / second["instance_map"]), -1)
    assert mask[26, 44] == 2  # intersection belongs to newest/top instance
    assert mask[26, 12] == mask[26, 75] == 1  # disconnected lower strip stays one instance
    raw = selection.parent / "capture/pile_a/frame_002.png"
    assert raw.read_bytes() == (output / second["image"]).read_bytes()
    assert audit_dataset(output)["passed"]
    with pytest.raises(FileExistsError):
        export_reviewed_piles(selection, output)


@pytest.mark.parametrize("corruption", ["duplicate", "split", "unreviewed", "missing_mask", "overlap"])
def test_export_rejects_bad_lineage_or_masks_without_partial_output(selection, tmp_path, corruption):
    spec = read_json(selection)
    review = selection.parent / "capture/review/pile_a"
    if corruption == "duplicate":
        spec["placements"].append(copy.deepcopy(spec["placements"][0]))
        write_json(selection, spec)
    elif corruption == "split":
        spec["placements"][0]["split"] = "validation"
        write_json(selection, spec)
    elif corruption == "unreviewed":
        a = read_json(review / "polygons.json")
        a["objects"][0]["reviewed"] = False
        write_json(review / "polygons.json", a)
    elif corruption == "missing_mask":
        (review / "frame_002_strip_001_visible.png").unlink()
    else:
        mask = cv2.imread(str(review / "frame_002_strip_001_visible.png"), -1)
        mask[26, 44] = 255
        cv2.imwrite(str(review / "frame_002_strip_001_visible.png"), mask)
    with pytest.raises(ValueError):
        export_reviewed_piles(selection, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_truncated_stage_is_explicitly_excluded(selection, tmp_path):
    path = selection.parent / "capture/review/pile_a/frame_003_labels.json"
    d = read_json(path)
    d["full_frame_mask_eligible"] = False
    d["instances"][-1]["truncated"] = True
    write_json(path, d)
    report = export_reviewed_piles(selection, tmp_path / "data")
    assert report["splits"]["train"]["images"] == 3
    assert report["excluded_frames"] == [{"placement_id": "pile_a", "frame_index": 3,
                                           "reason": "full_frame_mask_ineligible"}]


def test_adapter_empty_foreground_and_rgb(selection, tmp_path):
    pytest.importorskip("transformers")
    from transformers import Mask2FormerImageProcessor
    from lerobot.bamboo_sorting.strip_segmentation_training import StripDataset, collate_samples

    export_reviewed_piles(selection, tmp_path / "data")
    dataset = StripDataset(tmp_path / "data", "train")
    assert dataset[0]["image"][0, 0].tolist() == [10, 20, 100]  # preserve original RGB order
    assert dataset[0]["mask_labels"].shape == (0, 64, 96)
    batch = collate_samples([dataset[0], dataset[2]], Mask2FormerImageProcessor(do_resize=False))
    assert batch["pixel_values"].shape == (2, 3, 64, 96)
    assert batch["class_labels"][0].numel() == 0
    assert batch["class_labels"][1].tolist() == [1, 0]


def test_matching_penalizes_class_errors_duplicates_and_empty():
    pytest.importorskip("scipy")
    from lerobot.bamboo_sorting.strip_segmentation_training import match_instances

    a = np.zeros((8, 8), bool)
    a[1:3, 1:5] = True
    assert match_instances([(0, a), (0, a)], [(0, a)])[0] == {"tp": 1, "fp": 1, "fn": 0}
    wrong = match_instances([(1, a)], [(0, a)])
    assert wrong[0]["fn"] == 1 and wrong[1]["fp"] == 1
    assert match_instances([(0, a)], [])[0]["fp"] == 1
    assert match_instances([], [])[0] == {"tp": 0, "fp": 0, "fn": 0}


def test_independent_test_export_frozen_evaluation_and_review(selection, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("scipy")
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, SwinConfig
    from lerobot.bamboo_sorting.strip_segmentation_data import CLASSES

    # A separate synthetic test group, never relabel an actual training group.
    spec = read_json(selection)
    spec["placements"] = [{"placement_id": "pile_a", "split": "test"}]
    write_json(selection, spec)
    review = selection.parent / "capture/review/pile_a"
    for path in [selection.parent / "capture/pile_a/sequence.json", review / "review.json",
                 *review.glob("frame_*_labels.json")]:
        value = read_json(path)
        value["split"] = "test"
        write_json(path, value)
    data_path = tmp_path / "test_data"
    audit = export_reviewed_piles(selection, data_path, test_only=True)
    assert audit["splits"]["test"]["images"] == 4
    with pytest.raises(ValueError, match="both train and validation"):
        audit_dataset(data_path)
    original = tmp_path / "training_data/dataset.json"
    original.parent.mkdir()
    write_json(original, {"samples": [{"placement_id": "another_group"}]})
    backbone = SwinConfig(embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                          window_size=2, out_features=["stage1", "stage2", "stage3", "stage4"])
    config = Mask2FormerConfig(backbone_config=backbone.to_dict(), feature_size=32, mask_feature_size=32,
                              hidden_dim=32, encoder_layers=1, decoder_layers=2, num_attention_heads=4,
                              encoder_feedforward_dim=64, dim_feedforward=64, num_queries=5, train_num_points=32,
                              id2label={c["id"]: c["name"] for c in CLASSES})
    best = tmp_path / "training_run/best"
    Mask2FormerForUniversalSegmentation(config).save_pretrained(best)
    Mask2FormerImageProcessor(do_resize=False).save_pretrained(best)
    write_json(best / "strip_contract.json", {"classes": CLASSES, "resize": False,
               "image_size": [96, 64], "strip_dimensions_mm": spec["strip_dimensions_mm"]})
    write_json(best.parent / "complete.json", {"best_reloaded_and_evaluated": True, "best_epoch": 15})
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in best.iterdir()}

    def prohibited(*args, **kwargs):
        raise AssertionError("evaluation must not construct an optimizer or backpropagate")
    monkeypatch.setattr(torch.optim, "AdamW", prohibited)
    monkeypatch.setattr(torch.Tensor, "backward", prohibited)
    directory = Path(__file__).resolve().parents[2] / "examples/phone_to_auboi10"

    def load(name):
        module_spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    evaluation = load("evaluate_strip_segmentation")
    output = tmp_path / "evaluation"
    args = ["--dataset", str(data_path), "--training-manifest", str(original),
            "--model-path", str(best), "--output", str(output)]
    assert evaluation.main(args) == 0
    assert read_json(output / "complete.json")["optimizer_steps"] == 0
    assert len(list((output / "test_masks").glob("*.png"))) == 4
    assert before == {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in best.iterdir()}
    reviewer = load("review_strip_predictions")
    summary, samples = reviewer.build_review(data_path, output, review.parent, split="test")
    page = reviewer.render_page(summary, samples)
    assert summary["split"] == "test" and summary["images"] == 4
    assert "独立测试预测轮廓复核" in page and "4图／1组" in page
    assert "__DATA__" not in page
    with pytest.raises(FileExistsError):
        evaluation.main(args)
    write_json(original, {"samples": [{"placement_id": "pile_a"}]})
    with pytest.raises(ValueError, match="overlap"):
        evaluation.main(args)


def test_synthetic_model_forward_backward_and_training_entry(selection, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("scipy")
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, SwinConfig
    from lerobot.bamboo_sorting.strip_segmentation_training import StripDataset, collate_samples, forward_loss

    torch.set_num_threads(2)
    torch.manual_seed(4)
    backbone = SwinConfig(embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                          window_size=2, out_features=["stage1", "stage2", "stage3", "stage4"])
    config = Mask2FormerConfig(backbone_config=backbone.to_dict(), feature_size=32, mask_feature_size=32,
                              hidden_dim=32, encoder_layers=1, decoder_layers=2, num_attention_heads=4,
                              encoder_feedforward_dim=64, dim_feedforward=64, num_queries=5,
                              train_num_points=32, num_labels=2)
    model = Mask2FormerForUniversalSegmentation(config)
    processor = Mask2FormerImageProcessor(do_resize=False)
    export_reviewed_piles(selection, tmp_path / "data")
    data = StripDataset(tmp_path / "data", "train")
    for indices in ((0, 0), (0, 2), (2, 3)):
        batch = collate_samples([data[i] for i in indices], processor)
        batch.pop("samples")
        _, loss = forward_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.class_predictor.weight.grad).all()
        model.zero_grad(set_to_none=True)
    base = tmp_path / "base"
    # Reproduce the real transfer: COCO's 80 classes become two strip classes.
    pretrained_config = Mask2FormerConfig.from_dict(config.to_dict())
    pretrained_config.num_labels = 80
    Mask2FormerForUniversalSegmentation(pretrained_config).save_pretrained(base)
    processor.save_pretrained(base)
    path = Path(__file__).resolve().parents[2] / "examples/phone_to_auboi10/train_strip_segmentation.py"
    spec = importlib.util.spec_from_file_location("strip_train_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--stage", "train", "--dataset", str(tmp_path / "data"),
                        "--base-path", str(base), "--output", str(tmp_path / "run"),
                        "--epochs", "1", "--eval-every", "1", "--batch-size", "2"]) == 0
    complete = read_json(tmp_path / "run/complete.json")
    assert complete["steps"] == 2 and complete["best_reloaded_and_evaluated"]
    assert (tmp_path / "run/best_validation.json").exists()
    loading = read_json(tmp_path / "run/loading_info.json")
    assert set(loading["mismatched_keys"]) == {
        "class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"}
    restored = Mask2FormerForUniversalSegmentation.from_pretrained(tmp_path / "run/best")
    assert restored.config.num_labels == 2
    assert restored.criterion.empty_weight.shape == (3,)
