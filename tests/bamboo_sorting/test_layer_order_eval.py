"""Static layer-order evaluation must keep uncertainty and image identity visible."""

import hashlib
import json

import pytest

from lerobot.bamboo_sorting.layer_order_eval import evaluate, validate_truth
from lerobot.bamboo_sorting.rgb_gate import FROZEN_CAMERA_SET_V2_SHA256


def _scene(tmp_path, scene_id="scene-1", relation="first_above_second"):
    images = {}
    hashes = {}
    for name in ("global_rgb", "grasp_rgb"):
        path = tmp_path / f"{scene_id}-{name}.jpg"
        path.write_bytes(f"{scene_id}-{name}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        images[name] = {"path": path.name, "sha256": digest}
        hashes[name] = digest
    truth = {
        "schema_version": "TopLayerTruthV1",
        "scene_id": scene_id,
        "placement_id": scene_id,
        "split": "test",
        "camera_set_sha256": FROZEN_CAMERA_SET_V2_SHA256,
        "images": images,
        "strips": [{"id": "A", "contact_visible": True}, {"id": "B", "contact_visible": True}],
        "relations": [{"first": "A", "second": "B", "relation": relation}],
    }
    prediction = {
        "schema_version": "TopLayerPredictionV1",
        "scene_id": scene_id,
        "camera_set_sha256": FROZEN_CAMERA_SET_V2_SHA256,
        "image_sha256": hashes,
        "relations": truth["relations"],
        "selected_id": "A",
    }
    return truth, prediction


def _evaluate(tmp_path, truths, predictions):
    truth_path = tmp_path / "truth.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    truth_path.write_text("".join(json.dumps(item) + "\n" for item in truths), encoding="utf-8")
    prediction_path.write_text("".join(json.dumps(item) + "\n" for item in predictions), encoding="utf-8")
    return evaluate(truth_path, prediction_path)


def test_reports_layer_accuracy_and_rejects_lower_selection(tmp_path):
    truth, prediction = _scene(tmp_path)
    second_truth, second_prediction = _scene(tmp_path, "scene-2")
    second_prediction["selected_id"] = "B"
    report = _evaluate(tmp_path, [truth, second_truth], [prediction, second_prediction])
    assert report["pairs_correct"] == 2
    assert report["correct_top_selection"] == 1
    assert report["wrong_selection"] == 1
    assert report["unsupported_selection"] == 1
    assert report["top_selection_precision"] == 0.5
    assert report["physical_grasp_success_evaluated"] is False


def test_uncertain_relation_requires_abstention(tmp_path):
    truth, prediction = _scene(tmp_path, relation="uncertain")
    prediction["relations"] = [{"first": "A", "second": "B", "relation": "first_above_second"}]
    prediction["selected_id"] = None
    report = _evaluate(tmp_path, [truth], [prediction])
    assert report["pair_accuracy"] is None
    assert report["uncertain_overclaims"] == 1
    assert report["correct_abstain"] == 1


def test_image_mismatch_and_missing_pair_fail_closed(tmp_path):
    truth, prediction = _scene(tmp_path)
    prediction["image_sha256"]["global_rgb"] = "0" * 64
    with pytest.raises(ValueError, match="image hashes do not match"):
        _evaluate(tmp_path, [truth], [prediction])

    truth, prediction = _scene(tmp_path)
    prediction["relations"] = []
    with pytest.raises(ValueError, match="cover every unordered strip pair"):
        _evaluate(tmp_path, [truth], [prediction])


def test_actual_image_bytes_must_match_manifest(tmp_path):
    truth, prediction = _scene(tmp_path)
    (tmp_path / "scene-1-global_rgb.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image SHA-256 mismatch"):
        _evaluate(tmp_path, [truth], [prediction])


def test_three_strip_chain_checks_top_choice_and_truth_without_predictions(tmp_path):
    truth, prediction = _scene(tmp_path)
    truth["strips"].append({"id": "C", "contact_visible": True})
    truth["relations"] = [
        {"first": "A", "second": "B", "relation": "first_above_second"},
        {"first": "A", "second": "C", "relation": "separate"},
        {"first": "B", "second": "C", "relation": "first_above_second"},
    ]
    prediction["relations"] = truth["relations"]
    report = _evaluate(tmp_path, [truth], [prediction])
    assert report["correct_top_selection"] == 1
    assert report["pairs_correct"] == 3

    truth_path = tmp_path / "truth.jsonl"
    validation = validate_truth(truth_path)
    assert validation["scenes_by_strip_count"] == {"2": 0, "3": 1}
    assert validation["scenes_by_split"] == {"development": 0, "test": 1}

    prediction["selected_id"] = "B"
    assert _evaluate(tmp_path, [truth], [prediction])["wrong_selection"] == 1


def test_three_strip_uncertainty_preserves_independent_top_candidate(tmp_path):
    truth, prediction = _scene(tmp_path, relation="uncertain")
    truth["strips"].append({"id": "C", "contact_visible": True})
    truth["relations"].extend([
        {"first": "A", "second": "C", "relation": "separate"},
        {"first": "B", "second": "C", "relation": "separate"},
    ])
    prediction["relations"] = truth["relations"]
    prediction["selected_id"] = "C"
    report = _evaluate(tmp_path, [truth], [prediction])
    assert report["correct_top_selection"] == 1
    assert report["unsupported_selection"] == 0

    prediction["selected_id"] = None
    assert _evaluate(tmp_path, [truth], [prediction])["unnecessary_abstain"] == 1


def test_three_strip_cycle_has_no_confirmed_top_candidate(tmp_path):
    truth, prediction = _scene(tmp_path)
    truth["strips"].append({"id": "C", "contact_visible": True})
    truth["relations"] = [
        {"first": "A", "second": "B", "relation": "first_above_second"},
        {"first": "A", "second": "C", "relation": "second_above_first"},
        {"first": "B", "second": "C", "relation": "first_above_second"},
    ]
    prediction["relations"] = truth["relations"]
    prediction["selected_id"] = None
    assert _evaluate(tmp_path, [truth], [prediction])["correct_abstain"] == 1


def test_missing_selection_is_not_counted_as_abstention(tmp_path):
    truth, prediction = _scene(tmp_path, relation="uncertain")
    del prediction["selected_id"]
    with pytest.raises(ValueError, match="selected_id must be explicit"):
        _evaluate(tmp_path, [truth], [prediction])


def test_guessing_a_top_strip_despite_predicted_uncertainty_is_reported(tmp_path):
    truth, prediction = _scene(tmp_path)
    prediction["relations"] = [{"first": "A", "second": "B", "relation": "uncertain"}]
    report = _evaluate(tmp_path, [truth], [prediction])
    assert report["correct_top_selection"] == 1
    assert report["unsupported_selection"] == 1


def test_same_physical_placement_cannot_be_reused_across_splits(tmp_path):
    truth, prediction = _scene(tmp_path)
    second_truth, second_prediction = _scene(tmp_path, "scene-2")
    second_truth["split"] = "development"
    second_truth["placement_id"] = truth["placement_id"]
    with pytest.raises(ValueError, match="duplicate physical placement_id"):
        _evaluate(tmp_path, [truth, second_truth], [prediction, second_prediction])


def test_identical_image_pair_cannot_be_counted_twice(tmp_path):
    truth, prediction = _scene(tmp_path)
    second_truth, second_prediction = _scene(tmp_path, "scene-2")
    second_truth["images"] = truth["images"]
    with pytest.raises(ValueError, match="duplicate image pair"):
        _evaluate(tmp_path, [truth, second_truth], [prediction, second_prediction])
