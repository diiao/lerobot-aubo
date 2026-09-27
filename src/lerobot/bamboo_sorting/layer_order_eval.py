"""Read-only, human-grounded evaluation of top-layer decisions on static RGB scenes."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

from .rgb_gate import FROZEN_CAMERA_SET_V2_SHA256

TRUTH_SCHEMA = "TopLayerTruthV1"
PREDICTION_SCHEMA = "TopLayerPredictionV1"
REPORT_SCHEMA = "TopLayerEvaluationV1"
TRUTH_VALIDATION_SCHEMA = "TopLayerTruthValidationV1"
IMAGE_KEYS = frozenset({"global_rgb", "grasp_rgb"})
RELATIONS = frozenset({"first_above_second", "second_above_first", "separate", "uncertain"})
SPLITS = frozenset({"development", "test"})


def _objects(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _camera_set(record: dict) -> None:
    if record.get("camera_set_sha256") != FROZEN_CAMERA_SET_V2_SHA256:
        raise ValueError("camera_set_sha256 must equal frozen CameraSetV2")


def _image_hashes(record: dict, root: Path, *, verify_files: bool) -> dict[str, str]:
    images = record.get("images")
    if not isinstance(images, dict) or set(images) != IMAGE_KEYS:
        raise ValueError("images must contain exactly global_rgb and grasp_rgb")
    hashes = {}
    for name in sorted(IMAGE_KEYS):
        item = images[name]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"{name} requires path and sha256")
        reference = _nonempty(item["path"], f"{name}.path")
        expected = _digest(item["sha256"], f"{name}.sha256")
        if verify_files:
            path = Path(reference)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise ValueError(f"{name} image does not exist: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"{name} image SHA-256 mismatch: {path}")
        hashes[name] = expected
    return hashes


def _relations(value: object, strip_ids: set[str]) -> dict[tuple[str, str], str]:
    if not isinstance(value, list):
        raise ValueError("relations must be a list")
    result = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"first", "second", "relation"}:
            raise ValueError("each relation requires first, second and relation")
        first = _nonempty(item["first"], "relation.first")
        second = _nonempty(item["second"], "relation.second")
        key = (first, second)
        if first >= second or first not in strip_ids or second not in strip_ids or key in result:
            raise ValueError("relation pairs must be unique, known and lexicographically ordered")
        if not isinstance(item["relation"], str) or item["relation"] not in RELATIONS:
            raise ValueError(f"invalid relation for {first}, {second}")
        result[key] = item["relation"]
    if set(result) != set(combinations(sorted(strip_ids), 2)):
        raise ValueError("relations must cover every unordered strip pair exactly once")
    return result


def _truth(record: dict, root: Path, *, verify_files: bool) -> dict:
    if record.get("schema_version") != TRUTH_SCHEMA:
        raise ValueError("unsupported truth schema")
    scene_id = _nonempty(record.get("scene_id"), "scene_id")
    placement_id = _nonempty(record.get("placement_id"), "placement_id")
    if not isinstance(record.get("split"), str) or record["split"] not in SPLITS:
        raise ValueError(f"{scene_id}: split must be development or test")
    _camera_set(record)
    hashes = _image_hashes(record, root, verify_files=verify_files)
    strips = record.get("strips")
    if not isinstance(strips, list) or len(strips) not in (2, 3):
        raise ValueError(f"{scene_id}: exactly two or three strips are required")
    contact_visible = {}
    for strip in strips:
        if not isinstance(strip, dict) or set(strip) != {"id", "contact_visible"}:
            raise ValueError(f"{scene_id}: each strip requires id and contact_visible")
        strip_id = _nonempty(strip["id"], "strip.id")
        if strip_id in contact_visible or not isinstance(strip["contact_visible"], bool):
            raise ValueError(f"{scene_id}: strip ids must be unique and contact_visible must be bool")
        contact_visible[strip_id] = strip["contact_visible"]
    relations = _relations(record.get("relations"), set(contact_visible))
    return {"scene_id": scene_id, "placement_id": placement_id, "split": record["split"], "hashes": hashes,
            "contact_visible": contact_visible, "relations": relations}


def _prediction(record: dict, truth: dict) -> dict:
    if record.get("schema_version") != PREDICTION_SCHEMA:
        raise ValueError("unsupported prediction schema")
    if record.get("scene_id") != truth["scene_id"]:
        raise ValueError("prediction scene_id does not match truth")
    _camera_set(record)
    if record.get("image_sha256") != truth["hashes"]:
        raise ValueError(f"{truth['scene_id']}: prediction image hashes do not match truth")
    if "selected_id" not in record:
        raise ValueError(f"{truth['scene_id']}: selected_id must be explicit, including null for abstention")
    strip_ids = set(truth["contact_visible"])
    selected = record.get("selected_id")
    if selected is not None and (not isinstance(selected, str) or selected not in strip_ids):
        raise ValueError(f"{truth['scene_id']}: selected_id must be a known strip or null")
    return {"selected_id": selected, "relations": _relations(record.get("relations"), strip_ids)}


def _eligible(truth: dict) -> set[str]:
    candidates = {strip_id for strip_id, visible in truth["contact_visible"].items() if visible}
    for (first, second), relation in truth["relations"].items():
        if relation == "first_above_second":
            candidates.discard(second)
        elif relation == "second_above_first":
            candidates.discard(first)
        elif relation == "uncertain":
            candidates.discard(first)
            candidates.discard(second)
    return candidates


def _load_truths(truth_path: Path) -> dict[str, dict]:
    """Validate image bytes and scene identities before using any labels."""
    scenes = {}
    placements = set()
    image_pairs = set()
    for record in _objects(truth_path):
        truth = _truth(record, truth_path.parent, verify_files=True)
        if truth["scene_id"] in scenes:
            raise ValueError(f"duplicate truth scene_id: {truth['scene_id']}")
        if truth["placement_id"] in placements:
            raise ValueError(f"duplicate physical placement_id: {truth['placement_id']}")
        image_pair = tuple(sorted(truth["hashes"].items()))
        if image_pair in image_pairs:
            raise ValueError(f"duplicate image pair: {truth['scene_id']}")
        scenes[truth["scene_id"]] = truth
        placements.add(truth["placement_id"])
        image_pairs.add(image_pair)
    return scenes


def validate_truth(truth_path: str | Path) -> dict:
    """Check a human label file without requiring model predictions."""
    scenes = _load_truths(Path(truth_path))
    return {
        "schema_version": TRUTH_VALIDATION_SCHEMA,
        "scenes": len(scenes),
        "scenes_by_split": {split: sum(scene["split"] == split for scene in scenes.values())
                            for split in sorted(SPLITS)},
        "scenes_by_strip_count": {str(count): sum(len(scene["contact_visible"]) == count
                                                  for scene in scenes.values()) for count in (2, 3)},
        "uncertain_pairs": sum(relation == "uncertain" for scene in scenes.values()
                               for relation in scene["relations"].values()),
        "physical_grasp_success_evaluated": False,
    }


def evaluate(truth_path: str | Path, prediction_path: str | Path, *, split: str = "test") -> dict:
    """Compare VLM decisions with human labels; never infer physical grasp success."""
    if split not in SPLITS:
        raise ValueError("split must be development or test")
    scenes = _load_truths(Path(truth_path))
    prediction_path = Path(prediction_path)
    predictions = {}
    for record in _objects(prediction_path):
        scene_id = _nonempty(record.get("scene_id"), "prediction.scene_id")
        if scene_id not in scenes or scene_id in predictions:
            raise ValueError(f"unknown or duplicate prediction scene_id: {scene_id}")
        predictions[scene_id] = _prediction(record, scenes[scene_id])
    selected_scenes = {key: value for key, value in scenes.items() if value["split"] == split}
    if not selected_scenes or not set(selected_scenes) <= set(predictions):
        raise ValueError(f"{split} scenes or their predictions are missing")

    counts = {name: 0 for name in (
        "pairs_decidable", "pairs_correct", "uncertain_pairs", "uncertain_overclaims",
        "selected", "correct_top_selection", "wrong_selection", "unsupported_selection",
        "correct_abstain", "unnecessary_abstain",
    )}
    for scene_id, truth in selected_scenes.items():
        prediction = predictions[scene_id]
        for pair, label in truth["relations"].items():
            proposed = prediction["relations"][pair]
            if label == "uncertain":
                counts["uncertain_pairs"] += 1
                counts["uncertain_overclaims"] += proposed != "uncertain"
            else:
                counts["pairs_decidable"] += 1
                counts["pairs_correct"] += proposed == label
        eligible = _eligible(truth)
        chosen = prediction["selected_id"]
        if chosen is None:
            counts["correct_abstain" if not eligible else "unnecessary_abstain"] += 1
        else:
            counts["selected"] += 1
            counts["correct_top_selection" if chosen in eligible else "wrong_selection"] += 1
            predicted_eligible = _eligible({
                "contact_visible": {strip_id: True for strip_id in truth["contact_visible"]},
                "relations": prediction["relations"],
            })
            counts["unsupported_selection"] += chosen not in predicted_eligible
    return {"schema_version": REPORT_SCHEMA, "split": split, "scenes": len(selected_scenes), **counts,
            "pair_accuracy": (counts["pairs_correct"] / counts["pairs_decidable"]
                              if counts["pairs_decidable"] else None),
            "top_selection_precision": (counts["correct_top_selection"] / counts["selected"]
                                        if counts["selected"] else None),
            "selection_coverage": counts["selected"] / len(selected_scenes),
            "physical_grasp_success_evaluated": False}
