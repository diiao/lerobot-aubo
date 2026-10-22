"""Export reviewed stationary piles as RGB + instance PNG + class metadata."""

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = [{"id": 0, "name": "top_strip"}, {"id": 1, "name": "covered_strip"}]
SCHEMA = "bamboo_visible_instance_dataset"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def child(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"path escapes dataset: {name}")
    return path


def read_mask(path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != shape or not set(np.unique(mask)) <= {0, 255}:
        raise ValueError(f"invalid binary mask: {path}")
    return mask > 0


def export_reviewed_piles(selection_path, output, *, test_only=False):
    """Publish a new, portable export; never change source reviews or split IDs."""
    selection_path, output = Path(selection_path).resolve(), Path(output).resolve()
    spec = read_json(selection_path)
    if test_only and any(p["split"] != "test" for p in spec["placements"]):
        raise ValueError("test-only export requires every selected group to be test")
    root = (selection_path.parent / spec["review_root"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("export must be separate from review root")
    names = [x["placement_id"] for x in spec["placements"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate placement selection (possible split leakage)")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".strip-export-", dir=output.parent))
    try:
        (staging / "images").mkdir()
        (staging / "instances").mkdir()
        samples, exclusions = [], []
        for entry in spec["placements"]:
            name, split = entry["placement_id"], entry["split"]
            if Path(name).name != name or split not in ("train", "validation", "test"):
                raise ValueError("invalid placement name or split")
            directory = child(root, name)
            review = read_json(directory / "review.json")
            polygons = read_json(directory / "polygons.json")
            source = Path(review["source_sequence"])
            sequence = read_json(source)
            if not sequence.get("finished") or sequence.get("stop_reason") != "completed":
                raise ValueError(f"incomplete capture: {name}")
            if any(x["placement_id"] != name for x in (review, polygons, sequence)):
                raise ValueError(f"placement mismatch: {name}")
            if review["split"] != split or sequence["split"] != split:
                raise ValueError(f"split changed: {name}")
            if not review["placement_order_confirmed"] or not all(
                x["reviewed"] is True for x in polygons["objects"]
            ):
                raise ValueError(f"unreviewed or unconfirmed group: {name}")
            records = review["frames"]
            if [x["index"] for x in records] != list(range(len(records))) or len(records) < 2:
                raise ValueError(f"missing stage: {name}")
            if [x["index"] for x in sequence["frames"]] != [x["index"] for x in records]:
                raise ValueError(f"source/review frame mismatch: {name}")
            for record in records:
                index = record["index"]
                # Copy the actual raw capture, never an overlay selected by a label.
                image_name = sequence["frames"][index]["image"]
                raw = child(source.parent, image_name)
                reviewed_image = child(directory, record["image"])
                if raw.read_bytes() != reviewed_image.read_bytes():
                    raise ValueError(f"review image differs from raw capture: {name}/{index}")
                image = cv2.imread(str(raw))
                if image is None or [image.shape[1], image.shape[0]] != spec["image_size"]:
                    raise ValueError(f"invalid image dimensions: {raw}")
                instance_map = np.zeros(image.shape[:2], dtype=np.uint16)
                segments, invisible = [], []
                if index == 0:
                    if record.get("role") != "empty_baseline":
                        raise ValueError("frame zero must explicitly be an empty baseline")
                else:
                    labels = read_json(child(directory, record["labels"]))
                    if (labels["placement_id"] != name or labels["split"] != split
                            or labels["image"] != record["image"]):
                        raise ValueError(f"label lineage mismatch: {name}/{index}")
                    if not labels["mask_review_complete"] or not labels["placement_order_confirmed"]:
                        raise ValueError(f"unconfirmed labels: {name}/{index}")
                    if not labels["full_frame_mask_eligible"]:
                        exclusions.append({"placement_id": name, "frame_index": index,
                                           "reason": "full_frame_mask_ineligible"})
                        continue
                    objects = labels["instances"]
                    ids = [x["id"] for x in objects]
                    tops = labels["top_ids_candidate"]
                    if len(ids) != len(set(ids)) or tops is None or not set(tops) <= set(ids):
                        raise ValueError("invalid instance or top IDs")
                    relations = labels["relations_from_order_and_overlap"]
                    order = {x["id"]: x["introduced_at_frame"] for x in objects}
                    for relation in relations:
                        if (relation["above"] not in order or relation["below"] not in order
                                or order[relation["above"]] <= order[relation["below"]]):
                            raise ValueError("invalid placement-order relation")
                    if set(tops) != set(ids) - {x["below"] for x in relations}:
                        raise ValueError("top IDs disagree with confirmed relations")
                    for obj in objects:
                        if not obj["polygon_reviewed"] or obj["truncated"] or not obj["visible_mask"]:
                            raise ValueError("invalid eligible instance")
                        visible = read_mask(child(directory, obj["visible_mask"]), instance_map.shape)
                        footprint = read_mask(child(directory, obj["footprint_in_image"]), instance_map.shape)
                        if np.any(visible & ~footprint) or np.any(visible & (instance_map != 0)):
                            raise ValueError("visible masks overlap or escape their footprints")
                        if not visible.any():
                            invisible.append(obj["id"])
                            continue
                        mask_id = len(segments) + 1
                        instance_map[visible] = mask_id
                        ys, xs = np.nonzero(visible)
                        segments.append({"mask_id": mask_id, "strip_id": obj["id"],
                                         "class_id": 0 if obj["id"] in tops else 1,
                                         "area": int(visible.sum()),
                                         "bbox_xywh": [int(xs.min()), int(ys.min()),
                                                       int(xs.max() - xs.min() + 1),
                                                       int(ys.max() - ys.min() + 1)]})
                sample_id = f"{name}_frame_{index:03d}"
                image_rel, mask_rel = f"images/{sample_id}.png", f"instances/{sample_id}.png"
                shutil.copyfile(raw, staging / image_rel)
                if not cv2.imwrite(str(staging / mask_rel), instance_map):
                    raise OSError("cannot write instance PNG")
                samples.append({"id": sample_id, "placement_id": name, "split": split,
                                "frame_index": index, "image": image_rel, "instance_map": mask_rel,
                                "width": image.shape[1], "height": image.shape[0],
                                "empty_baseline": index == 0, "segments": segments,
                                "invisible_strip_ids": invisible})
        manifest = {"schema": SCHEMA, "classes": CLASSES, "background_mask_id": 0,
                    "strip_dimensions_mm": spec["strip_dimensions_mm"],
                    "source_selection": spec, "samples": samples, "excluded_frames": exclusions,
                    "annotation_scope": "visible instances and order-derived top/covered classes",
                    "graspability_labeled": False, "training_started": False}
        write_json(staging / "dataset.json", manifest)
        report = audit_dataset(staging, test_only=test_only)
        write_json(staging / "audit.json", report)
        staging.rename(output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def audit_dataset(root, *, test_only=False):
    """Validate portable files, explicit empty targets and group split isolation."""
    root = Path(root)
    data = read_json(root / "dataset.json")
    if data["schema"] != SCHEMA or data["classes"] != CLASSES:
        raise ValueError("unsupported dataset or category mapping")
    seen, groups, counts = set(), {}, {}
    for sample in data["samples"]:
        name, split = sample["placement_id"], sample["split"]
        if sample["id"] in seen or (name in groups and groups[name] != split):
            raise ValueError("duplicate sample or placement crossing splits")
        if split not in ("train", "validation", "test"):
            raise ValueError("invalid split")
        seen.add(sample["id"])
        groups[name] = split
        image = cv2.imread(str(child(root, sample["image"])))
        mask = cv2.imread(str(child(root, sample["instance_map"])), cv2.IMREAD_UNCHANGED)
        shape = (sample["height"], sample["width"])
        if image is None or image.shape != (*shape, 3) or mask is None or mask.shape != shape:
            raise ValueError("image/instance dimensions mismatch")
        ids = [s["mask_id"] for s in sample["segments"]]
        if len(ids) != len(set(ids)) or any(i <= 0 for i in ids) or set(np.unique(mask)) - {0} != set(ids):
            raise ValueError("instance map does not match segments")
        if sample["empty_baseline"] and (ids or sample["frame_index"] != 0):
            raise ValueError("invalid empty baseline")
        count = counts.setdefault(split, {"images": 0, "empty_images": 0, "instances": 0,
                                          "top_strip": 0, "covered_strip": 0})
        count["images"] += 1
        count["empty_images"] += int(not ids)
        for segment in sample["segments"]:
            if segment["class_id"] not in (0, 1):
                raise ValueError("unknown foreground class")
            ys, xs = np.nonzero(mask == segment["mask_id"])
            bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            if len(xs) != segment["area"] or bbox != segment["bbox_xywh"]:
                raise ValueError("area/bbox does not match mask")
            count["instances"] += 1
            count[CLASSES[segment["class_id"]]["name"]] += 1
    for split, number in Counter(groups.values()).items():
        counts[split]["groups"] = number
    if test_only and set(counts) != {"test"}:
        raise ValueError("test-only dataset must contain test samples exclusively")
    if not test_only and (not counts.get("train") or not counts.get("validation")):
        raise ValueError("both train and validation are required")
    return {"passed": True, "splits": counts, "excluded_frames": data["excluded_frames"],
            "strip_dimensions_mm": data["strip_dimensions_mm"], "group_split_overlap": []}
