"""Target prediction import, seed review, separation and review-page rendering.

These functions operate on saved files only; they never load a model or robot.
"""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..file_io import read_json as read, write_json as write, write_text_atomic as write_text
from ..paths import EXAMPLE_ROOT
from ..policy.joint_target import prepare_target_images
from .joint_target_separation import MultiStripSeparation, visible_seed_masks


def import_predictions(folder, spec, path, *, single_strip_scenes=False):
    report = read(path)
    source_run = report.get("plan", {}).get("source_run")
    if not source_run or Path(source_run).resolve() != folder.parent.resolve():
        raise ValueError("prediction source run mismatch")
    frames = {f["sample_id"]: f for ep in spec["episodes"] for f in ep["frames"]}
    predictions = {s["id"]: s for s in report["samples"]}
    if not report.get("complete") or len(predictions) != len(report["samples"]) or set(predictions) != set(frames):
        raise ValueError("complete exact-frame predictions required")
    pending = []
    for name, row in frames.items():
        if row["reviewed"] or (folder / row["mask_path"]).exists():
            raise ValueError("existing mask/review must not be overwritten by predictions")
        prediction = predictions[name]
        if any(prediction[k] != row[k] for k in ("frame_index", "frame_id", "image")):
            raise ValueError("prediction frame identity mismatch")
        instance_path = (path.parent / prediction["instance_map"]).resolve()
        if not instance_path.is_relative_to(path.parent.resolve()):
            raise ValueError("instance path outside predictions")
        with Image.open(instance_path) as im:
            instance = np.array(im)
        if instance.shape != (480, 640) or not np.issubdtype(instance.dtype, np.integer):
            raise ValueError("invalid instance PNG")
        segments = prediction["segments"]
        ids = [s["mask_id"] for s in segments]
        if len(ids) != len(set(ids)) or any(type(i) is not int or not 0 < i <= 65535 for i in ids) or not set(np.unique(instance)) <= {0, *ids}:
            raise ValueError("invalid instance-to-segment mapping")
        for s in segments:
            if s["class_id"] not in (0, 1) or int((instance == s["mask_id"]).sum()) != s["pixels"]:
                raise ValueError("segment class/area mismatch")
        # One detection in a multi-strip scene may be the wrong object.
        # Only explicitly identified single-strip scenes can use this draft shortcut.
        mask = instance == ids[0] if single_strip_scenes and len(ids) == 1 else None
        if mask is not None and not mask.any():
            raise ValueError("empty selected instance")
        pending.append((row, mask, prediction))
    for row, mask, prediction in pending:
        instance = np.array(Image.open(path.parent / prediction["instance_map"]))
        (folder / "instances").mkdir(exist_ok=True)
        row["instance_map"] = f"instances/{row['sample_id']}.png"
        Image.fromarray(instance.astype(np.uint16)).save(folder / row["instance_map"])
        # RGB ID encoding lets a file:// review page select exact pixels with
        # a hidden canvas, including holes and disconnected visible regions.
        ids_rgb = np.zeros((480, 640, 3), np.uint8)
        ids_rgb[..., 0] = instance % 256
        ids_rgb[..., 1] = instance // 256
        row["instance_ids_image"] = f"instances/{row['sample_id']}_ids.png"
        Image.fromarray(ids_rgb).save(folder / row["instance_ids_image"])
        if any(row is ep["frames"][0] for ep in spec["episodes"]):
            row["instance_ids_base64"] = base64.b64encode((folder / row["instance_ids_image"]).read_bytes()).decode()
        preview = np.zeros((480, 640, 4), np.uint8)
        colors = [(255, 90, 90), (0, 190, 255), (255, 200, 0), (130, 90, 255), (0, 220, 120)]
        for i, segment in enumerate(prediction["segments"]):
            preview[instance == segment["mask_id"]] = [*colors[i % len(colors)], 130]
        row["candidates_image"] = f"instances/{row['sample_id']}_candidates.png"
        Image.fromarray(preview).save(folder / row["candidates_image"])
        row["prediction_segments"] = prediction["segments"]
        row["prediction_source"] = str(path.resolve())
        row["draft_selection"] = "sole_candidate_not_identity_proof" if mask is not None else "human_selection_required"
        if mask is None:
            row["status"] = "missing" if not prediction["segments"] else "ambiguous"
        else:
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / row["mask_path"])
            row["status"] = "visible"
            row["selected_instance_id"] = prediction["segments"][0]["mask_id"]
    write(folder / "annotations.json", spec)


def import_seeds(folder, spec, path):
    decisions = read(path)
    if (decisions.get("schema") != "aubo_joint_target_review_decisions"
            or decisions.get("source_dataset_root") != spec["source_dataset_root"]):
        raise ValueError("review source mismatch")
    by_episode = {str(ep["episode_index"]): ep for ep in spec["episodes"]}
    pending = []
    for key, selection in decisions.get("seeds", {}).items():
        if key not in by_episode:
            raise ValueError("unknown seed episode")
        ep = by_episode[key]
        first = ep["frames"][0]
        if (selection.get("frame_id") != first["frame_id"] or selection.get("human_reviewed") is not True
                or not selection.get("reviewed_at")):
            raise ValueError("explicit seed review bound to first approach frame required")
        if ep.get("visible_seeds") or any(f["reviewed"] or (folder / f["mask_path"]).exists() for f in ep["frames"]):
            raise ValueError("preserve existing seeds and target masks; edit individual masks through review")
        instances = np.array(Image.open(folder / first["instance_map"])) if "instance_map" in first else None
        masks = visible_seed_masks(selection.get("targets", []), instances)
        if ep["target_id"] not in masks:
            raise ValueError("selected target absent from visible seeds")
        if len(masks) < ep.get("available_target_count", 1):
            raise ValueError("visible seeds cannot be fewer than available upper targets")
        pending.append((ep, selection))
    if not pending:
        raise ValueError("no seed selections in export")
    for ep, selection in pending:
        ep["visible_seeds"] = selection
    write(folder / "annotations.json", spec)


def separate_predictions(folder, spec):
    """Generate unreviewed masks using the same separator as live preparation."""
    jobs = []
    for ep in spec["episodes"]:
        selection = ep.get("visible_seeds")
        if not selection:
            raise ValueError("every episode needs explicit visible seeds before separation")
        first = ep["frames"][0]
        for row in ep["frames"]:
            if "instance_map" not in row:
                raise ValueError("import all same-frame predictions first")
            if row["reviewed"] or (folder / row["mask_path"]).exists():
                raise ValueError("separation must not overwrite existing masks or review")
        instance = np.array(Image.open(folder / first["instance_map"]))
        jobs.append((ep, visible_seed_masks(selection["targets"], instance)))
    for ep, seeds in jobs:
        separator = MultiStripSeparation(seeds)
        for index, row in enumerate(ep["frames"]):
            instances = np.array(Image.open(folder / row["instance_map"]))
            masks, evidence = separator.step(index, instances, row["prediction_segments"])
            mask = masks[ep["target_id"]]
            row.update(separation=evidence, reviewed=False, status="visible" if mask.any() else "missing",
                       draft_selection="multi_instance_visible_seed_support")
            if mask.any():
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / row["mask_path"])
            if evidence["merged_instances"]:
                row["review_issue"] = "交叉／合并实例按可见种子拆分：请核对像素归属，修正或明确登记可接受误差"
        ep["reviewed"] = False
    spec["target_training_ready"] = False
    write(folder / "annotations.json", spec)


def import_decisions(folder, spec, path):
    """Apply explicit human exports; keep edited masks as drafts for a second look."""
    decisions = read(path)
    if (decisions.get("schema") != "aubo_joint_target_review_decisions"
            or decisions.get("source_dataset_root") != spec["source_dataset_root"]):
        raise ValueError("review source mismatch")
    frames = {f["frame_id"]: (ep, f) for ep in spec["episodes"] for f in ep["frames"]}
    pending = []
    for key, decision in decisions["frames"].items():
        if key not in frames:
            raise ValueError("unknown review frame")
        ep, row = frames[key]
        if (decision["frame_id"] != key or decision["episode_index"] != ep["episode_index"]
                or decision["frame_index"] != row["frame_index"]
                or decision.get("annotation_revision", 0) != row.get("annotation_revision", 0)):
            raise ValueError("review frame identity/revision mismatch; reopen current page")
        mask = None
        polygons = decision.get("polygons", [])
        limitation = decision.get("accepted_limitation")
        if limitation and (not isinstance(limitation, str) or not limitation.strip() or polygons
                           or decision.get("reviewed") is True or not decision.get("reviewed_at")):
            raise ValueError("accept a limitation explicitly on an unchanged draft, then review its contour")
        if polygons:
            mask = np.zeros((480, 640), np.uint8)
            for polygon in polygons:
                points = np.asarray(polygon)
                if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                        or not np.isfinite(points).all() or (points < 0).any()
                        or (points[:, 0] > 639).any() or (points[:, 1] > 479).any()):
                    raise ValueError("invalid visible polygon")
                # Fill independently so disjoint visible areas stay one object.
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
            if not mask.any():
                raise ValueError("empty polygon mask")
        if decision.get("reviewed") is True:
            if (mask is not None or decision.get("issue", row.get("review_issue")) or row["status"] != "visible"
                    or not (folder / row["mask_path"]).exists()
                    or decision.get("confirmation") != "explicit_user_range_review"
                    or not decision.get("reviewed_at")):
                raise ValueError("only explicit review of existing unchanged masks can confirm")
        pending.append((row, decision, mask))
    for row, decision, mask in pending:
        if mask is not None:
            target = folder / row["mask_path"]
            if target.exists():
                row.setdefault("previous_masks", []).append({
                    "annotation_revision": row.get("annotation_revision", 0),
                    "png_base64": base64.b64encode(target.read_bytes()).decode()})
            Image.fromarray(mask).save(target)
            row["status"] = "visible"
        row.setdefault("review_history", []).append(decision)
        if decision.get("accepted_limitation"):
            row.setdefault("accepted_annotation_limitations", []).append({
                "issue": decision["accepted_limitation"], "reviewed_at": decision["reviewed_at"],
                "source": "explicit_user_limitation_acceptance", "mask_modified": False})
        row["reviewed"] = decision.get("reviewed") is True
        row["annotation_revision"] = row.get("annotation_revision", 0) + 1
        row["review_issue"] = decision.get("issue", "")
    for ep in spec["episodes"]:
        ep["reviewed"] = all(f["reviewed"] for f in ep["frames"])
    # Passing review alone does not assert downstream full-trajectory validation.
    spec["target_training_ready"] = False
    write(folder / "annotations.json", spec)


def render(evidence, folder, spec):
    """Replace only our delimited section; preserve all earlier capture evidence."""
    summaries = []
    for ep in spec["episodes"]:
        previous_area = None
        for row in ep["frames"]:
            row["flags"] = []
            if row.get("review_issue"):
                row["flags"].append(row["review_issue"])
            for limitation in row.get("accepted_annotation_limitations", []):
                row["flags"].append("已接受误差（未修正）：" + limitation["issue"])
            if ep["close_frame"] is not None and ep["close_frame"] - row["frame_index"] <= 50:
                row["flags"].append("闭合前2秒：检查遮挡")
            path = folder / row["mask_path"]
            if not path.exists():
                row["flags"].append("缺少目标掩码")
                continue
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (480, 640) or not np.isin(mask, [0, 1, 255]).all() or not mask.any():
                raise ValueError(f"invalid binary mask: {path}")
            mask = mask != 0
            raw = np.array(Image.open(folder / row["image"]).convert("RGB"))
            result = prepare_target_images({"global_rgb": raw, "grasp_rgb": raw}, mask, row,
                                           frame_id=row["frame_id"], target_id=row["target_id"], allow_draft=True)
            boundary = np.any(result["images"]["target_global_rgb"] != raw, axis=-1)
            overlay = np.zeros((480, 640, 4), np.uint8)
            overlay[boundary] = [255, 0, 255, 255]
            overlay_path = folder / "outlines" / path.name
            Image.fromarray(overlay).save(overlay_path)
            row["outline"] = f"outlines/{path.name}"
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            row["visible_components"] = int(sum(s[cv2.CC_STAT_AREA] >= 8 for s in stats[1:]))
            row["mask_area"] = int(mask.sum())
            if row["visible_components"] > 1:
                row["flags"].append("多块可见区域：检查有无漏段")
            if previous_area and abs(row["mask_area"] / previous_area - 1) > .2:
                row["flags"].append("面积突变超过20%：仅为复核提示")
            previous_area = row["mask_area"]
        summaries.append({"episode_index": ep["episode_index"], "approach_frames": len(ep["frames"]),
                          "masks_present": sum((folder / f["mask_path"]).exists() for f in ep["frames"]),
                          "reviewed_frames": sum(f["reviewed"] for f in ep["frames"]),
                          "decoded_pts_match": all(abs(f["decoded_video_time_s"] - f["video_time_s"]) < 1e-4 for f in ep["frames"])})
    write(folder / "annotations.json", spec)
    report = {"schema": "aubo_joint_target_annotation_preparation", "episodes": summaries,
              "approach_frames": sum(s["approach_frames"] for s in summaries),
              "masks_present": sum(s["masks_present"] for s in summaries),
              "reviewed_frames": sum(s["reviewed_frames"] for s in summaries),
              "accepted_limitation_frames": sum(bool(f.get("accepted_annotation_limitations"))
                                                for e in spec["episodes"] for f in e["frames"]),
              "target_source_ready": spec.get("target_source_ready", False),
              "target_training_ready": False, "model_executed_by_preparation_tool": False,
              "original_dataset_modified": False}
    report["missing_masks"] = report["approach_frames"] - report["masks_present"]
    write(folder / "preparation_check.json", report)
    template = (EXAMPLE_ROOT / "joint_target_review.html").read_text()
    data = base64.b64encode(json.dumps(spec, ensure_ascii=False).encode()).decode()
    section = template.replace("__ANNOTATIONS_BASE64__", data)
    page = evidence / "review.html"
    text = page.read_text()
    start, end = "<!-- target-approach-review:start -->", "<!-- target-approach-review:end -->"
    if start in text:
        a, tail = text.split(start, 1)
        _, b = tail.split(end, 1)
        text = a + start + section + end + b
    else:
        position = text.index("<h1>")
        text = text[:position] + start + section + end + text[position:]
    write_text(page, text)
    print(json.dumps(report, ensure_ascii=False, indent=2))
