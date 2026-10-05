"""Offline polygon review of stationary, progressively assembled strip piles."""

import base64
import html
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .placement_sequence import write_image, write_json


def update_review_index(root):
    """List the current result for each group in the shared review directory."""
    root = Path(root)
    rows = []
    for directory in sorted(root.iterdir()):
        if not (directory / "review.json").is_file() or not (directory / "polygons.json").is_file():
            continue
        manifest = json.loads((directory / "review.json").read_text())
        objects = json.loads((directory / "polygons.json").read_text())["objects"]
        count = sum(item["reviewed"] is True for item in objects)
        complete = bool(objects) and count == len(objects)
        status = f"轮廓已确认 {count}/{len(objects)}" if complete else f"待复核 {count}/{len(objects)}"
        name = html.escape(directory.name, quote=True)
        rows.append((complete, directory.name,
                     f'<tr><td>{html.escape(manifest["placement_id"])}</td>'
                     f'<td>{html.escape(manifest["split"])}</td><td>{status}</td>'
                     f'<td><a href="{name}/review.html">打开复核页</a> · '
                     f'<a href="{name}/review.json">结果记录</a></td></tr>'))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>竹条轮廓复核</title>
<style>body{font:17px sans-serif;max-width:1000px;margin:32px auto;padding:0 16px}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}p{line-height:1.7}</style>
<h1>竹条轮廓复核</h1><p>每组只维护一份结果。待复核组排在前面，已完成组无需重复确认。</p>
<p>打开后逐根检查边界，必要时拖点修边，确认后下载JSON。下载文件经离线导入后更新本组及此索引，无需填写姓名。</p>
<p>轮廓确认不等于物理可夹、完整轮廓训练资格或已接入训练；截断及其他条件以组内结果记录为准。</p>
<table><tr><th>组号</th><th>集合</th><th>轮廓状态</th><th>入口</th></tr>'''
    page += "".join(row[2] for row in sorted(rows)) + "</table></html>\n"
    (root / "index.html").write_text(page)


def save_review(source, annotations, output=None):
    """Create or replace one group's derived result without writing source data."""
    source = Path(source).resolve()
    output = Path(output).resolve() if output is not None else source.parent / "review" / source.name
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("review output must be separate from the source sequence")
    if output.exists():
        record = output / "review.json"
        if not record.is_file():
            raise ValueError("existing output is not a review directory")
        previous = json.loads(record.read_text())
        if (previous.get("schema") != "bamboo_polygon_review"
                or previous.get("source_sequence") != str(source / "sequence.json")
                or previous.get("placement_id") != annotations.get("placement_id")):
            raise ValueError("existing review belongs to another source or placement")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".review-update-", dir=output.parent))
    old = staging / "previous"
    try:
        result = build_review(source, annotations, staging / "result")
        if output.exists():
            output.rename(old)
        try:
            (staging / "result").rename(output)
        except BaseException:
            if old.exists() and not output.exists():
                old.rename(output)
            raise
    finally:
        # Retain the only previous copy if restoring it itself fails.
        if not old.exists() or output.exists():
            shutil.rmtree(staging)
    if output.parent == source.parent / "review":
        update_review_index(output.parent)
    return result


def polygon_mask(points, shape):
    height, width = shape[:2]
    polygon = np.asarray(points, dtype=float)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all()):
        raise ValueError("polygon requires at least three finite x,y vertices")
    if ((polygon < 0).any() or (polygon[:, 0] > width - 1).any() or (polygon[:, 1] > height - 1).any()):
        raise ValueError("polygon vertices must lie inside the original image")
    contour = np.rint(polygon).astype(np.int32)
    if len(np.unique(contour, axis=0)) != len(contour) or cv2.contourArea(contour) < 1:
        raise ValueError("polygon has duplicate vertices or no area")
    def intersects(a, b, c, d):
        def cross(p, q, r):
            return int(q[0] - p[0]) * int(r[1] - p[1]) - int(q[1] - p[1]) * int(r[0] - p[0])
        if any(max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
            return False
        return cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0
    for i in range(len(contour)):
        for j in range(i + 2, len(contour)):
            if i == 0 and j == len(contour) - 1:
                continue
            if intersects(contour[i], contour[(i + 1) % len(contour)], contour[j], contour[(j + 1) % len(contour)]):
                raise ValueError("polygon edges cross; reorder or adjust vertices")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    return mask > 0


def build_review(source, annotations, output):
    """Preserve source files; polygon edits never imply operator approval."""
    source, output = Path(source).resolve(), Path(output).resolve()
    sequence = json.loads((source / "sequence.json").read_text())
    if annotations.get("placement_id") != sequence["placement_id"]:
        raise ValueError("placement_id does not match source sequence")
    records = sequence["frames"]
    if len(records) < 2 or [r["index"] for r in records] != list(range(len(records))):
        raise ValueError("source must contain an empty frame and consecutive additions")
    frames = []
    for record in records:
        path = (source / record["image"]).resolve()
        if path.parent != source:
            raise ValueError("source images must be directly inside the source directory")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"cannot decode {path}")
        frames.append(bgr)
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("all source images must have identical dimensions")
    if annotations.get("image_size") != [shape[1], shape[0]]:
        raise ValueError("annotation image_size does not match source")
    objects = annotations.get("objects", [])
    if [item["frame_index"] for item in objects] != list(range(1, len(frames))):
        raise ValueError("provide exactly one introduced strip polygon for each nonempty frame")
    if [item["id"] for item in objects] != [f"strip_{i:03d}" for i in range(1, len(frames))]:
        raise ValueError("strip IDs must follow their introduction order")
    masks = [polygon_mask(item["polygon"], shape) for item in objects]
    for item in objects:
        if type(item.get("truncated")) is not bool or type(item.get("reviewed")) is not bool:
            raise ValueError("truncated and reviewed must be explicit booleans")
    # This is a statement from the operator, not inferred from image differences.
    confirmed = (annotations.get("placement_from_above_confirmed") is True
                 and annotations.get("older_strips_stationary_confirmed") is True
                 and bool(annotations.get("placement_confirmation_source", "").strip()))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "polygons.json", annotations)
    manifest = {"schema": "bamboo_polygon_review", "source_sequence": str(source / "sequence.json"),
                "placement_id": sequence["placement_id"], "split": sequence["split"],
                "annotation_source": annotations.get("annotation_source", "unspecified"),
                "placement_order_confirmed": confirmed, "training_ready": False, "frames": []}
    colors = [(255, 170, 0), (0, 210, 0), (0, 150, 255)]  # BGR display colors only.
    comparisons = []
    for index, (record, bgr) in enumerate(zip(records, frames, strict=True)):
        image_name = f"frame_{index:03d}.png"
        shutil.copyfile(source / record["image"], output / image_name)
        if index == 0:
            manifest["frames"].append({"index": 0, "image": image_name, "role": "empty_baseline"})
            continue
        introduced = objects[index - 1]
        introduced_mask = masks[index - 1]
        write_image(output / f"frame_{index:03d}_polygon.png", introduced_mask.astype(np.uint8) * 255)
        instances, relations, covered = [], [], set()
        overlay = bgr.copy()
        for j, item in enumerate(objects[:index]):
            visible = masks[j].copy()
            if confirmed:
                for k in range(j + 1, index):
                    overlap = int(np.count_nonzero(masks[j] & masks[k]))
                    if overlap:
                        relations.append({"above": objects[k]["id"], "below": item["id"],
                                          "projection_overlap_pixels": overlap})
                        covered.add(item["id"])
                    visible &= ~masks[k]
            border = any((masks[j][0].any(), masks[j][-1].any(), masks[j][:, 0].any(), masks[j][:, -1].any()))
            truncated = item["truncated"] or border
            visible_path = f"frame_{index:03d}_{item['id']}_visible.png" if confirmed or j == index - 1 else None
            if visible_path:
                write_image(output / visible_path, visible.astype(np.uint8) * 255)
                contour, _ = cv2.findContours(visible.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contour, -1, colors[j % len(colors)], 1)
            instances.append({"id": item["id"], "introduced_at_frame": item["frame_index"],
                              "footprint_in_image": f"frame_{item['frame_index']:03d}_polygon.png",
                              "visible_mask": visible_path, "truncated": bool(truncated),
                              "polygon_reviewed": item["reviewed"], "graspable": None})
        reviewed = all(item["reviewed"] for item in objects[:index])
        clipped = any(item["truncated"] for item in instances)
        labels = {"placement_id": sequence["placement_id"], "split": sequence["split"], "image": image_name,
                  "annotation_status": "polygons_reviewed" if reviewed else "polygon_draft",
                  "placement_order_confirmed": confirmed, "training_ready": False,
                  "mask_review_complete": reviewed,
                  "full_frame_mask_eligible": reviewed and confirmed and not clipped,
                  "instances": instances, "relations_from_order_and_overlap": relations if confirmed else None,
                  "top_ids_candidate": [item["id"] for item in instances if item["id"] not in covered] if confirmed else None,
                  "warnings": (["truncated_object"] if clipped else []) + ([] if confirmed else ["placement_unconfirmed"]),
                  "note": "Draft polygons are not ground truth. Visibility propagation assumes unchanged projection; graspability remains unreviewed."}
        label_name = f"frame_{index:03d}_labels.json"
        write_json(output / label_name, labels)
        write_image(output / f"frame_{index:03d}_instances.png", overlay)
        current_overlay = bgr.copy()
        cv2.polylines(current_overlay, [np.rint(introduced["polygon"]).astype(np.int32)], True, (0, 255, 0), 1)
        original_overlay = bgr.copy()
        candidate = cv2.imread(str(source / record.get("candidate_mask", "missing.png")), cv2.IMREAD_GRAYSCALE)
        if candidate is not None:
            contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_overlay, contours, -1, (0, 0, 255), 1)
        for view, title in ((original_overlay, f"Step {index}: difference candidate"),
                            (current_overlay, f"Step {index}: polygon DRAFT")):
            cv2.rectangle(view, (0, 0), (shape[1] - 1, 30), (20, 20, 20), -1)
            cv2.putText(view, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        comparison = np.hstack([original_overlay, current_overlay])
        write_image(output / f"frame_{index:03d}_comparison.png", comparison)
        comparisons.append(comparison)
        manifest["frames"].append({"index": index, "image": image_name, "labels": label_name})
    write_image(output / "comparison.png", np.vstack(comparisons))
    write_json(output / "review.json", manifest)
    template = Path(__file__).with_name("placement_review.html").read_text()
    payload = {"annotations": annotations, "images": [base64.b64encode((output / f"frame_{i:03d}.png").read_bytes()).decode()
                                                    for i in range(1, len(frames))]}
    # Escape script end tags in user-entered notes without changing JSON values.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__REVIEW_DATA__", data))
    return manifest
