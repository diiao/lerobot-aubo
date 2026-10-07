"""Prepare saved approach frames and the existing review page; no model/hardware.

Run prepare once. Import separately authorized frozen predictions later, then
review them in the fixed HTML entry. Browser edits are exported as decisions;
they do not silently change the training sidecar.
"""

import argparse
import base64
import json
from pathlib import Path
import tempfile

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from lerobot.bamboo_sorting.aubo_joint_contract import GRIPPER_INDEX
from lerobot.bamboo_sorting.joint_target import prepare_target_images
from lerobot.bamboo_sorting.joint_target_policy import is_transport


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_text(path, value):
    # Readers opening the review page during refresh must see a complete file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def approach_manifest(dataset, evidence):
    """Use accepted command history, metadata offsets and original row identity."""
    metadata = {}
    for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            metadata[row["episode_index"]] = row
    check = {e["episode_index"]: e for e in read(evidence / "capture_check.json")["episodes"]}
    episodes, samples = [], []
    for capture in read(evidence / "session.json")["attempts"]:
        if capture["status"] != "saved_pending_finalize":
            continue
        ep = capture["episode_index"]
        meta = metadata[ep]
        fields = capture["metadata"]
        data_path = dataset / f'data/chunk-{meta["data/chunk_index"]:03d}/file-{meta["data/file_index"]:03d}.parquet'
        data = {r["frame_index"]: r for r in pq.read_table(
            data_path, columns=["episode_index", "frame_index", "timestamp", "index"]
        ).to_pylist() if r["episode_index"] == ep}
        video_key = "videos/observation.images.global_rgb"
        video = f'videos/observation.images.global_rgb/chunk-{meta[video_key + "/chunk_index"]:03d}/file-{meta[video_key + "/file_index"]:03d}.mp4'
        frames, close_time, close_frame = [], None, None
        clocks = capture["sensor_and_suction_evidence"]["timestamp_journal"]["frames"]
        for action, clock in zip(capture["joint_frames"], clocks, strict=True):
            index = action["frame_index"]
            if clock["frame_index"] != index or data[index]["index"] != meta["dataset_from_index"] + index:
                raise ValueError("frame/data index mismatch")
            source_time = clock["timestamps"]["global_rgb"]["host_receive_monotonic_s"]
            if not is_transport(source_time, close_time):
                sample_id = f"episode-{ep:04d}-frame-{index:06d}"
                row = {"frame_index": index, "schema": "aubo_joint_target_frame",
                       "frame_id": f"{ep}/global_rgb/{index}", "target_id": fields["target_id"],
                       "status": "pending", "reviewed": False,
                       "mask_path": f"masks/{sample_id}.png", "sample_id": sample_id,
                       "image": f"images/{sample_id}.png", "dataset_index": data[index]["index"],
                       "episode_timestamp_s": data[index]["timestamp"],
                       "video": video, "video_time_s": meta[video_key + "/from_timestamp"] + data[index]["timestamp"],
                       "global_source_monotonic_s": source_time}
                if row["video_time_s"] >= meta[video_key + "/to_timestamp"]:
                    raise ValueError("video time outside episode")
                frames.append(row)
                samples.append({"id": sample_id, "image": row["image"], "episode_index": ep,
                                "frame_index": index, "frame_id": row["frame_id"]})
            if action["action"][GRIPPER_INDEX] == 100 and close_time is None:
                trace = action["joint_command_trace"]
                if trace["return_code"] != 0:
                    raise ValueError("unaccepted close")
                close_time, close_frame = trace["accepted_monotonic_s"], index
        if len(frames) != check[ep]["approach_frames"] or close_frame != check[ep]["close_frame"]:
            raise ValueError("phase recomputation disagrees with capture check")
        episodes.append({"episode_index": ep, "scene_id": fields["scene_id"],
                         "target_id": fields["target_id"], "reviewed": False,
                         "target_description": fields.get("target_description", ""),
                         "placement_reference": fields.get("placement_reference", ""),
                         "close_frame": close_frame, "accepted_close_monotonic_s": close_time,
                         "total_frames": meta["length"], "frames": frames})
    spec = {"schema": "aubo_joint_target_annotations", "source_dataset_root": str(dataset),
            "target_training_ready": False, "episodes": episodes}
    inference = {"purpose": "target_video_development_diagnostic", "image_size": [640, 480],
                 "source_run": str(evidence), "samples": samples}
    return spec, inference


def extract_images(dataset, folder, spec):
    """Seek per approach interval, check decoded PTS; never use local index as MP4 index."""
    for ep in spec["episodes"]:
        rows = ep["frames"]
        with av.open(str(dataset / rows[0]["video"])) as container:
            stream = container.streams.video[0]
            container.seek(int(rows[0]["video_time_s"] / stream.time_base), stream=stream, backward=True)
            cursor = 0
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                row = rows[cursor]
                wanted = row["video_time_s"]
                if t < wanted - 1e-4:
                    continue
                if abs(t - wanted) > 1e-4:
                    raise ValueError(f"decoded timestamp mismatch: {row['frame_id']}: {t} != {wanted}")
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.shape != (480, 640, 3):
                    raise ValueError("wrong RGB resolution")
                Image.fromarray(rgb).save(folder / row["image"], compress_level=1)
                row["decoded_video_time_s"] = t
                cursor += 1
                if cursor == len(rows):
                    break
            if cursor != len(rows):
                raise ValueError("incomplete video interval")
        print(f'episode {ep["episode_index"]}: {cursor} approach RGB frames', flush=True)


def import_predictions(folder, spec, path, *, single_strip_scenes=False):
    report = read(path)
    if report.get("plan", {}).get("source_run") != str(folder.parent.resolve()):
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
        if len(ids) != len(set(ids)) or any(i <= 0 for i in ids) or not set(np.unique(instance)) <= {0, *ids}:
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
            if ep["close_frame"] - row["frame_index"] <= 50:
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
    template = Path(__file__).with_name("joint_target_review.html").read_text()
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


def verify_dataset(dataset, evidence, folder, spec):
    """Check every real sample through the existing training and runtime views.

    No policy is loaded. One decoded sample is cached only to avoid decoding it
    twice for the raw/transformed comparison; the real LeRobot loader is used.
    """
    import torch

    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_FIELDS
    from lerobot.bamboo_sorting.joint_target_policy import (
        TargetInputSession, prepare_target_source, read_visible_mask,
    )
    from lerobot.bamboo_sorting.joint_training import load_prepared_dataset

    torch.set_num_threads(2)
    capture_check = read(evidence / "capture_check.json")
    if not capture_check["raw_contract_and_accepted_command_audit_passed"]:
        raise ValueError("existing raw capture audit must have passed")
    record = {"root": str(dataset), "source_dataset_root": str(dataset),
              "evidence_root": str(evidence),
              "episodes": [e["episode_index"] for e in spec["episodes"]],
              "frames": sum(e["total_frames"] for e in spec["episodes"])}
    prepared = prepare_target_source(record, folder / "annotations.json")
    view = load_prepared_dataset(prepared)

    class CurrentSample:
        def __init__(self, source):
            self.source, self.index, self.value = source, None, None

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            if self.index != index:
                self.value, self.index = self.source[index], index
            return self.value

    view.dataset = CurrentSample(view.dataset)
    captures = {e["episode_index"]: e for e in read(evidence / "session.json")["attempts"]
                if e["status"] == "saved_pending_finalize"}
    data = pq.read_table(sorted((dataset / "data").glob("chunk-*/*.parquet")),
                         columns=["episode_index", "frame_index", "observation.state", "action"]).to_pylist()
    by_ep = {ep: sorted([r for r in data if r["episode_index"] == ep], key=lambda r: r["frame_index"])
             for ep in record["episodes"]}
    sessions = {e["episode_index"]: TargetInputSession(e["target_id"]) for e in spec["episodes"]}
    counts = {ep: {"episode_index": ep, "frames": 0, "approach": 0, "transport": 0,
                   "padded_chunks": 0} for ep in record["episodes"]}
    labels = {(e["episode_index"], f["frame_index"]): f for e in spec["episodes"] for f in e["frames"]}
    for index in range(len(view)):
        raw = view.dataset[index]
        ep, fi = int(raw["episode_index"]), int(raw["frame_index"])
        frame = view.frames[(ep, fi)]
        output = view[index]
        for key, value in raw.items():
            if key == "observation.images.global_rgb":
                if key in output:
                    raise ValueError("unprocessed global image leaked into target view")
                continue
            if isinstance(value, torch.Tensor):
                if not torch.equal(value, output[key]):
                    raise ValueError(f"modified raw field: {ep}/{fi} {key}")
            elif value != output[key]:
                raise ValueError(f"modified raw metadata: {ep}/{fi} {key}")
        expected_state = torch.tensor(by_ep[ep][fi]["observation.state"], dtype=raw["observation.state"].dtype)
        expected_indices = np.minimum(fi + np.arange(50), len(by_ep[ep]) - 1)
        expected_actions = torch.tensor([by_ep[ep][int(i)]["action"] for i in expected_indices], dtype=raw["action"].dtype)
        expected_pad = torch.from_numpy(fi + np.arange(50) >= len(by_ep[ep]))
        if (not torch.equal(raw["observation.state"], expected_state)
                or not torch.equal(raw["action"], expected_actions)
                or not torch.equal(raw["action_is_pad"], expected_pad)):
            raise ValueError(f"source state/action/chunk padding mismatch: {ep}/{fi}")
        images = {name: raw[f"observation.images.{name}"].mul(255).round().to(torch.uint8)
                  .permute(1, 2, 0).numpy() for name in ("global_rgb", "grasp_rgb")}
        annotation = frame["annotation"]
        mask = read_visible_mask(annotation["mask_path"]) if annotation else None
        if annotation is not None:
            source = np.array(Image.open(folder / labels[(ep, fi)]["image"]).convert("RGB"))
            if not np.array_equal(source, images["global_rgb"]):
                raise ValueError(f"annotation image identity mismatch: {ep}/{fi}")
        clock = captures[ep]["sensor_and_suction_evidence"]["timestamp_journal"]["frames"][fi]
        runtime = sessions[ep].prepare(
            {"observation.state": raw["observation.state"].numpy(), "task": raw["task"],
             **{f"observation.images.{k}": v for k, v in images.items()}},
            frame_id=frame["frame_id"],
            global_timestamp=clock["timestamps"]["global_rgb"]["host_receive_monotonic_s"],
            visible_mask=mask, annotation=annotation)
        phase = "transport" if frame["transport"] else "approach"
        if runtime["target_input"]["phase"] != phase:
            raise ValueError("training/runtime phase disagreement")
        for key in ("target_global_rgb", "grasp_rgb"):
            actual = output[f"observation.images.{key}"].mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            if not np.array_equal(actual, runtime[f"observation.images.{key}"]):
                raise ValueError(f"training/runtime image mismatch: {ep}/{fi} {key}")
        action = captures[ep]["joint_frames"][fi]
        sessions[ep].accept_command(dict(zip(JOINT_FIELDS, action["action"], strict=True)),
                                   accepted_monotonic_s=action["joint_command_trace"]["accepted_monotonic_s"])
        counts[ep]["frames"] += 1
        counts[ep][phase] += 1
        counts[ep]["padded_chunks"] += int(expected_pad.any())
        if index % 100 == 0:
            print(f"checked {index + 1}/{len(view)}: episode {ep}, frame {fi}", flush=True)
    for e in capture_check["episodes"]:
        if counts[e["episode_index"]]["approach"] != e["approach_frames"]:
            raise ValueError("phase count differs from capture evidence")
    result = {"schema": "aubo_joint_target_dataset_acceptance", "passed": True,
              "episodes": list(counts.values()), "frames": len(view),
              "source_dataset_root": str(dataset), "annotation_path": str(folder / "annotations.json"),
              "prepare_target_source_passed": True, "real_lerobot_dataset_backend": "pyav",
              "all_approach_images_match_annotation_source": True,
              "all_states_actions_and_50_step_padding_unchanged": True,
              "all_training_runtime_images_and_phases_equal": True,
              "user_review": spec.get("human_review"), "pixel_exact_ground_truth_claimed": False,
              "training_started": False, "model_loaded": False, "hardware_access": False,
              "independent_validation_available": False}
    write(folder / "dataset_check.json", result)
    spec["target_source_ready"] = True
    spec["target_training_ready"] = False
    spec["training_readiness_note"] = "source interface accepted; independent validation source and training authorization still required"
    write(folder / "annotations.json", spec)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/joint_target_pilot"))
    parser.add_argument("--evidence-root", type=Path, default=Path("artifacts/joint_target_pilot"))
    parser.add_argument("--stage", choices=("prepare", "render", "import-predictions", "import-review", "verify"), default="prepare")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--single-strip-scenes", action="store_true",
                        help="Only for known single-strip scenes: use a sole detection as an unreviewed draft; never use for A/B pairs")
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    dataset, evidence = args.dataset_root.resolve(), args.evidence_root.resolve()
    folder = evidence / "target_annotations"
    if args.stage == "prepare":
        if folder.exists():
            raise FileExistsError(f"preserve existing annotation work: {folder}")
        spec, inference = approach_manifest(dataset, evidence)
        folder.mkdir()
        for name in ("images", "masks", "outlines"):
            (folder / name).mkdir()
        extract_images(dataset, folder, spec)
        write(folder / "annotations.json", spec)
        write(folder / "inference_manifest.json", inference)
        write(folder / "inference_request.json", {
            "authorization": "not_requested_yet", "scope": "one frozen offline inference over these approach frames",
            "samples": len(inference["samples"]), "input": "inference_manifest.json",
            "remote_host_hint": "gpu", "remote_paths_verified": False,
            "remote_project_hint": "/home/rentao/program/lerobot-aubo-strip-segmentation-20261005",
            "remote_python_hint": "/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python",
            "model_hint": "outputs/short_strip_mask2former_baseline/best/",
            "remote_input_relative": f"artifacts/{evidence.name}/target_annotations",
            "remote_output_relative": f"artifacts/{evidence.name}/target_annotations/predictions",
            "local_output": str(folder / "predictions"),
            "entrypoint": "examples/phone_to_auboi10/predict_strip_images.py",
            "threshold": .5, "mask_threshold": .5, "overlap_mask_area_threshold": .8,
            "batch_size": 2, "resize": False, "image_size": [640, 480],
            "optimizer_steps": 0, "reviewed_on_import": False,
            "pre_execution": "verify remote Python, best checkpoint, input/output absence; copy only listed RGB inputs and standalone predictor after authorization"})
    else:
        spec = read(folder / "annotations.json")
        if spec["source_dataset_root"] != str(dataset):
            raise ValueError("source dataset mismatch")
        if args.stage == "import-predictions":
            if args.predictions is None:
                parser.error("--predictions required")
            import_predictions(folder, spec, args.predictions.resolve(),
                               single_strip_scenes=args.single_strip_scenes)
        if args.stage == "import-review":
            if args.decisions is None:
                parser.error("--decisions required")
            import_decisions(folder, spec, args.decisions.resolve())
        if args.stage == "verify":
            verify_dataset(dataset, evidence, folder, spec)
            return
    render(evidence, folder, spec)


if __name__ == "__main__":
    main()
