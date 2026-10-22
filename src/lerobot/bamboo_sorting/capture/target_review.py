"""Capture-side target review preparation: audits, manifests and frame extraction.

No model or hardware access. The command entry keeps the argparse surface.
"""

import html
import json

import av
import pyarrow.parquet as pq
from PIL import Image

from ..aubo_joint_contract import GRIPPER_INDEX
from ..file_io import read_json as read, write_json as write, write_text_atomic as write_text
from ..joint_stack_metadata import stack_metadata
from ..policy.joint_target_policy import is_transport


def prepare_capture_check(dataset, evidence):
    """Bootstrap the existing review entry from a NEW finalized capture.

    Fully decode both videos and match every saved PTS to Parquet metadata.
    This is a storage/command audit and keyframe index, not physical review.
    """
    from lerobot.bamboo_sorting.capture.aubo_joint_audit import audit_joint_dataset
    paths = [evidence / name for name in ("capture_check.json", "raw_audit.json", "review.html", "capture_frames")]
    if any(path.exists() for path in paths):
        raise FileExistsError("preserve existing capture checks and review; use their current entry")
    audit = audit_joint_dataset(dataset, evidence)
    session = read(evidence / "session.json")
    saved = [r for r in session["attempts"] if r["status"] == "saved_pending_finalize"]
    metadata = {r["episode_index"]: r for p in sorted((dataset / "meta/episodes").rglob("*.parquet"))
                for r in pq.read_table(p).to_pylist()}
    data = pq.read_table(sorted((dataset / "data").glob("chunk-*/*.parquet")),
                         columns=["episode_index", "frame_index", "timestamp"]).to_pylist()
    episodes, wanted = [], {}
    for capture, raw in zip(saved, audit["episodes"], strict=True):
        ep = capture["episode_index"]
        close_time, close_frame, approach = None, None, 0
        clocks = capture["sensor_and_suction_evidence"]["timestamp_journal"]["frames"]
        for row, clock in zip(capture["joint_frames"], clocks, strict=True):
            source_time = clock["timestamps"]["global_rgb"]["host_receive_monotonic_s"]
            approach += not is_transport(source_time, close_time)
            if row["action"][GRIPPER_INDEX] == 100 and close_time is None:
                close_time = row["joint_command_trace"]["accepted_monotonic_s"]
                close_frame = row["frame_index"]
        samples = sorted({0, raw["frames"]-1, close_frame if close_frame is not None else raw["frames"]//2})
        episodes.append({**raw, "close_frame": close_frame, "approach_frames": approach,
                         "transport_frames": raw["frames"]-approach, "sampled_frame_indices": samples})
        meta = metadata[ep]
        for name in ("global_rgb", "grasp_rgb"):
            key = "videos/observation.images." + name
            video = f'{key}/chunk-{meta[key + "/chunk_index"]:03d}/file-{meta[key + "/file_index"]:03d}.mp4'
            for row in data:
                if row["episode_index"] != ep:
                    continue
                time = meta[key + "/from_timestamp"] + row["timestamp"]
                wanted.setdefault(video, []).append((time, ep, row["frame_index"], name, row["frame_index"] in samples))
    frame_folder = evidence / "capture_frames"
    frame_folder.mkdir()
    videos = []
    for video, expected in wanted.items():
        expected.sort()
        count = 0
        with av.open(str(dataset / video)) as container:
            for frame in container.decode(video=0):
                if count >= len(expected):
                    raise ValueError("video contains frames outside finalized metadata")
                timestamp, ep, index, name, selected = expected[count]
                if abs(float(frame.pts * frame.time_base) - timestamp) > 1e-4:
                    raise ValueError("video PTS differs from saved frame timestamp")
                if (frame.height, frame.width, 3) != tuple(audit["image_shapes"][name]):
                    raise ValueError("decoded video resolution differs from contract")
                if selected:
                    Image.fromarray(frame.to_ndarray(format="rgb24")).save(frame_folder / f'{ep:04d}_{index:06d}_{name}.png')
                count += 1
        if count != len(expected):
            raise ValueError("incomplete video decode")
        videos.append({"path": video, "decoded_frames": count, "all_frame_timestamps_match_parquet": True})
    report = {"schema": "aubo_joint_target_capture_check", "dataset_root": str(dataset),
              "capture_complete": audit["capture_complete"], "dataset_finalized": True,
              "raw_contract_and_accepted_command_audit_passed": audit["audit_passed"],
              "gripper_quality_passed": audit["gripper_quality_passed"],
              "gripper_quality_issues": audit["gripper_quality_issues"],
              "episodes": episodes, "videos": videos, "frames": audit["frames"],
              "stack_summary": audit["stack_summary"], "physical_success_verified": False}
    cards = []
    for ep in episodes:
        cards.append(f'<section id="ep{ep["episode_index"]+1}"><h2>episode {ep["episode_index"]} / '
                     f'{html.escape(ep["scene_id"])} / {html.escape(ep.get("target_id", ""))}</h2>'
                     f'<p>{html.escape(json.dumps({k: ep.get(k) for k in ("pick_step", "available_target_count", "human_outcome", "target_outcome")}, ensure_ascii=False))}</p>')
        for index in ep["sampled_frame_indices"]:
            for name in ("global_rgb", "grasp_rgb"):
                cards.append(f'<img width="320" alt="{name} frame {index}" src="capture_frames/{ep["episode_index"]:04d}_{index:06d}_{name}.png">')
        cards.append('</section>')
    write(evidence / "raw_audit.json", audit)
    write(evidence / "capture_check.json", report)
    write_text(evidence / "review.html", '<!doctype html><meta charset="utf-8"><title>目标示教复核</title>'
               '<h1>采集存储检查与关键帧</h1><p>控制命令与视频完整不等于抓取成功；物理结果待现场和录像确认。</p>'
               + ''.join(cards))
    return report


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
                         **stack_metadata(fields),
                         **({"target_outcome": capture["target_outcome"]} if "target_outcome" in capture else {}),
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
