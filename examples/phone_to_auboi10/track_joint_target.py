"""Diagnose target-mask propagation on saved video; no neural model or hardware."""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from lerobot.bamboo_sorting.joint_target_tracking import TargetMaskFlow, mask_overlap


def track_video(annotations_path):
    path = Path(annotations_path).resolve()
    spec = json.loads(path.read_text())
    seeds = {row["video_frame_index"]: row for row in spec["frames"]}
    if len(seeds) != len(spec["frames"]) or 0 not in seeds:
        raise ValueError("unique keyframes including frame zero required")
    for index, row in seeds.items():
        if type(index) is not int or index < 0 or row.get("reviewed") is not True:
            raise ValueError("human-reviewed keyframes required")
        if row["target_id"] != spec["target_id"] or row["status"] != "visible":
            raise ValueError("keyframes must show the same selected target")
    run = (path.parent / spec["source_run"]).resolve()
    outputs = [path.parent / name for name in ("tracking.mp4", "tracking_frames.jsonl", "tracking.json")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("tracking outputs already exist; inspect before replacing")
    cap = cv2.VideoCapture(str(run / "videos/global_rgb.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps != 25 or max(seeds) >= count:
        cap.release()
        raise ValueError("expected a complete 25 fps source video")
    for index, row in seeds.items():
        if row["frame_id"] != f"{run.name}/global_rgb/{index}":
            cap.release()
            raise ValueError("keyframe source identity mismatch")
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "640x480", "-r", str(fps), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(outputs[0]),
    ], stdin=subprocess.PIPE)
    tracker = TargetMaskFlow()
    checkpoints, losses = [], []
    index, active_seed, draft_count = 0, None, 0
    try:
        with outputs[1].open("x") as journal:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                was_active = tracker.mask is not None
                mask, detail = tracker.step(rgb)
                if was_active and mask is None:
                    losses.append({"frame": index, "time_s": index / fps, "seed_frame": active_seed, **detail})
                if index in seeds:
                    reference = np.zeros((480, 640), np.uint8)
                    for polygon in seeds[index]["visible_polygons"]:
                        cv2.fillPoly(reference, [np.asarray(polygon, np.int32)], 1)
                    reference = reference.astype(bool)
                    if index:
                        checkpoints.append({"frame": index, "time_s": index / fps,
                                            "seed_frame": active_seed, **mask_overlap(mask, reference)})
                    tracker.seed(rgb, reference)
                    mask, active_seed = reference, index
                    detail = {"status": "human_keyframe", "reviewed": True}
                elif mask is not None:
                    draft_count += 1
                record = {"frame": index, "video_time_s": index / fps, "target_id": spec["target_id"],
                          "seed_frame": active_seed, "mask_pixels": int(mask.sum()) if mask is not None else 0,
                          **detail}
                journal.write(json.dumps(record) + "\n")
                # Green = confirmed keyframe; orange = unreviewed flow draft.
                marked = rgb.copy()
                if mask is not None:
                    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
                    marked[mask & (eroded == 0)] = (0, 255, 0) if index in seeds else (255, 150, 0)
                cv2.rectangle(marked, (0, 0), (640, 28), (0, 0, 0), -1)
                cv2.putText(marked, f'{index/fps:.2f}s | {detail["status"]} | seed={active_seed}',
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                encoder.stdin.write(marked.tobytes())
                index += 1
                if index % 500 == 0:
                    print(f"processed {index}/{count}", flush=True)
    finally:
        cap.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code or index != count:
        raise RuntimeError(f"incomplete diagnostic: encoder={code}, frames={index}/{count}")
    report = {"source_run": spec["source_run"], "target_id": spec["target_id"], "frames": index,
              "fps": fps, "draft_frames": draft_count, "human_keyframes": len(seeds),
              "lost_frames": index - draft_count - len(seeds), "loss_events": losses,
              "checkpoints_before_human_reset": checkpoints, "method": "OpenCV DIS FAST backward mask warp",
              "thresholds": {"fb_error_px": 2.0, "gray_residual": 35, "retained_fraction": 0.65,
                             "area_vs_seed_min": 0.35, "area_vs_seed_max": 2.0},
              "limitations": "development diagnostic; human reseeding; cannot establish identity or recover newly visible regions",
              "training_ready": False, "neural_inference_run": False}
    outputs[2].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(track_video(args.annotations), ensure_ascii=False, indent=2))
