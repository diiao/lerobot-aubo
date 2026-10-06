"""Associate saved instance predictions on the fixed clip; no model or hardware access."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from lerobot.bamboo_sorting.joint_target_association import TargetAssociation, TargetReconfirmation


def associate_clip(review_root, plan_name="association_plan.json"):
    root = Path(review_root).resolve()
    work = root / "association"
    if Path(plan_name).name != plan_name or not plan_name.endswith("_plan.json"):
        raise ValueError("plan must be a named JSON file inside association/")
    plan = json.loads((work / plan_name).read_text())
    if plan["parameters"] != TargetAssociation.PARAMETERS:
        raise ValueError("association settings differ from the frozen clip plan")
    recovery = plan.get("reconfirmation")
    if recovery is not None and recovery != TargetReconfirmation.PARAMETERS:
        raise ValueError("reconfirmation settings differ from the frozen clip plan")
    stem = plan_name.removesuffix("_plan.json")
    output = work / f"{stem}.json"
    video = work / f"{stem}.mp4"
    journal = work / f"{stem}_frames.jsonl"
    if any(p.exists() for p in (output, video, journal)):
        raise FileExistsError("association results already exist")
    sources = [root / "segmentation/result", work / "result"]
    reports = [json.loads((p / "predictions.json").read_text()) for p in sources]
    contract_keys = ("model_path", "best_epoch", "threshold", "mask_threshold", "overlap_mask_area_threshold", "image_size")
    if any(not r["complete"] or r["optimizer_steps"] != 0 for r in reports):
        raise ValueError("completed frozen predictions required")
    if any(reports[0]["plan"][k] != reports[1]["plan"][k] for k in contract_keys):
        raise ValueError("reused and new prediction contracts differ")
    records = {}
    for source, report in zip(sources, reports, strict=True):
        for row in report["samples"]:
            frame = row["video_frame_index"]
            if plan["start_frame"] <= frame <= plan["end_frame"]:
                if frame in records:
                    raise ValueError("duplicate predicted frame")
                records[frame] = (source, row)
    indices = list(range(plan["start_frame"], plan["end_frame"] + 1))
    if sorted(records) != indices:
        raise ValueError("missing clip predictions")
    annotations = json.loads((root / "annotations.json").read_text())
    run = (root / annotations["source_run"]).resolve()
    if any(r["plan"]["source_run"] != run.name for r in reports):
        raise ValueError("source video identity mismatch")
    cap = cv2.VideoCapture(str(run / "videos/global_rgb.mp4"))
    if cap.get(cv2.CAP_PROP_FPS) != 25:
        cap.release()
        raise ValueError("association clip requires the original 25 fps video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, plan["start_frame"])
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x480", "-r", "25", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ], stdin=subprocess.PIPE)
    tracker = (TargetReconfirmation(plan["target_id"]) if recovery else
               TargetAssociation(plan["target_id"], alignment=plan.get("alignment", "translation")))
    rows = []
    try:
        with journal.open("x") as stream:
            for frame in indices:
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"missing source RGB frame {frame}")
                source, prediction = records[frame]
                instance_map = cv2.imread(str(source / prediction["instance_map"]), cv2.IMREAD_UNCHANGED)
                if instance_map is None:
                    raise ValueError("missing prediction PNG")
                source_args = {"source_timestamp": prediction["source_timing"]["source_timestamp"]} if recovery else {}
                if frame == plan["start_frame"]:
                    selected, record = tracker.select(frame, instance_map, prediction["segments"], plan["seed_mask_id"], **source_args)
                else:
                    selected, record = tracker.step(frame, instance_map, prediction["segments"], **source_args)
                record.update(video_time_s=frame / 25, predicted_instances=len(prediction["segments"]),
                              prediction_source=str((source / prediction["instance_map"]).relative_to(root)),
                              source_timing=prediction["source_timing"],
                              mask_pixels=int(selected.sum()) if selected is not None else 0)
                rows.append(record)
                stream.write(json.dumps(record) + "\n")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                marked = rgb.copy()
                for segment in prediction["segments"]:
                    mask = instance_map == segment["mask_id"]
                    edge = mask & (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
                    marked[edge] = (0, 160, 255)
                if selected is not None:
                    edge = selected & (cv2.erode(selected.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
                    marked[edge] = (0, 255, 60)
                for image, title in [(rgb, f'{frame/25:.2f}s | original RGB'),
                                     (marked, f'{frame/25:.2f}s | {record["status"]} | {plan["target_id"]} | local={record["mask_id"]}')]:
                    cv2.rectangle(image, (0, 0), (640, 28), (0, 0, 0), -1)
                    cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                encoder.stdin.write(np.concatenate((rgb, marked), axis=1).tobytes())
    finally:
        cap.release()
        encoder.stdin.close()
        returncode = encoder.wait()
    if returncode or len(rows) != len(indices):
        raise RuntimeError("incomplete association review video")
    counts = Counter(row["status"] for row in rows)
    first_lost = next((row for row in rows if row["status"] == "lost"), None)
    report = {"complete": True, "plan": plan, "frames": len(rows), "counts": dict(counts),
              "first_lost": first_lost, "multiple_candidate_frames": sum(row["predicted_instances"] > 1 for row in rows),
              "first_unobserved": next((row for row in rows if row["status"] == "unobserved"), None),
              "reconfirmation_events": [row for row in rows if row["status"] == "reconfirmed"],
              "final_status": rows[-1]["status"],
              "prediction_instance_counts": dict(Counter(row["predicted_instances"] for row in rows)),
              "human_reviewed": False, "physical_identity_validated": False, "training_ready": False,
              "limitations": "single-strip development clip; matched is a geometric hypothesis, not verified identity or full visible contour"}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--plan", default="association_plan.json")
    args = parser.parse_args()
    print(json.dumps(associate_clip(args.review_root, args.plan), ensure_ascii=False, indent=2))
