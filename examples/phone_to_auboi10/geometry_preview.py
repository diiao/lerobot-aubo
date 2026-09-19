#!/usr/bin/env python
"""Offline bamboo geometry overlay. Does not connect to AUBO or send actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.bamboo_sorting.geometry_perception import (
    ALGORITHM_VERSION,
    detect_bamboo_geometry,
    render_geometry_overlay,
)


def _read_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _iter_video_rgb(path: Path, frame_ids: list[int]):
    import av

    wanted = set(frame_ids)
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                yield index, frame.to_ndarray(format="rgb24")
                wanted.remove(index)
                if not wanted:
                    break
    finally:
        container.close()
    if wanted:
        missing = ", ".join(str(item) for item in sorted(wanted))
        raise RuntimeError(f"video missing frame_ids: {missing}")


def _write_sample(output_dir: Path, stem: str, rgb: np.ndarray, frame_id: int) -> dict:
    detection = detect_bamboo_geometry(rgb, color_space="rgb")
    overlay_bgr = render_geometry_overlay(rgb, detection)
    record = {
        "frame_id": frame_id,
        "stem": stem,
        "image_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "algorithm_version": ALGORITHM_VERSION,
        "detection": detection.to_json_dict(),
        "note": "Pixel geometry only. Not robot-base meters. Not a motion command.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay_bgr)
    (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline bamboo geometry overlay; no robot I/O.")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frame-ids", default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/aubo_geometry_preview"))
    args = parser.parse_args()
    if not args.image and args.video is None:
        raise SystemExit("provide --image and/or --video")
    records = []
    for image_path in args.image:
        rgb = _read_rgb_image(image_path)
        records.append(_write_sample(args.output_dir, image_path.stem, rgb, frame_id=0))
    if args.video is not None:
        frame_ids = [int(item) for item in str(args.frame_ids).split(",") if item.strip()]
        for frame_id, rgb in _iter_video_rgb(args.video, frame_ids):
            records.append(_write_sample(args.output_dir, f"{args.video.stem}_f{frame_id}", rgb, frame_id))
    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "n": len(records),
        "n_valid": sum(1 for item in records if item["detection"]["valid"]),
        "n_invalid": sum(1 for item in records if not item["detection"]["valid"]),
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n": summary["n"], "n_valid": summary["n_valid"], "n_invalid": summary["n_invalid"]}))


if __name__ == "__main__":
    main()
