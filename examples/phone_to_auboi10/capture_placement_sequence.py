"""逐根摆放拍摄与 RGB 候选标注；默认只显示计划，--record 仅打开全局相机。"""

import argparse
import math
import sys
from pathlib import Path

from lerobot.bamboo_sorting.joint_camera_config import (
    CAMERA_SET_PATHS, DEFAULT_CAPTURE_CAMERA_SET, load_joint_camera_configuration,
)
from lerobot.bamboo_sorting.placement_sequence import PlacementSequence, validate_roi, write_json


def control_properties():
    import cv2

    return {"auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_white_balance": cv2.CAP_PROP_AUTO_WB, "white_balance": cv2.CAP_PROP_WB_TEMPERATURE}


def read_controls(capture):
    return {name: float(capture.get(prop)) for name, prop in control_properties().items()}


def lock_controls(capture, before, *, exposure=None, white_balance=None):
    """V4L2/OpenCV controls are driver-dependent; retain set results and readback."""
    properties = control_properties()
    requested = {"auto_exposure": 0.25, "exposure": exposure if exposure is not None else before["exposure"],
                 "auto_white_balance": 0.,
                 "white_balance": white_balance if white_balance is not None else before["white_balance"]}
    applied, skipped = {}, {}
    for mode, value_name in (("auto_exposure", "exposure"), ("auto_white_balance", "white_balance")):
        # A -1 automatic-mode readback means this backend cannot expose the
        # control. Never pass that sentinel back as a white-balance temperature.
        mode_value = before[mode]
        if not math.isfinite(mode_value) or mode_value < 0:
            skipped[mode] = "unsupported_readback"
            skipped[value_name] = "manual_mode_unverified"
            continue
        applied[mode] = bool(capture.set(properties[mode], requested[mode]))
        mode_readback = float(capture.get(properties[mode]))
        manual_confirmed = (mode_readback in (.25, 1.) if mode == "auto_exposure" else mode_readback == 0.)
        if not manual_confirmed:
            skipped[value_name] = "manual_mode_unverified"
            continue
        original = before[value_name]
        if not math.isfinite(original) or original == -1 or (value_name == "white_balance" and original <= 0):
            skipped[value_name] = "unsupported_readback"
            continue
        applied[value_name] = bool(capture.set(properties[value_name], requested[value_name]))
    actual = read_controls(capture)
    # V4L2 reports manual exposure as either normalized .25 or enum value 1.
    exposure_ok = (applied.get("auto_exposure", False) and applied.get("exposure", False)
                   and actual["auto_exposure"] in (0.25, 1.)
                   and math.isclose(actual["exposure"], requested["exposure"], abs_tol=0.01))
    wb_ok = (applied.get("auto_white_balance", False) and applied.get("white_balance", False)
             and actual["auto_white_balance"] == 0 and actual["white_balance"] > 0
             and math.isclose(actual["white_balance"], requested["white_balance"], abs_tol=1.))
    return {"before": before, "requested": requested, "set_succeeded": applied, "skipped": skipped, "readback": actual,
            "lock_verified": exposure_ok and wb_ok,
            "warnings": [] if exposure_ok and wb_ok else ["camera_control_lock_unverified"]}


def restore_controls(capture, before, applied):
    properties = control_properties()
    # Restore manual values first, then the original automatic modes.
    return {name: bool(capture.set(properties[name], before[name]))
            for name in ("exposure", "white_balance", "auto_exposure", "auto_white_balance")
            if applied.get(name, False)}


def capture_live(sequence, profile, *, strips, exposure, white_balance):
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    camera = OpenCVCamera(OpenCVCameraConfig(
        index_or_path=profile["device"], width=profile["width"], height=profile["height"],
        fps=profile["fps"], fourcc=profile["fourcc"], warmup_s=3,
    ))
    before = None
    controls = {}
    reason = "error"
    try:
        camera.connect(warmup=True)
        # Do not mutate the VideoCapture handle while its reader is running.
        camera._stop_read_thread()
        before = read_controls(camera.videocapture)
        controls = lock_controls(camera.videocapture, before, exposure=exposure, white_balance=white_balance)
        sequence.manifest["camera_controls"] = controls
        sequence.save()
        camera._start_read_thread()
        # Drain frames from before the control changes without sleeping on input.
        for _ in range(10):
            camera.read()
        if not controls["lock_verified"]:
            print("警告：曝光/白平衡锁定未获驱动完整确认；候选标签必须复核，详情见 sequence.json。", flush=True)
        for index in range(strips + 1):
            prompt = "清空料堆区域，手退出画面" if index == 0 else f"放入第 {index} 根竹条，等静止后手退出画面"
            reply = input(f"{prompt}；回车拍照，q 结束：").strip().lower()
            if reply == "q":
                reason = "operator_stopped"
                break
            frame = camera.read()
            record = sequence.append(frame, source={"camera": "global_rgb", "controls": read_controls(camera.videocapture)})
            print(f"已保存 {sequence.output / record['image']}；复核图：{record['review_image']}；"
                  f"提示：{record.get('warnings', [])}", flush=True)
        else:
            reason = "completed"
    except (KeyboardInterrupt, EOFError):
        reason = "operator_stopped"
    except Exception as exc:
        sequence.manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if before is not None and camera.is_connected:
                camera._stop_read_thread()
                controls["restore_succeeded"] = restore_controls(camera.videocapture, before, controls.get("set_succeeded", {}))
                controls["restore_status"] = "attempted" if controls["restore_succeeded"] else "not_required_no_successful_writes"
                sequence.manifest["camera_controls"] = controls
                if not all(controls["restore_succeeded"].values()):
                    print("提示：部分相机控制项恢复未获驱动确认，请检查 sequence.json。", file=sys.stderr)
        finally:
            try:
                if camera.is_connected or camera.thread is not None:
                    camera.disconnect()
            finally:
                sequence.finish(reason)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-set", choices=tuple(CAMERA_SET_PATHS), default=DEFAULT_CAPTURE_CAMERA_SET)
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwrite an existing group")
    parser.add_argument("--placement-id", required=True, help="Physical grouping ID shared by every derived frame")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--strips", type=int, default=5, help="Number of additions after the empty baseline")
    parser.add_argument("--threshold", type=int, default=25, help="Maximum RGB channel difference threshold, 1..255")
    parser.add_argument("--min-area", type=int, default=80, help="Minimum changed component area in pixels")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Only label this region in original pixels; originals remain full-size")
    parser.add_argument("--min-chroma-change", type=float, default=0.,
                        help="Optional Lab a/b color-distance filter; 0 disables, 8 is a pilot setting")
    parser.add_argument("--exposure", type=float, help="Driver-native exposure value; default: freeze warmed-up reading")
    parser.add_argument("--white-balance", type=float, help="Temperature in kelvin; default: freeze warmed-up reading")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="Open only the global camera for interactive capture")
    mode.add_argument("--input-images", type=Path, nargs="+", help="Offline images in explicit order: empty, +1, +2, ...")
    args = parser.parse_args(argv)
    if args.strips < 1 or not 1 <= args.threshold <= 255 or args.min_area < 1:
        parser.error("strips/min-area must be positive and threshold must be 1..255")
    if not args.placement_id.strip():
        parser.error("placement-id must not be empty")
    if any(value is not None and not math.isfinite(value) for value in (args.exposure, args.white_balance)):
        parser.error("camera controls must be finite")
    if args.white_balance is not None and args.white_balance <= 0:
        parser.error("white-balance temperature must be positive")
    if not math.isfinite(args.min_chroma_change) or args.min_chroma_change < 0:
        parser.error("min-chroma-change must be finite and nonnegative")
    if args.input_images and len(args.input_images) < 2:
        parser.error("input-images needs an empty baseline and at least one addition")
    configuration = load_joint_camera_configuration(args.camera_set)
    # Store only the camera that will actually be opened, not the wrist profile.
    configuration["camera_mapping"] = {"global_rgb": configuration["camera_mapping"]["global_rgb"]}
    if not args.input_images:
        profile = configuration["camera_mapping"]["global_rgb"]
        try:
            validate_roi(args.roi, (profile["height"], profile["width"]))
        except ValueError as exc:
            parser.error(str(exc))
    plan = {"mode": "offline" if args.input_images else "camera", "camera_configuration": configuration,
            "output": str(args.output), "placement_id": args.placement_id, "split": args.split,
            "strips": len(args.input_images) - 1 if args.input_images else args.strips,
            "threshold": args.threshold, "min_area": args.min_area,
            "roi_xyxy": args.roi, "min_chroma_change": args.min_chroma_change,
            "robot_connected": False, "annotation_status": "needs_review"}
    if not args.record and not args.input_images:
        import json

        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.record and not sys.stdin.isatty():
        parser.error("--record requires an interactive terminal")
    if args.input_images:
        missing = [str(path) for path in args.input_images if not path.is_file()]
        if missing:
            parser.error(f"input images do not exist: {missing}")
        configuration = {"source": "offline_images", "camera_configuration_verified": False}
    sequence = PlacementSequence(args.output, placement_id=args.placement_id, split=args.split,
                                 configuration=configuration, threshold=args.threshold, min_area=args.min_area,
                                 roi=args.roi, min_chroma_change=args.min_chroma_change)
    write_json(args.output / "plan.json", plan)
    if args.input_images:
        import cv2

        try:
            for path in args.input_images:
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise ValueError(f"cannot decode {path}")
                sequence.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), source={"path": str(path.resolve())})
        except Exception as exc:
            sequence.manifest["error"] = f"{type(exc).__name__}: {exc}"
            sequence.finish("error")
            raise
        sequence.finish("completed")
    else:
        capture_live(sequence, configuration["camera_mapping"]["global_rgb"], strips=args.strips,
                     exposure=args.exposure, white_balance=args.white_balance)
    print(f"序列记录：{args.output / 'sequence.json'}；所有标签均待人工复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
