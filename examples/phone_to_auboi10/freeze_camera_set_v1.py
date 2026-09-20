#!/usr/bin/env python
"""Permanently freeze CameraSetV1 from recomputed Phase A2 evidence.

This command is offline. It never imports device SDKs, connects hardware, or
overwrites an existing ``CameraSetV1.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_SCHEMA_VERSION,
    FIXED_RGB_STREAMS,
    THREE_RGB_STREAMS,
    CameraSetDecision,
    CameraSetV1Record,
    ConcurrentRGBMetrics,
    RGBGateDecision,
    RGBStreamMetrics,
    decide_camera_set_v1,
    evaluate_concurrent_rgb_gate,
    evaluate_isolated_wrist_rgb,
    freeze_camera_set_v1,
)


def _read_json(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _stream_metrics(value: object) -> RGBStreamMetrics:
    if not isinstance(value, dict):
        raise ValueError("RGB stream metrics must be an object")
    return RGBStreamMetrics(**value)


def _concurrent_metrics(value: object) -> ConcurrentRGBMetrics:
    if not isinstance(value, dict):
        raise ValueError("Concurrent metrics must be an object")
    fields = dict(value)
    streams = fields.get("stream_metrics")
    if not isinstance(streams, dict):
        raise ValueError("Concurrent stream_metrics must be an object")
    fields["stream_metrics"] = {
        name: _stream_metrics(metrics) for name, metrics in streams.items()
    }
    return ConcurrentRGBMetrics(**fields)


def _validate_common_report(report: dict[str, object], expected_mode: str) -> None:
    if report.get("mode") != expected_mode:
        raise ValueError(f"Expected report mode {expected_mode!r}")
    if report.get("depth_requested") is not False:
        raise ValueError("Phase A2 evidence must explicitly record depth_requested=false")
    if report.get("camera_parameters_changed") is not False:
        raise ValueError("Phase A2 evidence must explicitly record no camera parameter changes")
    authorization = report.get("authorization_evidence_ref")
    if not isinstance(authorization, str) or not authorization:
        raise ValueError("Report lacks an authorization evidence reference")


def _profile_map(report: dict[str, object], streams: tuple[str, ...]) -> dict[str, dict[str, object]]:
    profiles = report.get("capture_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Concurrent report lacks capture_profiles")
    result: dict[str, dict[str, object]] = {}
    for stream in streams:
        profile = profiles.get(stream)
        if not isinstance(profile, dict):
            raise ValueError(f"Concurrent report lacks capture profile for {stream}")
        if "observed_width" not in profile or "observed_height" not in profile:
            raise ValueError(f"Capture profile for {stream} lacks observed dimensions")
        result[stream] = profile
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-report", type=Path, required=True)
    parser.add_argument("--concurrent-report", type=Path, required=True)
    parser.add_argument("--global-roi-evidence-ref", required=True)
    parser.add_argument("--grasp-roi-evidence-ref", required=True)
    parser.add_argument("--global-roi-accepted", action="store_true", required=True)
    parser.add_argument("--grasp-roi-accepted", action="store_true", required=True)
    parser.add_argument("--confirm-permanent-v1-freeze", action="store_true", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/aubo_i10/CameraSetV1.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    isolated_report, isolated_sha256 = _read_json(args.isolated_report)
    concurrent_report, concurrent_sha256 = _read_json(args.concurrent_report)
    _validate_common_report(isolated_report, "mech_eye_2d_isolated")
    _validate_common_report(concurrent_report, "rgb_aubo_state_concurrent")
    if concurrent_report.get("aubo_motion_or_io_interface_obtained") is not False:
        raise ValueError("Concurrent evidence does not prove the AUBO connection remained read-only")

    isolated_metrics = _stream_metrics(isolated_report.get("metrics"))
    concurrent_metrics = _concurrent_metrics(concurrent_report.get("metrics"))
    isolated_result = evaluate_isolated_wrist_rgb(isolated_metrics)
    fixed_result = evaluate_concurrent_rgb_gate(
        concurrent_metrics, required_rgb_streams=FIXED_RGB_STREAMS
    )
    candidate_streams = tuple(concurrent_report.get("candidate_streams", ()))
    three_result = None
    if candidate_streams == THREE_RGB_STREAMS:
        three_result = evaluate_concurrent_rgb_gate(
            concurrent_metrics, required_rgb_streams=THREE_RGB_STREAMS
        )
    elif candidate_streams != FIXED_RGB_STREAMS:
        raise ValueError("Concurrent report has an unsupported candidate stream set")

    decision = decide_camera_set_v1(
        isolated_wrist_result=isolated_result,
        fixed_concurrent_result=fixed_result,
        three_concurrent_result=three_result,
        global_roi_accepted=args.global_roi_accepted,
        grasp_roi_accepted=args.grasp_roi_accepted,
    )
    if decision is CameraSetDecision.BLOCKED:
        raise RuntimeError("CameraSetV1 remains blocked; C0 is not eligible")
    if isolated_result.decision is RGBGateDecision.PASS and candidate_streams == FIXED_RGB_STREAMS:
        raise RuntimeError(
            "The isolated wrist gate passed, so a three-camera concurrent report is required before freezing"
        )

    streams = THREE_RGB_STREAMS if decision is CameraSetDecision.THREE_RGB else FIXED_RGB_STREAMS
    profiles = _profile_map(concurrent_report, streams)
    roles = {
        "global_rgb": (
            "external stationary global view covering the accepted pile, motion, and collection ROI"
        ),
        "grasp_rgb": (
            "eye-in-hand Sonix RGB view rigidly mounted on the wrist and moving with the robot"
        ),
    }
    if "wrist_rgb" in streams:
        roles["wrist_rgb"] = "wrist-mounted Mech-Eye pure 2D view; no 3D computation"
    timestamp_methods = {
        stream: concurrent_metrics.stream_metrics[stream].timestamp_method for stream in streams
    }
    record = CameraSetV1Record(
        schema_version=CAMERA_SET_SCHEMA_VERSION,
        frozen_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        decision=decision,
        camera_streams=streams,
        physical_roles=roles,
        capture_profiles=profiles,
        timestamp_methods=timestamp_methods,
        evidence_sha256={
            "mech_eye_2d_isolated_report": isolated_sha256,
            "rgb_aubo_state_concurrent_report": concurrent_sha256,
        },
        global_roi_evidence_ref=args.global_roi_evidence_ref,
        grasp_roi_evidence_ref=args.grasp_roi_evidence_ref,
    )
    digest = freeze_camera_set_v1(args.output, record)
    print(
        json.dumps(
            {
                "camera_set": list(streams),
                "camera_set_v1_path": str(args.output),
                "camera_set_v1_sha256": digest,
                "c0_eligible": True,
                "warning": "C0 still requires a separate on-site teleoperation authorization",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
