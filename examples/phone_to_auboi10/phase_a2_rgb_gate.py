#!/usr/bin/env python
"""Authorized Phase A2 Mech-Eye 2D and concurrent RGB/state gate runner.

The runner never requests 3D data, changes camera parameters, obtains AUBO
motion/IO interfaces, sends actions, or stores image pixels. It writes only
timing, identity hashes, error records, and frozen-threshold reports.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.bamboo_sorting.rgb_gate import (
    FIXED_RGB_STREAMS,
    RGB_GATE_SCHEMA_VERSION,
    THREE_RGB_STREAMS,
    ConcurrentRGBMetrics,
    RGBStreamMetrics,
    evaluate_concurrent_rgb_gate,
    evaluate_isolated_wrist_rgb,
    pair_key,
)
from lerobot.cameras.mech_mind import MechMindCamera, MechMindCameraConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

RUNNER_SCHEMA_VERSION = "PhaseA2RGBGateRunnerV1"
MIN_DURATION_S = 60.0


@dataclass(frozen=True)
class FrameSample:
    stream_name: str
    host_receive_monotonic_s: float
    identity: str
    content_sha256: str
    height: int
    width: int
    capture_call_elapsed_ms: float | None


@dataclass(frozen=True)
class StateSample:
    host_receive_monotonic_s: float
    joints_rad: tuple[float, ...]
    tcp_pose_m_rad: tuple[float, ...]


class SampleBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[Any] = []
        self._errors: list[dict[str, object]] = []

    def append(self, sample: Any) -> None:
        with self._lock:
            self._samples.append(sample)

    def error(self, operation: str, exc: BaseException) -> None:
        with self._lock:
            self._errors.append(
                {
                    "host_monotonic_s": time.monotonic(),
                    "operation": operation,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    def latest(self) -> Any | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def snapshot(self) -> tuple[list[Any], list[dict[str, object]]]:
        with self._lock:
            return list(self._samples), list(self._errors)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id(mode: str) -> str:
    return f"{mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _reserve_run_directory(root: Path, run_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / run_id
    path.mkdir(exist_ok=False)
    return path


def _write_json_new(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl_new(path: Path, rows: list[dict[str, object]]) -> str:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in rows
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _percentile_95(values: list[float]) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def _image_sha256(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def _timestamps_monotonic(samples: list[Any]) -> bool:
    return all(
        samples[index].host_receive_monotonic_s > samples[index - 1].host_receive_monotonic_s
        for index in range(1, len(samples))
    )


def _isolated_metrics(
    samples: list[FrameSample], errors: list[dict[str, object]], duration_s: float
) -> RGBStreamMetrics:
    identities = [sample.identity for sample in samples]
    unique_count = len(set(identities))
    duplicate_count = len(identities) - unique_count
    possible_frames = len(samples) + len(errors)
    error_drop_rate = len(errors) / possible_frames if possible_frames else 1.0

    numeric_ids: list[int] = []
    for identity in identities:
        if not identity.startswith("vendor_frame_id:"):
            numeric_ids = []
            break
        numeric_ids.append(int(identity.split(":", maxsplit=1)[1]))
    id_gap_drop_rate = 0.0
    if len(numeric_ids) >= 2 and numeric_ids[-1] >= numeric_ids[0]:
        expected_by_id = numeric_ids[-1] - numeric_ids[0] + 1
        id_gap_drop_rate = max(0, expected_by_id - unique_count) / expected_by_id

    elapsed = [
        sample.capture_call_elapsed_ms
        for sample in samples
        if sample.capture_call_elapsed_ms is not None
    ]
    return RGBStreamMetrics(
        stream_name="wrist_rgb",
        duration_s=duration_s,
        sample_count=len(samples),
        unique_sample_count=unique_count,
        effective_unique_fps=unique_count / duration_s if duration_s > 0 else 0.0,
        drop_rate_fraction=max(error_drop_rate, id_gap_drop_rate),
        duplicate_rate_fraction=duplicate_count / len(samples) if samples else 1.0,
        frame_age_p95_ms=_percentile_95(elapsed),
        capture_error_count=len(errors),
        timestamp_method="host_receive_monotonic_after_capture_2d_return",
        timestamp_traceable=True,
        timestamp_monotonic=_timestamps_monotonic(samples),
        frame_identity_method="vendor_frame_2d_id",
    )


def _serialize_stream_metrics(metrics: RGBStreamMetrics) -> dict[str, object]:
    return asdict(metrics)


def run_isolated(args: argparse.Namespace) -> int:
    run_id = _run_id("mech-eye-2d-isolated")
    run_dir = _reserve_run_directory(args.output_root, run_id)
    intent = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "mode": "mech_eye_2d_isolated",
        "created_at_utc": _utc_now(),
        "authorization_evidence_ref": args.authorization_ref,
        "hardware_operations": ["connect_mech_eye", "capture_2d", "disconnect_mech_eye"],
        "forbidden_operations": [
            "capture_2d_and_3d",
            "camera_parameter_write",
            "image_pixel_storage",
            "aubo_connection",
            "robot_motion",
            "io_output",
        ],
        "requested_duration_s": args.duration_s,
        "mech_eye_ip": args.mech_eye_ip,
        "capture_timeout_ms": args.capture_timeout_ms,
    }
    intent_sha256 = _write_json_new(run_dir / "intent.json", intent)

    camera = MechMindCamera(
        MechMindCameraConfig(
            ip_address=args.mech_eye_ip,
            use_depth=False,
            connect_timeout_ms=args.connect_timeout_ms,
            capture_timeout_ms=args.capture_timeout_ms,
        )
    )
    samples: list[FrameSample] = []
    errors: list[dict[str, object]] = []
    started = time.monotonic()
    try:
        camera.connect(warmup=False)
        started = time.monotonic()
        deadline = started + args.duration_s
        while time.monotonic() < deadline:
            try:
                frame = camera.read_rgbd()
                samples.append(
                    FrameSample(
                        stream_name="wrist_rgb",
                        host_receive_monotonic_s=frame.host_receive_monotonic_s,
                        identity=f"vendor_frame_id:{frame.frame_2d_id}",
                        content_sha256=_image_sha256(frame.rgb),
                        height=int(frame.rgb.shape[0]),
                        width=int(frame.rgb.shape[1]),
                        capture_call_elapsed_ms=frame.capture_latency_ms,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "host_monotonic_s": time.monotonic(),
                        "operation": "capture_2d",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    finally:
        if camera.is_connected:
            camera.disconnect()
    ended = time.monotonic()
    measured_duration = ended - started
    metrics = _isolated_metrics(samples, errors, measured_duration)
    gate = evaluate_isolated_wrist_rgb(metrics)
    rows = [
        {"record_type": "frame", **asdict(sample)} for sample in samples
    ] + [{"record_type": "error", **error} for error in errors]
    samples_sha256 = _write_jsonl_new(run_dir / "samples.jsonl", rows)
    report = {
        "schema_version": RGB_GATE_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "mode": "mech_eye_2d_isolated",
        "authorization_evidence_ref": args.authorization_ref,
        "depth_requested": False,
        "camera_parameters_changed": False,
        "image_pixels_stored": False,
        "metrics": _serialize_stream_metrics(metrics),
        "gate": gate.to_dict(),
        "evidence_sha256": {"intent.json": intent_sha256, "samples.jsonl": samples_sha256},
    }
    report_sha256 = _write_json_new(run_dir / "report.json", report)
    print(json.dumps({"run_dir": str(run_dir), "report_sha256": report_sha256, **gate.to_dict()}, indent=2))
    return 0 if gate.decision.value == "pass" else 2


class ReadOnlyAuboStateSource:
    """AUBO RPC reader that never obtains motion or IO interfaces."""

    def __init__(self, *, ip: str, port: int, username: str, password: str, timeout_ms: int):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.timeout_ms = timeout_ms
        self._rpc: Any | None = None
        self._state: Any | None = None

    def connect(self) -> None:
        import pyaubo_sdk

        rpc = pyaubo_sdk.RpcClient()
        rpc.setRequestTimeout(self.timeout_ms)
        rpc.connect(self.ip, self.port)
        if not rpc.hasConnected():
            raise ConnectionError(f"Failed to connect to AUBO RPC at {self.ip}:{self.port}")
        try:
            rpc.login(self.username, self.password)
            if not rpc.hasLogined():
                raise ConnectionError("Connected to AUBO RPC, but login failed")
            names = rpc.getRobotNames()
            if len(names) != 1:
                raise RuntimeError(f"Expected exactly one AUBO robot, received {len(names)}")
            interface = rpc.getRobotInterface(names[0])
            self._rpc = rpc
            self._state = interface.getRobotState()
        except Exception:
            rpc.disconnect()
            raise

    def read(self) -> StateSample:
        if self._state is None:
            raise RuntimeError("AUBO read-only state source is not connected")
        joints = tuple(float(value) for value in self._state.getJointPositions())
        tcp = tuple(float(value) for value in self._state.getTcpPose())
        received = time.monotonic()
        if len(joints) != 6 or len(tcp) != 6 or not all(math.isfinite(value) for value in (*joints, *tcp)):
            raise RuntimeError("AUBO returned invalid joint or TCP state")
        return StateSample(received, joints, tcp)

    def disconnect(self) -> None:
        if self._rpc is not None:
            try:
                if self._rpc.hasLogined():
                    self._rpc.logout()
            finally:
                if self._rpc.hasConnected():
                    self._rpc.disconnect()
        self._rpc = None
        self._state = None


def _monitor_opencv(
    name: str,
    camera: OpenCVCamera,
    buffer: SampleBuffer,
    stop: threading.Event,
) -> None:
    previous_timestamp: float | None = None
    while not stop.is_set():
        try:
            with camera.frame_lock:
                timestamp = camera.latest_timestamp
                frame = None if camera.latest_frame is None else camera.latest_frame.copy()
            if timestamp is not None and frame is not None and timestamp != previous_timestamp:
                buffer.append(
                    FrameSample(
                        stream_name=name,
                        host_receive_monotonic_s=timestamp,
                        identity=f"host_receive_ns:{round(timestamp * 1_000_000_000)}",
                        content_sha256=_image_sha256(frame),
                        height=int(frame.shape[0]),
                        width=int(frame.shape[1]),
                        capture_call_elapsed_ms=None,
                    )
                )
                previous_timestamp = timestamp
            if camera.thread is None or not camera.thread.is_alive():
                raise RuntimeError(f"{name} OpenCV read thread stopped")
        except Exception as exc:
            buffer.error("monitor_opencv", exc)
            return
        stop.wait(0.001)


def _capture_wrist(
    camera: MechMindCamera,
    buffer: SampleBuffer,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            frame = camera.read_rgbd()
            buffer.append(
                FrameSample(
                    stream_name="wrist_rgb",
                    host_receive_monotonic_s=frame.host_receive_monotonic_s,
                    identity=f"vendor_frame_id:{frame.frame_2d_id}",
                    content_sha256=_image_sha256(frame.rgb),
                    height=int(frame.rgb.shape[0]),
                    width=int(frame.rgb.shape[1]),
                    capture_call_elapsed_ms=frame.capture_latency_ms,
                )
            )
        except Exception as exc:
            buffer.error("capture_2d", exc)
            return


def _read_aubo_state(
    source: ReadOnlyAuboStateSource,
    buffer: SampleBuffer,
    stop: threading.Event,
    state_hz: float,
) -> None:
    period = 1.0 / state_hz
    next_read = time.monotonic()
    while not stop.is_set():
        try:
            buffer.append(source.read())
        except Exception as exc:
            buffer.error("get_joint_positions_and_tcp_pose", exc)
            return
        next_read += period
        stop.wait(max(0.0, next_read - time.monotonic()))


def _wait_for_sources(buffers: dict[str, SampleBuffer], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(buffer.latest() is not None for buffer in buffers.values()):
            return
        time.sleep(0.01)
    missing = [name for name, buffer in buffers.items() if buffer.latest() is None]
    raise TimeoutError(f"Timed out waiting for initial samples: {missing}")


def _concurrent_stream_metrics(
    name: str,
    samples: list[FrameSample],
    errors: list[dict[str, object]],
    groups: list[dict[str, object]],
    duration_s: float,
) -> RGBStreamMetrics:
    identities = [sample.identity for sample in samples]
    group_records = [group["streams"].get(name) for group in groups]
    missing = sum(record is None for record in group_records)
    present = [record for record in group_records if record is not None]
    duplicate = sum(
        present[index]["identity"] == present[index - 1]["identity"]
        for index in range(1, len(present))
    )
    ages = [float(record["age_ms"]) for record in present]
    return RGBStreamMetrics(
        stream_name=name,
        duration_s=duration_s,
        sample_count=len(samples),
        unique_sample_count=len(set(identities)),
        effective_unique_fps=len(set(identities)) / duration_s if duration_s > 0 else 0.0,
        drop_rate_fraction=missing / len(groups) if groups else 1.0,
        duplicate_rate_fraction=duplicate / max(1, len(present) - 1),
        frame_age_p95_ms=_percentile_95(ages),
        capture_error_count=len(errors),
        timestamp_method=(
            "host_receive_monotonic_after_capture_2d_return"
            if name == "wrist_rgb"
            else "host_receive_monotonic_after_opencv_read"
        ),
        timestamp_traceable=True,
        timestamp_monotonic=_timestamps_monotonic(samples),
        frame_identity_method=("vendor_frame_2d_id" if name == "wrist_rgb" else "host_receive_timestamp"),
    )


def _pairwise_skews(groups: list[dict[str, object]], names: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    pairs = [
        (names[index], names[other])
        for index in range(len(names))
        for other in range(index + 1, len(names))
    ]
    pairs.extend((name, "robot_state") for name in names)
    for left, right in pairs:
        key = pair_key(left, right)
        values[key] = []
        for group in groups:
            left_record = group["state"] if left == "robot_state" else group["streams"].get(left)
            right_record = group["state"] if right == "robot_state" else group["streams"].get(right)
            if left_record is not None and right_record is not None:
                values[key].append(
                    abs(
                        float(left_record["host_receive_monotonic_s"])
                        - float(right_record["host_receive_monotonic_s"])
                    )
                    * 1000.0
                )
    return {key: _percentile_95(skews) for key, skews in values.items()}


def _metrics_from_reportable(value: ConcurrentRGBMetrics) -> dict[str, object]:
    return {
        "duration_s": value.duration_s,
        "policy_tick_count": value.policy_tick_count,
        "stream_metrics": {
            name: _serialize_stream_metrics(metrics) for name, metrics in value.stream_metrics.items()
        },
        "pairwise_skew_p95_ms": dict(value.pairwise_skew_p95_ms),
        "robot_state_sample_count": value.robot_state_sample_count,
        "robot_state_error_count": value.robot_state_error_count,
        "robot_state_timestamp_traceable": value.robot_state_timestamp_traceable,
        "robot_state_timestamp_monotonic": value.robot_state_timestamp_monotonic,
    }


def run_concurrent(args: argparse.Namespace) -> int:
    names = THREE_RGB_STREAMS if args.camera_candidate == "three" else FIXED_RGB_STREAMS
    run_id = _run_id(f"{args.camera_candidate}-rgb-concurrent")
    run_dir = _reserve_run_directory(args.output_root, run_id)
    profiles = {
        "global_rgb": {
            "device": args.global_device,
            "width": args.width,
            "height": args.height,
            "fps": args.global_fps,
            "fourcc": args.global_fourcc,
        },
        "grasp_rgb": {
            "device": args.grasp_device,
            "width": args.width,
            "height": args.height,
            "fps": args.grasp_fps,
            "fourcc": args.grasp_fourcc,
        },
    }
    if args.camera_candidate == "three":
        profiles["wrist_rgb"] = {
            "device": args.mech_eye_ip,
            "capture_api": "capture_2d",
            "capture_timeout_ms": args.capture_timeout_ms,
        }
    intent = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "mode": "rgb_aubo_state_concurrent",
        "created_at_utc": _utc_now(),
        "authorization_evidence_ref": args.authorization_ref,
        "hardware_operations": [
            "connect_selected_rgb_cameras",
            "capture_rgb_without_image_storage",
            "connect_aubo_rpc",
            "read_joint_positions",
            "read_tcp_pose",
            "disconnect_all",
        ],
        "forbidden_operations": [
            "capture_2d_and_3d",
            "camera_parameter_write",
            "image_pixel_storage",
            "getMotionControl",
            "getIoControl",
            "robot_motion",
            "io_output",
        ],
        "requested_duration_s": args.duration_s,
        "policy_hz": args.policy_hz,
        "state_hz": args.state_hz,
        "candidate_streams": list(names),
        "capture_profiles": profiles,
        "aubo_endpoint": f"{args.aubo_ip}:{args.aubo_port}",
    }
    intent_sha256 = _write_json_new(run_dir / "intent.json", intent)

    password = os.environ.get("A2_AUBO_PASSWORD")
    if password is None:
        password = getpass.getpass("AUBO read-only RPC password (not stored): ")
    if not password:
        raise ValueError("AUBO password cannot be empty")

    cameras: dict[str, Any] = {}
    buffers = {name: SampleBuffer() for name in (*names, "robot_state")}
    threads: list[threading.Thread] = []
    stop = threading.Event()
    state_source = ReadOnlyAuboStateSource(
        ip=args.aubo_ip,
        port=args.aubo_port,
        username=args.aubo_username,
        password=password,
        timeout_ms=args.aubo_timeout_ms,
    )
    groups: list[dict[str, object]] = []
    measured_start = time.monotonic()
    try:
        for name in FIXED_RGB_STREAMS:
            profile = profiles[name]
            camera = OpenCVCamera(
                OpenCVCameraConfig(
                    index_or_path=profile["device"],
                    width=profile["width"],
                    height=profile["height"],
                    fps=profile["fps"],
                    fourcc=profile["fourcc"],
                    warmup_s=args.usb_warmup_s,
                )
            )
            camera.connect(warmup=True)
            cameras[name] = camera
        if args.camera_candidate == "three":
            wrist = MechMindCamera(
                MechMindCameraConfig(
                    ip_address=args.mech_eye_ip,
                    use_depth=False,
                    connect_timeout_ms=args.connect_timeout_ms,
                    capture_timeout_ms=args.capture_timeout_ms,
                )
            )
            wrist.connect(warmup=False)
            cameras["wrist_rgb"] = wrist
        state_source.connect()

        for name in FIXED_RGB_STREAMS:
            thread = threading.Thread(
                target=_monitor_opencv,
                args=(name, cameras[name], buffers[name], stop),
                name=f"phase-a2-{name}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        if "wrist_rgb" in cameras:
            thread = threading.Thread(
                target=_capture_wrist,
                args=(cameras["wrist_rgb"], buffers["wrist_rgb"], stop),
                name="phase-a2-wrist-rgb",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        state_thread = threading.Thread(
            target=_read_aubo_state,
            args=(state_source, buffers["robot_state"], stop, args.state_hz),
            name="phase-a2-aubo-read-only-state",
            daemon=True,
        )
        state_thread.start()
        threads.append(state_thread)
        _wait_for_sources(buffers, args.initial_sample_timeout_s)

        measured_start = time.monotonic()
        deadline = measured_start + args.duration_s
        tick = 0
        while True:
            reference = measured_start + tick / args.policy_hz
            if reference >= deadline:
                break
            remaining = reference - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            actual_reference = time.monotonic()
            stream_records: dict[str, object] = {}
            for name in names:
                sample = buffers[name].latest()
                if sample is None:
                    stream_records[name] = None
                else:
                    stream_records[name] = {
                        "host_receive_monotonic_s": sample.host_receive_monotonic_s,
                        "identity": sample.identity,
                        "content_sha256": sample.content_sha256,
                        "age_ms": (actual_reference - sample.host_receive_monotonic_s) * 1000.0,
                    }
            state = buffers["robot_state"].latest()
            state_record = (
                None
                if state is None
                else {
                    "host_receive_monotonic_s": state.host_receive_monotonic_s,
                    "age_ms": (actual_reference - state.host_receive_monotonic_s) * 1000.0,
                }
            )
            groups.append(
                {
                    "tick": tick,
                    "reference_monotonic_s": actual_reference,
                    "streams": stream_records,
                    "state": state_record,
                }
            )
            tick += 1
        final_wait = deadline - time.monotonic()
        if final_wait > 0:
            time.sleep(final_wait)
        measured_end = time.monotonic()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=max(2.0, args.capture_timeout_ms / 1000.0 + 1.0))
        state_source.disconnect()
        for camera in reversed(tuple(cameras.values())):
            if camera.is_connected:
                camera.disconnect()

    duration_s = measured_end - measured_start
    snapshots = {name: buffers[name].snapshot() for name in (*names, "robot_state")}
    stream_metrics = {
        name: _concurrent_stream_metrics(
            name,
            [
                sample
                for sample in snapshots[name][0]
                if measured_start <= sample.host_receive_monotonic_s <= measured_end
            ],
            [
                error
                for error in snapshots[name][1]
                if measured_start <= float(error["host_monotonic_s"]) <= measured_end
            ],
            groups,
            duration_s,
        )
        for name in names
    }
    observed_profiles = {name: dict(profile) for name, profile in profiles.items()}
    for name in names:
        measured_samples = [
            sample
            for sample in snapshots[name][0]
            if measured_start <= sample.host_receive_monotonic_s <= measured_end
        ]
        if measured_samples:
            observed_profiles[name]["observed_width"] = measured_samples[0].width
            observed_profiles[name]["observed_height"] = measured_samples[0].height
            observed_profiles[name]["model_color_layout"] = "RGB_uint8"
    state_samples = [
        sample
        for sample in snapshots["robot_state"][0]
        if measured_start <= sample.host_receive_monotonic_s <= measured_end
    ]
    state_errors = [
        error
        for error in snapshots["robot_state"][1]
        if measured_start <= float(error["host_monotonic_s"]) <= measured_end
    ]
    concurrent = ConcurrentRGBMetrics(
        duration_s=duration_s,
        policy_tick_count=len(groups),
        stream_metrics=stream_metrics,
        pairwise_skew_p95_ms=_pairwise_skews(groups, names),
        robot_state_sample_count=len(state_samples),
        robot_state_error_count=len(state_errors),
        robot_state_timestamp_traceable=True,
        robot_state_timestamp_monotonic=_timestamps_monotonic(state_samples),
    )
    gate = evaluate_concurrent_rgb_gate(concurrent, required_rgb_streams=names)
    sample_rows: list[dict[str, object]] = []
    for name in names:
        samples, errors = snapshots[name]
        sample_rows.extend({"record_type": "frame", **asdict(sample)} for sample in samples)
        sample_rows.extend(
            {"record_type": "error", "stream_name": name, **error} for error in errors
        )
    sample_rows.extend(
        {
            "record_type": "robot_state",
            "host_receive_monotonic_s": sample.host_receive_monotonic_s,
            "joints_rad": sample.joints_rad,
            "tcp_pose_m_rad": sample.tcp_pose_m_rad,
        }
        for sample in state_samples
    )
    sample_rows.extend(
        {"record_type": "error", "stream_name": "robot_state", **error}
        for error in state_errors
    )
    samples_sha256 = _write_jsonl_new(run_dir / "samples.jsonl", sample_rows)
    groups_sha256 = _write_jsonl_new(run_dir / "policy_groups.jsonl", groups)
    report = {
        "schema_version": RGB_GATE_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "mode": "rgb_aubo_state_concurrent",
        "authorization_evidence_ref": args.authorization_ref,
        "depth_requested": False,
        "camera_parameters_changed": False,
        "aubo_motion_or_io_interface_obtained": False,
        "image_pixels_stored": False,
        "candidate_streams": list(names),
        "capture_profiles": observed_profiles,
        "metrics": _metrics_from_reportable(concurrent),
        "gate": gate.to_dict(),
        "evidence_sha256": {
            "intent.json": intent_sha256,
            "samples.jsonl": samples_sha256,
            "policy_groups.jsonl": groups_sha256,
        },
    }
    report_sha256 = _write_json_new(run_dir / "report.json", report)
    print(json.dumps({"run_dir": str(run_dir), "report_sha256": report_sha256, **gate.to_dict()}, indent=2))
    return 0 if gate.decision.value == "pass" else 2


def _positive_duration(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < MIN_DURATION_S:
        raise argparse.ArgumentTypeError(f"duration must be at least {MIN_DURATION_S:g} seconds")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--authorization-ref", required=True)
        subparser.add_argument("--confirm-read-only-hardware-test", action="store_true", required=True)
        subparser.add_argument("--duration-s", type=_positive_duration, default=60.0)
        subparser.add_argument("--output-root", type=Path, default=Path("artifacts/rgb_gate"))
        subparser.add_argument("--connect-timeout-ms", type=int, default=5000)
        subparser.add_argument("--capture-timeout-ms", type=int, default=2000)

    isolated = subparsers.add_parser("isolated", help="Run Mech-Eye capture_2d only")
    shared(isolated)
    isolated.add_argument("--mech-eye-ip", required=True)
    isolated.set_defaults(run=run_isolated)

    concurrent = subparsers.add_parser(
        "concurrent", help="Run two/three RGB streams with read-only AUBO state"
    )
    shared(concurrent)
    concurrent.add_argument("--camera-candidate", choices=("two", "three"), required=True)
    concurrent.add_argument("--global-device", required=True)
    concurrent.add_argument("--grasp-device", required=True)
    concurrent.add_argument("--mech-eye-ip")
    concurrent.add_argument("--width", type=int, default=640)
    concurrent.add_argument("--height", type=int, default=480)
    concurrent.add_argument("--global-fps", type=float, default=30.0)
    concurrent.add_argument("--grasp-fps", type=float, default=25.0)
    concurrent.add_argument("--global-fourcc", default="MJPG")
    concurrent.add_argument("--grasp-fourcc", default="MJPG")
    concurrent.add_argument("--usb-warmup-s", type=_positive_float, default=3.0)
    concurrent.add_argument("--policy-hz", type=_positive_float, default=10.0)
    concurrent.add_argument("--state-hz", type=_positive_float, default=25.0)
    concurrent.add_argument("--initial-sample-timeout-s", type=_positive_float, default=10.0)
    concurrent.add_argument("--aubo-ip", required=True)
    concurrent.add_argument("--aubo-port", type=int, default=30004)
    concurrent.add_argument("--aubo-username", default="aubo")
    concurrent.add_argument("--aubo-timeout-ms", type=int, default=1000)
    concurrent.set_defaults(run=run_concurrent)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.authorization_ref.strip():
        parser.error("--authorization-ref must be non-empty")
    if args.mode == "concurrent" and args.camera_candidate == "three" and not args.mech_eye_ip:
        parser.error("--mech-eye-ip is required for the three-camera candidate")
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
