import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "phone_to_auboi10"
    / "audit_recorded_batch.py"
)
MANIFEST_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "phone_to_auboi10"
    / "datasets"
    / "bamboo_act_report_manifest.json"
)
EXTENDED_SPLIT_PATH = (
    Path(__file__).parents[2]
    / "configs"
    / "aubo_i10"
    / "act_extended137_drop_gripper_12d_v1.json"
)

SPEC = importlib.util.spec_from_file_location("audit_recorded_batch_under_test", SCRIPT_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


ACTION_NAMES = list(audit.EXPECTED_ACTION_NAMES)
STATE_NAMES = list(audit.EXPECTED_STATE_NAMES)


def write_color_mp4(path: Path, n_frames: int, fps: int = 25) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = 640
    stream.height = 480
    stream.pix_fmt = "yuv420p"
    for index in range(n_frames):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = (index * 3) % 255
        image[:, :, 1] = 40
        image[:, :, 2] = 80
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _info(total_episodes: int, total_frames: int, fps: int = 25) -> dict:
    return {
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "robot_type": "aubo_i10",
        "features": {
            "action": {"dtype": "float32", "shape": [8], "names": ACTION_NAMES},
            "observation.state": {"dtype": "float32", "shape": [13], "names": STATE_NAMES},
            "observation.images.handeye": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.images.fixed": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
        },
    }


def _episode_arrays(n_frames: int, ons: list[int], offs: list[int], lift_after: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((n_frames, 8), dtype=np.float32)
    states = np.zeros((n_frames, 13), dtype=np.float32)
    actions[:, 1] = np.arange(n_frames, dtype=np.float32) * 0.0005
    states[:, 6] = actions[:, 1]
    states[:, 8] = 0.090
    gripper = np.zeros(n_frames, dtype=np.float32)
    for on, off in zip(ons, offs, strict=True):
        gripper[on:off] = 100.0
        lift_at = on + lift_after
        if on < lift_at < off:
            states[lift_at:off, 8] = 0.096
    actions[:, 7] = gripper
    states[:, 12] = gripper
    return actions, states


def make_batch(
    tmp_path: Path,
    *,
    name: str,
    strip_count: int,
    planned: int,
    n_frames: int = 80,
    ons: list[int] | None = None,
    offs: list[int] | None = None,
    lift_after: int | None = None,
    fps: int = 25,
    extra_gripper_pulse: bool = False,
    write_videos: bool = True,
) -> tuple[Path, Path]:
    root = tmp_path / name
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    if ons is None:
        ons = [10] if strip_count == 1 else [10, 45]
    if offs is None:
        offs = [70] if strip_count == 1 else [35, 70]
    if lift_after is None:
        lift_after = 45 if strip_count == 1 else 18

    actions, states = _episode_arrays(n_frames, ons, offs, lift_after)
    if extra_gripper_pulse:
        actions[12, 7] = 0.0
        actions[13, 7] = 100.0
        states[12, 12] = 0.0
        states[13, 12] = 100.0

    info = _info(1, n_frames, fps=fps)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    table = pa.table(
        {
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 8)),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 13)),
            "episode_index": pa.array(np.zeros(n_frames, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n_frames, dtype=np.int64)),
            "timestamp": pa.array((np.arange(n_frames) / fps).astype(np.float32)),
            "index": pa.array(np.arange(n_frames, dtype=np.int64)),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    duration = n_frames / fps
    episode_record = {
        "episode_index": 0,
        "length": n_frames,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": n_frames,
        "videos/observation.images.handeye/chunk_index": 0,
        "videos/observation.images.handeye/file_index": 0,
        "videos/observation.images.handeye/from_timestamp": 0.0,
        "videos/observation.images.handeye/to_timestamp": duration,
        "videos/observation.images.fixed/chunk_index": 0,
        "videos/observation.images.fixed/file_index": 0,
        "videos/observation.images.fixed/from_timestamp": 0.0,
        "videos/observation.images.fixed/to_timestamp": duration,
    }
    pq.write_table(pa.Table.from_pylist([episode_record]), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    if write_videos:
        for camera in ("observation.images.handeye", "observation.images.fixed"):
            write_color_mp4(root / "videos" / camera / "chunk-000" / "file-000.mp4", n_frames, fps=fps)

    layout = "single" if strip_count == 1 else "cross"
    grasp = "pick_place_one" if strip_count == 1 else "sequential_clear_two"
    manifest = {
        "batches": {
            name: {
                "strip_count": strip_count,
                "layout": layout,
                "grasp": grasp,
                "planned_success_episodes": planned,
                "local_train_episodes": list(range(max(planned - 1, 0))),
                "local_val_episodes": [planned - 1] if planned else [],
                "status": "planned",
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path


def test_single_strip_valid_batch_passes(tmp_path):
    root, manifest = make_batch(tmp_path, name="bamboo_act_report_s15", strip_count=1, planned=1)
    report = audit.audit_batch(root, manifest)
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["episodes"][0]["gripper_switch_count"] == 2
    assert report["physical_grasp"]["offline_proof"] is False


def test_double_strip_requires_four_switches(tmp_path):
    root, manifest = make_batch(tmp_path, name="bamboo_act_report_s23", strip_count=2, planned=1)
    report = audit.audit_batch(root, manifest)
    assert report["passed"] is True
    assert report["episodes"][0]["gripper_switch_count"] == 4


def test_gripper_jitter_fails(tmp_path):
    root, manifest = make_batch(
        tmp_path,
        name="bamboo_act_report_s15",
        strip_count=1,
        planned=1,
        extra_gripper_pulse=True,
    )
    report = audit.audit_batch(root, manifest)
    assert report["passed"] is False
    assert any("夹爪切换" in item for item in report["errors"])


def test_episode_count_mismatch_fails(tmp_path):
    root, manifest = make_batch(tmp_path, name="bamboo_act_report_s15", strip_count=1, planned=5)
    report = audit.audit_batch(root, manifest)
    assert report["passed"] is False
    assert any("episode 数应为 5" in item for item in report["errors"])


def test_wrong_fps_fails(tmp_path):
    root, manifest = make_batch(
        tmp_path,
        name="bamboo_act_report_s15",
        strip_count=1,
        planned=1,
        fps=30,
        n_frames=60,
        ons=[12],
        offs=[48],
        lift_after=40,
    )
    report = audit.audit_batch(root, manifest)
    assert report["passed"] is False
    assert any("25" in item for item in report["errors"])


def test_missing_lift_fails(tmp_path):
    root, manifest = make_batch(
        tmp_path,
        name="bamboo_act_report_s15",
        strip_count=1,
        planned=1,
        lift_after=10_000,
    )
    report = audit.audit_batch(root, manifest)
    assert report["passed"] is False
    assert any("未上升" in item for item in report["errors"])


def test_report_dir_refuses_overwrite_and_does_not_touch_dataset(tmp_path):
    root, manifest = make_batch(tmp_path, name="bamboo_act_report_s15", strip_count=1, planned=1)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    with pytest.raises(FileExistsError):
        audit.write_report({"passed": True}, report_dir)
    before = {path.relative_to(root): path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    report = audit.audit_batch(root, manifest)
    after = {path.relative_to(root): path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    assert report["passed"] is True
    assert before == after


def test_manifest_keeps_s01_s14_and_records_completed_extension():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    plan = manifest["extension_plan"]
    assert manifest["next_batch"] is None
    assert manifest["paused_after"] == "s28b"
    assert plan["status"] == "recording_complete"
    assert plan["record_recovery_starts"] is False
    assert plan["aggregation"]["status"] == "complete"
    assert plan["aggregation"]["must_not_overwrite"] == "bamboo_act_report_full"
    assert plan["training"]["status"] == "not_authorized"
    assert "independent recording-session" in " ".join(plan["validation_limitations"])
    for index in range(1, 15):
        batch = manifest["batches"][f"bamboo_act_report_s{index:02d}"]
        assert batch["status"] == "recorded"
        assert "planned_success_episodes" in batch

    extension_names = [
        *(f"bamboo_act_report_s{index:02d}" for index in range(15, 24)),
        "bamboo_act_report_s23b",
        *(f"bamboo_act_report_s{index:02d}" for index in range(24, 27)),
        "bamboo_act_report_s26b",
        "bamboo_act_report_s27",
        "bamboo_act_report_s28",
        "bamboo_act_report_s28b",
    ]
    extension = [manifest["batches"][name] for name in extension_names]
    assert sum(batch["actual_episodes"] for batch in extension) == 72
    assert sum(
        batch["actual_episodes"] for batch in extension if batch["strip_count"] == 1
    ) == 30
    assert sum(
        batch["actual_episodes"] for batch in extension if batch["strip_count"] == 2
    ) == 42
    assert sum(len(batch["local_train_episodes"]) for batch in extension) == 58
    assert sum(len(batch["local_val_episodes"]) for batch in extension) == 14

    for batch in extension:
        roles = batch["local_train_episodes"] + batch["local_val_episodes"]
        assert sorted(roles) == list(range(batch["actual_episodes"]))
    assert plan["expected_after_all_new_batches"] == {
        "total_episodes": 137,
        "train_episodes": 123,
        "val_episodes": 14,
        "original_s01_s14_all_to_train": 65,
        "new_train_episodes": 58,
        "new_val_episodes": 14,
    }
    split = json.loads(EXTENDED_SPLIT_PATH.read_text(encoding="utf-8"))
    train = split["train_episodes"]
    val = split["val_episodes"]
    assert len(train) == 123
    assert len(val) == 14
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(137))
    assert val == plan["aggregation"]["global_episode_numbers"]["validation"]
    assert split["dataset_info_sha256"] == plan["aggregation"]["dataset_info_sha256"]
    assert split["dataset_summary"]["train_frames"] == 164676
    assert split["dataset_summary"]["val_frames"] == 15618
