"""Offline annotation and single-camera lifetime checks; no physical devices."""

import builtins
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.bamboo_sorting.placement_sequence import PlacementSequence, difference_candidate


@pytest.fixture
def entry():
    path = Path(__file__).parents[2] / "examples/phone_to_auboi10/capture_placement_sequence.py"
    spec = importlib.util.spec_from_file_location("placement_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frames():
    empty = np.zeros((80, 120, 3), np.uint8)
    first = empty.copy()
    first[30:40, 10:110] = [170, 100, 60]
    second = first.copy()
    second[10:70, 50:60] = [210, 180, 100]
    return [empty, first, second]


def forbid_devices(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyaubo_sdk", "lerobot.cameras", "lerobot.robots", "lerobot.teleoperators")):
            raise AssertionError(f"hardware imported: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_plan_uses_only_old_global_without_writing_or_devices(entry, tmp_path, monkeypatch, capsys):
    forbid_devices(monkeypatch)
    assert entry.main(["--output", str(tmp_path / "absent"), "--placement-id", "pile-a"]) == 0
    plan = json.loads(capsys.readouterr().out)
    mapping = plan["camera_configuration"]["camera_mapping"]
    assert list(mapping) == ["global_rgb"]
    assert "GENERAL_WEBCAM" in mapping["global_rgb"]["device"]
    assert list(tmp_path.iterdir()) == []


def test_offline_crossing_retains_sources_visibility_and_group(entry, frames, tmp_path, monkeypatch):
    paths = []
    for i, rgb in enumerate(frames):
        path = tmp_path / f"source_{i}.png"
        assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    forbid_devices(monkeypatch)
    output = tmp_path / "labels"
    args = ["--output", str(output), "--placement-id", "pile-a", "--split", "test", "--input-images", *paths]
    assert entry.main(args) == 0
    manifest = json.loads((output / "sequence.json").read_text())
    assert manifest["split"] == "test" and manifest["finished"]
    assert manifest["placement_id"] == "pile-a" and not manifest["training_ready"]
    labels = json.loads((output / "frame_002_labels.json").read_text())
    assert labels["top_ids_candidate"] == ["strip_002"]
    assert labels["relations_candidate"] == [{"above": "strip_002", "below": "strip_001", "overlap_pixels": 100}]
    assert all(instance["graspable"] is None for instance in labels["instances"])
    old_visible = cv2.imread(str(output / "frame_002_strip_001_visible.png"), cv2.IMREAD_GRAYSCALE)
    old_full = cv2.imread(str(output / "frame_001_candidate.png"), cv2.IMREAD_GRAYSCALE)
    assert np.count_nonzero(old_visible) == 900 and np.count_nonzero(old_full) == 1000
    for i, rgb in enumerate(frames):
        assert np.array_equal(cv2.cvtColor(cv2.imread(str(output / f"frame_{i:03d}.png")), cv2.COLOR_BGR2RGB), rgb)
    with pytest.raises(FileExistsError):
        entry.main(args)


def test_difference_flags_missing_multiple_and_global_changes(frames):
    empty, first, _ = frames
    mask, report = difference_candidate(empty, empty)
    assert not mask.any() and report["warnings"] == ["no_new_object_detected"]
    moved = first.copy()
    moved[55:65, 10:110] = 200
    moved[0, 0] = 255  # Small noise should not become an instance.
    mask, report = difference_candidate(empty, moved)
    assert report["component_count"] == 2 and not mask[0, 0]
    assert "multiple_changed_components" in report["warnings"]
    assert "large_scene_change" in difference_candidate(empty, empty + 100)[1]["warnings"]
    with pytest.raises(ValueError, match="identical"):
        difference_candidate(empty, first[:40])


@pytest.mark.parametrize("failure", [False, True])
def test_live_opens_only_global_and_restores_controls_on_exit(entry, frames, tmp_path, monkeypatch, failure):
    import lerobot.cameras.opencv

    events = []
    initial = {cv2.CAP_PROP_AUTO_EXPOSURE: .75, cv2.CAP_PROP_EXPOSURE: 100.,
               cv2.CAP_PROP_AUTO_WB: 1., cv2.CAP_PROP_WB_TEMPERATURE: 4500.}
    values = initial.copy()

    class Capture:
        def get(self, key):
            return values[key]

        def set(self, key, value):
            values[key] = value
            return True

    class Camera:
        def __init__(self, config):
            events.append(("created", str(config.index_or_path)))
            self.is_connected = False
            self.thread = None
            self.videocapture = Capture()
            self.reads = 0

        def connect(self, warmup):
            self.is_connected = True

        def _stop_read_thread(self):
            events.append(("reader", "stopped"))

        def _start_read_thread(self):
            events.append(("reader", "started"))

        def read(self):
            self.reads += 1
            return frames[0 if self.reads <= 11 else 1].copy()

        def disconnect(self):
            events.append(("closed", True))
            self.is_connected = False

    monkeypatch.setattr(lerobot.cameras.opencv, "OpenCVCamera", Camera)
    monkeypatch.setattr(entry.sys.stdin, "isatty", lambda: True)
    replies = iter(["", "", "q"])
    monkeypatch.setattr(builtins, "input", lambda _: next(replies))
    if failure:
        def fail(*args, **kwargs):
            raise OSError("disk write failed")
        monkeypatch.setattr(PlacementSequence, "append", fail)
    output = tmp_path / "capture"
    args = ["--record", "--output", str(output), "--placement-id", "pile-live", "--strips", "3"]
    if failure:
        with pytest.raises(OSError, match="disk write"):
            entry.main(args)
    else:
        assert entry.main(args) == 0
    assert values == initial
    opened = [value for kind, value in events if kind == "created"]
    assert len(opened) == 1 and "GENERAL_WEBCAM" in opened[0]
    assert events[-1] == ("closed", True)
    manifest = json.loads((output / "sequence.json").read_text())
    assert manifest["camera_controls"]["lock_verified"]
    assert all(manifest["camera_controls"]["restore_succeeded"].values())
    assert manifest["stop_reason"] == ("error" if failure else "operator_stopped")
    assert not manifest["finished"]
    assert len(manifest["frames"]) == (0 if failure else 2)


def test_unsupported_controls_are_not_reported_as_locked(entry):
    class Unsupported:
        def get(self, key):
            return 0.

        def set(self, key, value):
            return False

    capture = Unsupported()
    report = entry.lock_controls(capture, entry.read_controls(capture))
    assert not report["lock_verified"]
    assert report["warnings"] == ["camera_control_lock_unverified"]


def test_roi_and_chroma_remove_distant_motion_and_neutral_shadow():
    previous = np.full((80, 120, 3), 150, np.uint8)
    current = previous.copy()
    current[2:18, 2:20] = [255, 0, 0]  # Person outside the working region.
    current[25:35, 40:95] = [210, 170, 105]  # New strip.
    current[55:65, 40:70] = 70  # Brightness-only shadow.
    options = dict(roi=[30, 20, 110, 70], min_area=20)
    raw, _ = difference_candidate(previous, current, **options)
    refined, report = difference_candidate(previous, current, min_chroma_change=8, **options)
    assert not raw[:20].any() and raw[55:65, 40:70].all()
    assert refined[25:35, 40:95].all() and refined.sum() == 550
    assert report["chroma_filtered_pixels"] == 300
    assert "chroma_filter_may_remove_same_color_crossings" in report["warnings"]


def test_same_color_crossing_is_unresolved_instead_of_all_top(tmp_path):
    sequence = PlacementSequence(tmp_path / "sequence", placement_id="same-color", split="train",
                                 configuration={}, min_chroma_change=8)
    frame = np.zeros((80, 120, 3), np.uint8)
    sequence.append(frame)
    frame[30:40, 10:110] = [210, 170, 105]
    sequence.append(frame)
    frame[10:70, 50:60] = [210, 170, 105]
    sequence.append(frame)
    labels = json.loads((sequence.output / "frame_002_labels.json").read_text())
    assert labels["relations_candidate"] == []  # Unchanged intersection is invisible to difference.
    assert labels["top_ids_candidate"] is None
    assert labels["layer_order_status"] == "unresolved" and not labels["training_ready"]
    assert (sequence.output / "frame_002_unfiltered.png").is_file()


def test_invalid_roi_rejected_before_device_import(entry, tmp_path, monkeypatch):
    forbid_devices(monkeypatch)
    with pytest.raises(SystemExit):
        entry.main(["--record", "--output", str(tmp_path / "absent"), "--placement-id", "bad-roi",
                    "--roi", "280", "200", "650", "480"])
    assert not (tmp_path / "absent").exists()


def test_candidate_clipped_by_roi_remains_flagged(frames):
    mask, report = difference_candidate(frames[0], frames[1], roi=[20, 20, 100, 60])
    assert mask.sum() == 800
    assert "candidate_touches_roi_boundary" in report["warnings"]


@pytest.mark.parametrize("exposure_supported", [False, True])
def test_realistic_unsupported_readbacks_never_written_or_restored(entry, exposure_supported):
    values = {cv2.CAP_PROP_AUTO_EXPOSURE: .75 if exposure_supported else -1.,
              cv2.CAP_PROP_EXPOSURE: 625., cv2.CAP_PROP_AUTO_WB: -1., cv2.CAP_PROP_WB_TEMPERATURE: -1.}
    initial = values.copy()
    writes = []

    class Capture:
        def get(self, key):
            return values[key]

        def set(self, key, value):
            assert value != -1
            writes.append((key, value))
            values[key] = value
            return True

    camera = Capture()
    before = entry.read_controls(camera)
    report = entry.lock_controls(camera, before)
    assert not report["lock_verified"]
    assert report["skipped"]["auto_white_balance"] == "unsupported_readback"
    restored = entry.restore_controls(camera, before, report["set_succeeded"])
    assert values == initial
    assert len(writes) == (4 if exposure_supported else 0)
    assert set(restored) == ({"auto_exposure", "exposure"} if exposure_supported else set())
