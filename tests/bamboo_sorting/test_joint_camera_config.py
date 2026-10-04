"""Camera selection and diagnostics with no physical devices."""

import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.bamboo_sorting import joint_camera_config as config


def entry(name):
    path = Path(__file__).parents[2] / "examples/phone_to_auboi10" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"camera_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["record_joint", "diag_preflight_cams"])
@pytest.mark.parametrize("selection", [None, "original-global"])
def test_plan_selects_new_camera_or_old_rollback_without_devices(name, selection, tmp_path, monkeypatch, capsys):
    module = entry(name)
    args = (["--dataset-root", str(tmp_path / "data"), "--evidence-root", str(tmp_path / "evidence"),
             "--split", "train"] if name == "record_joint" else ["--plan", "--output", str(tmp_path / "preview")])
    if selection:
        args += ["--camera-set", selection]
    real_import = builtins.__import__

    def guarded(module_name, *a, **kw):
        if module_name.startswith(("pyaubo_sdk", "lerobot.cameras", "lerobot.robots", "lerobot.teleoperators")):
            raise AssertionError(f"hardware import: {module_name}")
        return real_import(module_name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert module.main(args) == 0
    plan, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert plan["camera_set"] == (selection or "wide-global")
    assert ("GENERAL_WEBCAM" if selection == "original-global" else "2MP_USB_Camera") in plan["camera_mapping"]["global_rgb"]["device"]
    assert "Sonix" in plan["camera_mapping"]["grasp_rgb"]["device"]
    expected_global_shape = [480, 640, 3] if selection == "original-global" else [1080, 1920, 3]
    if name == "record_joint":
        assert plan["dataset_features"]["observation.images.global_rgb"]["shape"] == expected_global_shape
        assert plan["dataset_features"]["observation.images.grasp_rgb"]["shape"] == [480, 640, 3]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("selection", ["original-global", "wide-global"])
def test_diagnostic_opens_selected_profiles_and_saves_configuration(selection, tmp_path, monkeypatch):
    import lerobot.cameras.opencv

    module = entry("diag_preflight_cams")
    opened, closed = [], []

    class FakeCamera:
        def __init__(self, profile):
            self.config = profile
            self.is_connected = False
            self.videocapture = SimpleNamespace(get=lambda key: 30.0)

        def connect(self, warmup):
            opened.append(self.config)
            self.is_connected = True

        def disconnect(self):
            closed.append(self.config)
            self.is_connected = False

    monkeypatch.setattr(lerobot.cameras.opencv, "OpenCVCamera", FakeCamera)
    monkeypatch.setattr(module, "measure_camera", lambda *a, **kw: [])
    output = tmp_path / "preview"
    assert module.main(["--camera-set", selection, "--output", str(output)]) == 0
    saved = json.loads((output / "camera_configuration.json").read_text())
    assert len(opened) == len(closed) == 2
    for actual, expected in zip(opened, saved["camera_mapping"].values(), strict=True):
        assert str(actual.index_or_path) == expected["device"]
        assert (actual.width, actual.height, actual.fps, actual.fourcc) == (
            expected["width"], expected["height"], expected["fps"], expected["fourcc"])


@pytest.mark.parametrize("change, message", [("resolution", "positive integers"), ("duplicate", "different devices")])
def test_new_config_rejects_incompatible_frames_or_duplicate_camera(change, message, tmp_path, monkeypatch):
    value = json.loads(config.CAMERA_SET_PATHS["wide-global"].read_text())
    if change == "resolution":
        value["capture_profiles"]["global_rgb"]["width"] = 0
    else:
        value["capture_profiles"]["global_rgb"]["device"] = value["capture_profiles"]["grasp_rgb"]["device"]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    monkeypatch.setitem(config.CAMERA_SET_PATHS, "wide-global", path)
    with pytest.raises(ValueError, match=message):
        config.load_joint_camera_configuration("wide-global")
