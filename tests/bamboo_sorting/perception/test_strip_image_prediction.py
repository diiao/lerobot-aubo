"""Prediction contract and shared output behavior, using synthetic images/models."""

from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tests.bamboo_sorting.paths import REPO_ROOT


@pytest.fixture
def prediction(tmp_path):
    api = runpy.run_path(str(REPO_ROOT / "examples/phone_to_auboi10/predict_strip_images.py"))
    inputs, model, output = tmp_path / "input", tmp_path / "training/best", tmp_path / "predictions"
    inputs.mkdir()
    model.mkdir(parents=True)
    output.mkdir()
    Image.new("RGB", (640, 480)).save(inputs / "raw.png")
    spec = {"purpose": api["TEST_PURPOSE"], "source_run": "synthetic", "image_size": [640, 480],
            "independent_test": True, "human_ground_truth_review_complete": True,
            "samples": [{"id": "sample", "image": "raw.png", "split": "test"}]}
    manifest = inputs / "manifest.json"
    manifest.write_text(json.dumps(spec))
    (model / "strip_contract.json").write_text(json.dumps({
        "classes": api["CLASSES"], "image_size": [640, 480], "resize": False}))
    (model.parent / "complete.json").write_text(json.dumps({
        "best_epoch": 5, "best_reloaded_and_evaluated": True}))
    return api, manifest, model, output, spec


def test_independent_test_requires_review_and_test_split(prediction):
    api, manifest, _, _, spec = prediction
    assert api["validate_samples"](manifest)["independent_test"]
    spec["samples"][0]["split"] = "train"
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="test-only"):
        api["validate_samples"](manifest)
    spec["samples"][0]["split"] = "test"
    spec["human_ground_truth_review_complete"] = False
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="reviewed ground truth"):
        api["validate_samples"](manifest)
    spec["purpose"] = "static_placement_development_diagnostic"
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="development diagnostics"):
        api["validate_samples"](manifest)


def test_plan_has_no_writes_and_preserves_default_directory_rule(prediction, monkeypatch, capsys):
    api, manifest, model, output, spec = prediction
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    argv = ["--manifest", str(manifest), "--model-path", str(model), "--output", str(output)]
    assert api["main"](argv + ["--output-prefix", "test_old_"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan_only" and plan["independent_test"] and plan["optimizer_steps"] == 0
    assert list(output.iterdir()) == []
    with pytest.raises(FileExistsError):
        api["main"](argv)
    with pytest.raises(ValueError, match="simple filename prefix"):
        api["main"](argv + ["--output-prefix", "../escape_"])
    spec.update(purpose="static_placement_development_diagnostic", independent_test=False)
    manifest.write_text(json.dumps(spec))
    argv[-1] = str(output.parent / "new_output")
    assert api["main"](argv) == 0
    assert not json.loads(capsys.readouterr().out)["independent_test"]
    assert not Path(argv[-1]).exists()


def test_two_prefixed_predictions_preserve_existing_results(prediction, monkeypatch):
    api, manifest, model, output, _ = prediction
    legacy = output / "predictions.json"
    legacy.write_text("old result unchanged")

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, **kwargs):
            return SimpleNamespace(to=lambda device: {})

        def post_process_instance_segmentation(self, *args, **kwargs):
            mask = np.full((480, 640), -1, dtype=np.int64)
            mask[10:20, 30:60] = 7
            tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: mask))
            return [{"segmentation": tensor, "segments_info": [{"id": 7, "label_id": 1, "score": .9}]}]

    class Model:
        config = SimpleNamespace(id2label={0: "top_strip", 1: "covered_strip"})

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(), {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

        def requires_grad_(self, value):
            assert value is False
            return self

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return None

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda value: None, set_num_threads=lambda value: None, inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Mask2FormerForUniversalSegmentation=Model, Mask2FormerImageProcessor=Processor))
    argv = ["--manifest", str(manifest), "--model-path", str(model), "--output", str(output), "--execute"]
    for prefix in ("test_old_", "test_new_"):
        assert api["main"](argv + ["--output-prefix", prefix]) == 0
        report = json.loads((output / (prefix + "predictions.json")).read_text())
        assert report["complete"] and report["independent_test"] and report["optimizer_steps"] == 0
        row = report["samples"][0]
        assert row["instance_map"] == prefix + "sample.png"
        assert row["segments"][0]["class_id"] == 1
        assert set(np.unique(np.array(Image.open(output / row["instance_map"])))) == {0, 1}
        assert (output / (prefix + "plan.json")).is_file()
        assert (output / (prefix + "loading_info.json")).is_file()
    assert legacy.read_text() == "old result unchanged"
    with pytest.raises(FileExistsError):
        api["main"](argv + ["--output-prefix", "test_old_"])
