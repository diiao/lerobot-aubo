#!/usr/bin/env python
"""Prepare saved approach frames and the existing review page; no model/hardware.

Run prepare once. Import separately authorized frozen predictions later, then
review them in the fixed HTML entry. Browser edits are exported as decisions;
they do not silently change the training sidecar.
"""

import argparse
import json
from pathlib import Path

from lerobot.bamboo_sorting.capture.target_review import *
from lerobot.bamboo_sorting.perception.target_review import *
from lerobot.bamboo_sorting.policy.target_verification import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/joint_target_pilot"))
    parser.add_argument("--evidence-root", type=Path, default=Path("artifacts/joint_target_pilot"))
    parser.add_argument("--stage", choices=("capture-check", "prepare", "render", "import-predictions", "import-seeds", "separate", "import-review", "verify"), default="prepare")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--single-strip-scenes", action="store_true",
                        help="Only for known single-strip scenes: use a sole detection as an unreviewed draft; never use for A/B pairs")
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    dataset, evidence = args.dataset_root.resolve(), args.evidence_root.resolve()
    folder = evidence / "target_annotations"
    if args.stage == "capture-check":
        print(json.dumps(prepare_capture_check(dataset, evidence), ensure_ascii=False, indent=2))
        return
    if args.stage == "prepare":
        if folder.exists():
            raise FileExistsError(f"preserve existing annotation work: {folder}")
        spec, inference = approach_manifest(dataset, evidence)
        folder.mkdir()
        for name in ("images", "masks", "outlines"):
            (folder / name).mkdir()
        extract_images(dataset, folder, spec)
        write(folder / "annotations.json", spec)
        write(folder / "inference_manifest.json", inference)
        write(folder / "inference_request.json", {
            "authorization": "not_requested_yet", "scope": "one frozen offline inference over these approach frames",
            "samples": len(inference["samples"]), "input": "inference_manifest.json",
            "remote_host_hint": "gpu", "remote_paths_verified": False,
            "remote_project_hint": "/home/rentao/program/lerobot-aubo-strip-segmentation-20261005",
            "remote_python_hint": "/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python",
            "model_hint": "outputs/short_strip_mask2former_baseline/best/",
            "remote_input_relative": f"artifacts/{evidence.name}/target_annotations",
            "remote_output_relative": f"artifacts/{evidence.name}/target_annotations/predictions",
            "local_output": str(folder / "predictions"),
            "entrypoint": "examples/phone_to_auboi10/predict_strip_images.py",
            "threshold": .5, "mask_threshold": .5, "overlap_mask_area_threshold": .8,
            "batch_size": 2, "resize": False, "image_size": [640, 480],
            "optimizer_steps": 0, "reviewed_on_import": False,
            "pre_execution": "verify remote Python, best checkpoint, input/output absence; copy only listed RGB inputs and standalone predictor after authorization"})
    else:
        spec = read(folder / "annotations.json")
        if Path(spec["source_dataset_root"]).resolve() != dataset:
            raise ValueError("source dataset mismatch")
        if args.stage == "import-predictions":
            if args.predictions is None:
                parser.error("--predictions required")
            import_predictions(folder, spec, args.predictions.resolve(),
                               single_strip_scenes=args.single_strip_scenes)
        if args.stage == "import-review":
            if args.decisions is None:
                parser.error("--decisions required")
            import_decisions(folder, spec, args.decisions.resolve())
        if args.stage == "import-seeds":
            if args.decisions is None:
                parser.error("--decisions required")
            import_seeds(folder, spec, args.decisions.resolve())
        if args.stage == "separate":
            separate_predictions(folder, spec)
        if args.stage == "verify":
            verify_dataset(dataset, evidence, folder, spec)
            return
    render(evidence, folder, spec)


if __name__ == "__main__":
    main()
