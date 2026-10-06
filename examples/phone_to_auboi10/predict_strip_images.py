"""Frozen Mask2Former image diagnostic. Default validates/prints plan without inference.

This accepts raw RGB samples, not a train/test dataset or target labels. It is
standalone so remote deployment does not replace existing training sources.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

CLASSES = [{"id": 0, "name": "top_strip"}, {"id": 1, "name": "covered_strip"}]


def validate_samples(manifest_path):
    path = Path(manifest_path).resolve()
    spec = json.loads(path.read_text())
    if spec.get("purpose") != "target_video_development_diagnostic" or spec.get("image_size") != [640, 480]:
        raise ValueError("expected 640x480 target-video diagnostic")
    ids = [row["id"] for row in spec["samples"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("unique sample IDs required")
    for row in spec["samples"]:
        if Path(row["id"]).name != row["id"] or row["id"] in (".", ".."):
            raise ValueError("invalid sample ID")
        image = (path.parent / row["image"]).resolve()
        if not image.is_relative_to(path.parent):
            raise ValueError("image path escapes manifest directory")
        with Image.open(image) as im:
            if im.size != (640, 480) or im.mode != "RGB":
                raise ValueError("raw RGB image must be 640x480")
    return spec


def instance_png(segmentation, segments_info):
    """Reserve PNG zero for background and record each ID-to-class mapping."""
    png = np.zeros(segmentation.shape, np.uint16)
    segments = []
    for mask_id, segment in enumerate(segments_info, 1):
        label = int(segment["label_id"])
        if label not in (0, 1):
            raise ValueError("unknown predicted class")
        pixels = segmentation == segment["id"]
        png[pixels] = mask_id
        segments.append({"mask_id": mask_id, "class_id": label, "score": float(segment["score"]),
                         "pixels": int(pixels.sum())})
    return png, segments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    spec = validate_samples(args.manifest)
    contract = json.loads((args.model_path / "strip_contract.json").read_text())
    completed = json.loads((args.model_path.parent / "complete.json").read_text())
    if args.model_path.name != "best" or completed.get("best_reloaded_and_evaluated") is not True:
        raise ValueError("completed frozen best checkpoint required")
    if contract["classes"] != CLASSES or contract["image_size"] != [640, 480] or contract["resize"] is not False:
        raise ValueError("checkpoint contract mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    for source in (args.model_path.parent, args.manifest.parent):
        if source.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("output must not contain model or inputs")
    if args.output.resolve().is_relative_to(args.model_path.parent.resolve()):
        raise ValueError("output must be outside the training run")
    plan = {"source_run": spec["source_run"], "samples": len(spec["samples"]),
            "model_path": str(args.model_path.resolve()), "best_epoch": completed["best_epoch"],
            "threshold": 0.5, "mask_threshold": 0.5, "overlap_mask_area_threshold": 0.8,
            "image_size": [640, 480], "resize": False, "batch_size": 2,
            "optimizer_steps": 0, "device": args.device, "mode": "infer" if args.execute else "plan_only"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0

    import torch
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    torch.manual_seed(42)
    torch.set_num_threads(4)
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_path, local_files_only=True, do_resize=False)
    model, loading = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_path, local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"checkpoint did not load exactly: {loading}")
    if model.config.id2label != {c["id"]: c["name"] for c in CLASSES}:
        raise ValueError("checkpoint label mapping mismatch")
    model.requires_grad_(False).eval().to(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (args.output / "loading_info.json").write_text(json.dumps(loading, indent=2) + "\n")
    started = time.monotonic()
    details = []
    with torch.inference_mode():
        for start in range(0, len(spec["samples"]), 2):
            rows = spec["samples"][start:start + 2]
            images = []
            for row in rows:
                with Image.open(args.manifest.parent / row["image"]) as im:
                    images.append(np.array(im))
            batch = processor(images=images, do_resize=False, return_tensors="pt").to(args.device)
            output = model(**batch)
            results = processor.post_process_instance_segmentation(
                output, threshold=0.5, mask_threshold=0.5, overlap_mask_area_threshold=0.8,
                target_sizes=[(480, 640)] * len(rows))
            for row, result in zip(rows, results, strict=True):
                png, segments = instance_png(result["segmentation"].cpu().numpy(), result["segments_info"])
                name = row["id"] + ".png"
                Image.fromarray(png).save(args.output / name)
                details.append({**row, "instance_map": name, "segments": segments})
    report = {"plan": plan, "samples": details, "elapsed_seconds": time.monotonic() - started,
              "optimizer_steps": 0, "complete": True, "independent_test": False,
              "physical_grasp_validated": False}
    (args.output / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "samples": len(details), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
