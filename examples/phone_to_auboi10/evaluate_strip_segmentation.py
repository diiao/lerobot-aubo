"""Evaluate a frozen local best checkpoint on independent test images; never optimize or save weights."""

import argparse
import functools
import importlib.metadata
import json
import time
from pathlib import Path

from lerobot.bamboo_sorting.strip_segmentation_data import CLASSES, audit_dataset, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True,
                        help="Original training/validation dataset.json for group-isolation check")
    parser.add_argument("--model-path", type=Path, required=True, help="Completed training run's best/ directory")
    parser.add_argument("--output", type=Path, required=True, help="New independent result directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    audit = audit_dataset(args.dataset, test_only=True)
    data = read_json(args.dataset / "dataset.json")
    training = read_json(args.training_manifest)
    groups = {s["placement_id"] for s in data["samples"]}
    overlap = groups & {s["placement_id"] for s in training["samples"]}
    if overlap:
        raise ValueError(f"test groups overlap training/validation: {sorted(overlap)}")
    if data["excluded_frames"]:
        raise ValueError("review test exclusions explicitly before running evaluation")
    contract = read_json(args.model_path / "strip_contract.json")
    completed = read_json(args.model_path.parent / "complete.json")
    if args.model_path.name != "best" or not completed["best_reloaded_and_evaluated"]:
        raise ValueError("use the completed training run's frozen best checkpoint")
    if (contract["classes"] != CLASSES or contract["resize"] is not False
            or contract["strip_dimensions_mm"] != data["strip_dimensions_mm"]
            or any([s["width"], s["height"]] != contract["image_size"] for s in data["samples"])):
        raise ValueError("test data does not match model contract")
    if args.output.exists():
        raise FileExistsError(args.output)
    for source in (args.dataset, args.model_path.parent, args.training_manifest.parent):
        if args.output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("result directory must be separate from source data and training run")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    from lerobot.bamboo_sorting.strip_segmentation_training import StripDataset, collate_samples, evaluate

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_path, local_files_only=True, do_resize=False)
    model, loading = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_path, local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"checkpoint did not load exactly: {loading}")
    if model.config.id2label != {c["id"]: c["name"] for c in CLASSES}:
        raise ValueError("checkpoint class mapping differs")
    model.requires_grad_(False)
    model.to(args.device)
    loader = DataLoader(StripDataset(args.dataset, "test"), batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=functools.partial(collate_samples, processor=processor))
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(split="test", best_epoch=completed["best_epoch"], groups=sorted(groups),
                score_threshold=0.5, mask_threshold=0.5, matching_mask_iou=0.5, batch_size=2,
                overlap_mask_area_threshold=0.8, image_resize=False, optimizer_steps=0,
                class_mapping=CLASSES, model_contract=contract, seed=42,
                environment={k: importlib.metadata.version(k) for k in
                             ("torch", "torchvision", "transformers", "scipy", "numpy", "Pillow")})
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "data_audit.json", audit)
    write_json(args.output / "loading_info.json", loading)
    started = time.monotonic()
    metrics = evaluate(model, processor, loader, args.device, score_threshold=0.5,
                       prediction_dir=args.output / "test_masks")
    metrics.update(split="test", best_epoch=completed["best_epoch"], optimizer_steps=0)
    write_json(args.output / "test_evaluation.json", metrics)
    write_json(args.output / "complete.json", {"best_epoch": completed["best_epoch"],
               "best_reloaded_and_evaluated": True, "split": "test", "images": metrics["images"],
               "optimizer_steps": 0, "elapsed_seconds": time.monotonic() - started,
               "training_group_overlap": sorted(overlap), "physical_grasp_validated": False})
    print(json.dumps({k: v for k, v in metrics.items() if k != "samples"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
