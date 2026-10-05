"""竹条实例分割：preflight只核对数据；train在明确指定的设备上训练，无硬件连接。"""

import argparse
import importlib.metadata
import json
from pathlib import Path

from lerobot.bamboo_sorting.strip_segmentation_data import CLASSES, audit_dataset, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, help="Local Mask2Former pretrained directory; no implicit download")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.eval_every) < 1 or min(args.lr, args.backbone_lr) <= 0:
        parser.error("epochs, batch size, eval interval and learning rates must be positive")
    audit = audit_dataset(args.dataset)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    if args.stage == "preflight":
        return 0
    if args.base_path is None or not args.base_path.is_dir() or args.output is None:
        parser.error("train requires a local --base-path and new --output")
    if args.output.exists():
        parser.error("output already exists; this entrypoint does not resume implicitly")
    for path in (args.dataset, args.base_path):
        if args.output.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(args.output.resolve()):
            parser.error("output must be separate from data and base model")
    run(args, audit)
    return 0


def run(args, audit):
    import functools
    import random
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    from lerobot.bamboo_sorting.strip_segmentation_training import (
        StripDataset, collate_samples, evaluate, forward_loss, move_batch,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = Mask2FormerImageProcessor.from_pretrained(str(args.base_path), local_files_only=True,
                                                         do_resize=False)
    id2label = {c["id"]: c["name"] for c in CLASSES}
    model, loading = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(args.base_path), local_files_only=True, num_labels=2, id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True,
        output_loading_info=True)
    mismatched = set(loading["mismatched_keys"])
    # Transformers 4.57 returns names here. Class count also changes the
    # criterion's no-object class-weight buffer, not only the prediction head.
    class_dependent = {"class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"}
    if (loading["missing_keys"] or loading["unexpected_keys"]
            or loading["error_msgs"] or mismatched - class_dependent):
        raise ValueError(f"unexpected pretrained loading differences: {loading}")
    model.to(args.device)
    collate = functools.partial(collate_samples, processor=processor)
    train_loader = DataLoader(StripDataset(args.dataset, "train"), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(StripDataset(args.dataset, "validation"), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        (backbone if "pixel_level_module.encoder" in name else other).append(parameter)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.backbone_lr},
                                   {"params": other, "lr": args.lr}], weight_decay=0.05)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(classes=CLASSES, image_resize=False, optimizer="AdamW", weight_decay=0.05,
                best_selection="validation macro F1 at mask IoU 0.5; loss as tie breaker",
                expected_steps=len(train_loader) * args.epochs, augmentation="none for first baseline",
                environment={k: importlib.metadata.version(k) for k in
                             ("torch", "torchvision", "transformers", "scipy", "numpy", "Pillow")})
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "data_audit.json", audit)
    write_json(args.output / "loading_info.json", loading)
    dataset_info = read_json(args.dataset / "dataset.json")
    first_sample = dataset_info["samples"][0]
    contract = {"classes": CLASSES, "image_size": [first_sample["width"], first_sample["height"]], "resize": False,
                "strip_dimensions_mm": dataset_info["strip_dimensions_mm"],
                "task": "visible instance segmentation, top vs covered; not physical graspability"}

    def save(destination):
        model.save_pretrained(destination)
        processor.save_pretrained(destination)
        write_json(destination / "strip_contract.json", contract)

    best_key, best_epoch, step = None, None, 0
    started = time.monotonic()
    with (args.output / "metrics.jsonl").open("w", buffering=1) as log:
        def record(value):
            line = json.dumps(value, ensure_ascii=False)
            log.write(line + "\n")
            print(line, flush=True)

        # A baseline score prevents treating a decreasing training loss as success.
        baseline = evaluate(model, processor, val_loader, args.device)
        write_json(args.output / "baseline_validation.json", baseline)
        best_key = (baseline["macro_f1"], -baseline["loss"])
        best_epoch = 0
        save(args.output / "best")
        record({"epoch": 0, "validation": {k: v for k, v in baseline.items() if k != "samples"}})
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, images = 0.0, 0
            for cpu_batch in train_loader:
                batch = move_batch(cpu_batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = forward_loss(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                size = len(cpu_batch["samples"])
                total += loss.item() * size
                images += size
                if step % 10 == 0:
                    record({"epoch": epoch, "step": step, "batch_loss": loss.item(),
                            "elapsed_seconds": round(time.monotonic() - started, 1)})
            row = {"epoch": epoch, "step": step, "train_loss": total / images}
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                metrics = evaluate(model, processor, val_loader, args.device)
                write_json(args.output / "latest_validation.json", metrics)
                row["validation"] = {k: v for k, v in metrics.items() if k != "samples"}
                key = (metrics["macro_f1"], -metrics["loss"])
                if key > best_key:
                    best_key, best_epoch = key, epoch
                    save(args.output / "best")
                row["best_epoch"] = best_epoch
            record(row)
        save(args.output / "final")
    # Evaluate reloaded best weights, not the in-memory final epoch.
    del model, optimizer, backbone, other, parameter, batch, loss
    if args.device == "cuda":
        torch.cuda.empty_cache()
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.output / "best", local_files_only=True)
    model.to(args.device)
    metrics = evaluate(model, processor, val_loader, args.device, prediction_dir=args.output / "validation_masks")
    write_json(args.output / "best_validation.json", metrics)
    write_json(args.output / "complete.json", {"epochs": args.epochs, "steps": step, "best_epoch": best_epoch,
               "elapsed_seconds": time.monotonic() - started, "best_reloaded_and_evaluated": True,
               "physical_grasp_validated": False})


if __name__ == "__main__":
    raise SystemExit(main())
