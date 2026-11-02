#!/usr/bin/env python3
"""AUBO PI05 preflight and zero-update GPU smoke test. No training or hardware."""

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    required = ("data_manifest", "prepared_data", "tokenizer_path") if args.stage == "preflight" else ("preflight", "base_path")
    if any(getattr(args, key) is None for key in required):
        parser.error(f"{args.stage} requires " + ", ".join("--" + k.replace("_", "-") for k in required))
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("output must be a NEW directory")
    for key in required:
        source = getattr(args, key).resolve()
        if not source.exists():
            parser.error(f"missing input: {source}")
        if source == args.output or source in args.output.parents or args.output in source.parents:
            parser.error("output must be independent of inputs")
    return args


def preflight(args):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import default_collate
    from lerobot.bamboo_sorting.policy.joint_training import read_prepared_training_data, load_prepared_dataset, write_json
    from lerobot.bamboo_sorting.policy.pi05_joint_adapter import (
        train_quantile_stats, make_public_paligemma_tokenizer, make_joint_pi05_config, make_joint_pi05_processors)
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

    prepared = read_prepared_training_data(args.prepared_data, args.data_manifest, target_conditioned=True)
    # Protect source directories even if the output was not nested in the manifest file itself.
    for split in ("train", "validation"):
        for source in prepared[split].get("sources", [prepared[split]]):
            for key in ("root", "evidence_root", "target_annotations"):
                path = Path(source[key]).resolve()
                if path == args.output or path in args.output.parents or args.output in path.parents:
                    raise ValueError("output overlaps original data/evidence")
    args.output.mkdir(parents=True)
    stats = train_quantile_stats(prepared)
    write_json(args.output / "train_quantile_stats.json", stats)
    tokenizer_report = make_public_paligemma_tokenizer(args.tokenizer_path)
    write_json(args.output / "tokenizer_check.json", tokenizer_report)
    config = make_joint_pi05_config()
    pre, post = make_joint_pi05_processors(config, stats, args.tokenizer_path)
    # Cover three-strip, independent four-strip, overlapping four-strip, and held-out three-strip.
    train = prepared["train"].get("sources", [prepared["train"]])
    validations = prepared["validation"].get("sources", [prepared["validation"]])
    choices = [("train", index, train[index]) for index in sorted({0, min(9, len(train)-1), len(train)-1})]
    choices.append(("validation", 0, validations[0]))
    sheet = Image.new("RGB", (224 * 4, 252 * len(choices)), "white")
    draw = ImageDraw.Draw(sheet)
    sample_report, smoke_samples = [], []
    for row, (split, source_index, source) in enumerate(choices):
        dataset = load_prepared_dataset(source)
        episode = source["episodes"][0]
        selected = [(i, f) for i, f in enumerate(source["target_frames"]) if f["episode_index"] == episode]
        approach = [(i, f) for i, f in selected if not f["transport"]]
        transport = [(i, f) for i, f in selected if f["transport"]]
        if not approach or not transport:
            raise ValueError("review sample must cover approach and transport")
        indices = [approach[0][0], approach[-1][0], transport[0][0], selected[-1][0]]
        raw_samples = [dataset[index] for index in indices]
        counts = []
        for column, (index, raw) in enumerate(zip(indices, raw_samples, strict=True)):
            spec = source["target_frames"][index]
            if (int(raw["episode_index"]), int(raw["frame_index"])) != (spec["episode_index"], spec["frame_index"]):
                raise ValueError("sample identity changed")
            raw_global = dataset.dataset[index]["observation.images.global_rgb"]
            # Match the policy's normalization-before-padding order, then display as RGB.
            image = (resize_with_pad_torch(raw["observation.images.target_global_rgb"] * 2 - 1, 224, 224)[0] + 1) / 2
            original = (resize_with_pad_torch(raw_global * 2 - 1, 224, 224)[0] + 1) / 2
            difference = (image - original).abs().amax(0)
            changed = int((difference > 0.05).sum())
            if not spec["transport"] and changed == 0:
                raise ValueError("target outline disappears at model resolution")
            if spec["transport"] and not torch.equal(image, original):
                raise ValueError("transport unexpectedly contains target overlay")
            pixels = image.permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
            sheet.paste(Image.fromarray(pixels), (column * 224, row * 252 + 28))
            label = ("first", "before close", "transport", "tail")[column]
            draw.text((column * 224 + 3, row * 252 + 3), f"{split}/{source['scenes'][0]} {label}", fill="black")
            counts.append({"frame_index": int(raw["frame_index"]), "phase": "transport" if spec["transport"] else "approach",
                           "changed_pixels_above_0_05": changed, "max_pixel_difference": float(difference.max()),
                           "padded_action_rows": int(raw["action_is_pad"].sum())})
        # A mixed two-sample batch uses an approach observation from each strip count.
        if split == "train" and source_index in (0, len(train)-1):
            smoke_samples.append(raw_samples[0])
        sample_report.append({"split": split, "source_index": source_index, "root": source["root"],
                              "episode_index": episode, "samples": counts})
    sheet.save(args.output / "input_224_review.png")
    raw = default_collate(smoke_samples)
    torch.save(raw, args.output / "sample_batch.pt")
    batch = pre(raw)
    roundtrip_error = float((post(batch["action"]) - raw["action"]).abs().max())
    if not np.isfinite(roundtrip_error) or roundtrip_error > 1e-4:
        raise ValueError("absolute action normalization/inverse normalization mismatch")
    pre.save_pretrained(args.output)
    post.save_pretrained(args.output)
    report = {"stage": "preflight", "optimizer_steps": 0, "physical_grasp_verified": False,
              "data_manifest": str(args.data_manifest.resolve()), "prepared_data": str(args.prepared_data.resolve()),
              "tokenizer_path": str(args.tokenizer_path.resolve()), "contract": prepared["contract"],
              "normalization_scope": "selected_successful_train_episodes_only", "quantile_method": "numpy linear, q01/q99",
              "split_counts": {split: {"sources": len(prepared[split].get("sources", [prepared[split]])),
                                       "episodes": sum(len(s["episodes"]) for s in prepared[split].get("sources", [prepared[split]])),
                                       "frames": prepared[split]["frames"], "scenes": prepared[split]["scenes"]}
                               for split in ("train", "validation")},
              "validation_limitation": "existing three-strip piles only; no independent four-strip validation",
              "image_review": sample_report, "sample_batch_size": len(smoke_samples),
              "action_roundtrip_max_error": roundtrip_error,
              "language_token_shape": list(batch["observation.language.tokens"].shape),
              "tokenizer": tokenizer_report}
    write_json(args.output / "result.json", report)
    print(json.dumps({k: report[k] for k in ("stage", "sample_batch_size", "action_roundtrip_max_error", "language_token_shape")}), flush=True)


def smoke(args):
    import gc
    import importlib.metadata
    import shutil
    import time
    import torch
    from lerobot.bamboo_sorting.policy.joint_training import write_json
    from lerobot.bamboo_sorting.policy.pi05_joint_adapter import (
        make_joint_pi05_config, make_joint_pi05_processors, joint_pi05_loss, load_joint_pi05_weights)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action

    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 9 * 2**30:
        raise RuntimeError("at least 9 GiB free required for a new smoke checkpoint")
    provenance = json.loads((args.preflight / "result.json").read_text())
    stats = json.loads((args.preflight / "train_quantile_stats.json").read_text())
    raw = torch.load(args.preflight / "sample_batch.pt", map_location="cpu", weights_only=True)
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    config = make_joint_pi05_config(device="cuda")
    pre, post = make_joint_pi05_processors(config, stats, provenance["tokenizer_path"])
    print("loading pinned base weights strictly", flush=True)
    policy = PI05Policy(config)
    load_joint_pi05_weights(policy, args.base_path / "model.safetensors")
    batch = pre(raw)
    policy.train()
    frozen = list(policy.model.paligemma_with_expert.paligemma.parameters())
    if any(p.requires_grad for p in frozen):
        raise ValueError("vision-language backbone is not frozen")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    loss = joint_pi05_loss(policy, batch)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite smoke loss")
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    gradients = [p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("missing or nonfinite action-expert gradients")
    if any(p.grad is not None for p in frozen):
        raise ValueError("frozen backbone received gradients")
    grad_norm = float(torch.stack([torch.linalg.vector_norm(g.float()) for g in gradients]).norm())
    if grad_norm == 0:
        raise ValueError("zero action-expert gradient")
    result = {"stage": "smoke", "optimizer_steps": 0, "batch_size": len(raw["action"]),
              "loss": float(loss.detach()), "gradient_norm": grad_norm,
              "forward_backward_seconds": elapsed, "peak_allocated_GiB": torch.cuda.max_memory_allocated()/2**30,
              "peak_reserved_GiB": torch.cuda.max_memory_reserved()/2**30,
              "total_parameters": sum(p.numel() for p in policy.parameters()),
              "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
              "parameters_receiving_gradient": sum(g.numel() for g in gradients),
              "frozen_backbone_has_gradients": False, "strict_base_load": True,
              "preflight": str(args.preflight.resolve()), "seed": args.seed,
              "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "scipy", "lerobot", "safetensors", "sentencepiece")},
              "physical_grasp_verified": False, "policy_execution_authorized": False}
    write_json(args.output / "forward_backward.json", result)
    policy.zero_grad(set_to_none=True)
    del loss, gradients, frozen
    observations = {key: value for key, value in raw.items() if key.startswith("observation.") or key == "task"}
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        before = post(policy.predict_action_chunk(pre(observations))).cpu()
    if before.shape != (len(raw["action"]), 50, 7) or not torch.isfinite(before).all():
        raise ValueError("prediction must be finite [batch, 50, 7]")
    checkpoint = args.output / "checkpoint"
    checkpoint.mkdir()
    # Include the small tokenizer in the checkpoint and store an explicit local path.
    shutil.copytree(provenance["tokenizer_path"], checkpoint / "tokenizer")
    saved_pre, saved_post = make_joint_pi05_processors(config, stats, checkpoint / "tokenizer")
    saved_pre.save_pretrained(checkpoint)
    saved_post.save_pretrained(checkpoint)
    policy.to("cpu")
    policy.save_pretrained(checkpoint)
    write_json(checkpoint / "aubo_joint_contract.json", provenance["contract"])
    write_json(checkpoint / "training_progress.json", {"optimizer_steps": 0, "preflight": str(args.preflight.resolve()),
                                                       "physical_grasp_verified": False})
    del policy, batch, pre, post, saved_pre, saved_post
    gc.collect()
    torch.cuda.empty_cache()
    print("strict checkpoint reload and same-seed prediction comparison", flush=True)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    policy = PI05Policy(config)
    load_joint_pi05_weights(policy, checkpoint / "model.safetensors")
    pre = PolicyProcessorPipeline.from_pretrained(checkpoint, "policy_preprocessor.json", local_files_only=True,
             to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(checkpoint, "policy_postprocessor.json", local_files_only=True,
             to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        after = post(policy.predict_action_chunk(pre(observations))).cpu()
    if not torch.equal(before, after):
        raise ValueError(f"checkpoint reload prediction mismatch: {float((before-after).abs().max())}")
    result.update(strict_reload=True, prediction_shape=list(after.shape), reload_max_abs_difference=0.0,
                  checkpoint_bytes=(checkpoint / "model.safetensors").stat().st_size,
                  free_disk_GiB=shutil.disk_usage(args.output).free/2**30)
    torch.save({"before": before, "after": after}, args.output / "reload_predictions.pt")
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


def main(argv=None):
    args = parse_args(argv)
    import torch
    torch.set_num_threads(4)
    if args.stage == "preflight":
        preflight(args)
    else:
        smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
