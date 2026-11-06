#!/usr/bin/env python
"""Full-joint SmolVLA: plan / preflight / synthetic smoke / train / evaluate.

Default plan is hardware-free, model-free and writes nothing. All model use is
offline. Smoke uses synthetic data and ZERO optimizer steps, never real capture.
"""

import argparse
from functools import partial
import json
import math
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "preflight", "smoke", "train", "evaluate"), default="plan")
    parser.add_argument("--data-manifest", type=Path, help="Explicit training and validation sources")
    parser.add_argument("--prepared-data", type=Path,
                        help="Reuse an audited data_preflight.json with the same manifest; no repeated full hashing")
    parser.add_argument("--target-conditioned", action="store_true",
                        help="Approach-only target outline; every source needs reviewed target_annotations")
    parser.add_argument("--target-outline-width-px", type=int, choices=(1, 2), default=2,
                        help="Offline target input candidate; checkpoint records its explicit width (default 2)")
    for name in ("train-root", "train-evidence", "validation-root", "validation-evidence",
                 "train-source-root", "validation-source-root", "base-path", "vlm-path", "output", "checkpoint"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--learning-rate", type=float, help="Override optimizer peak learning rate")
    parser.add_argument("--local-demo-fraction", type=float,
                        help="Expected sampling fraction for local_alignment (e.g. 0.5); remaining mass goes to full demos")
    parser.add_argument("--normalization-source", choices=("train_data", "base_checkpoint"), default="train_data",
                        help="base_checkpoint preserves saved AUBO processors for continuation; pooled new stats remain diagnostic")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[],
                        help="Save intermediate weights at these training updates; final is always saved")
    parser.add_argument("--validation-every", type=int, default=0,
                        help="Fixed held-out checks every N updates; retain verified best and final only (0 disables)")
    parser.add_argument("--validation-points-per-phase", type=int, default=8)
    parser.add_argument("--validation-seeds", type=int, nargs="+",
                        help="Fixed paired flow-noise seeds; default is seed, seed+1, seed+2")
    parser.add_argument("--resume-training-state", type=Path,
                        help="Explicit optimizer/scheduler continuation into a NEW output; steps are additional updates")
    args = parser.parse_args(argv)
    if args.target_outline_width_px != 2 and not args.target_conditioned:
        parser.error("target-outline-width-px requires --target-conditioned")
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("learning-rate must be finite and positive")
    if args.local_demo_fraction is not None:
        if not math.isfinite(args.local_demo_fraction) or not 0 < args.local_demo_fraction < 1:
            parser.error("local-demo-fraction must be finite and strictly between 0 and 1")
        if args.data_manifest is None or args.stage not in ("plan", "preflight", "train"):
            parser.error("local-demo-fraction requires a manifest and plan/preflight/train")
    if args.normalization_source == "base_checkpoint" and args.stage not in ("plan", "preflight", "train"):
        parser.error("base_checkpoint normalization requires plan/preflight/train; evaluation uses its saved processors")
    if (len(set(args.checkpoint_steps)) != len(args.checkpoint_steps)
            or any(step < 1 or step >= args.steps for step in args.checkpoint_steps)):
        parser.error("checkpoint-steps must be unique positive updates before the final step")
    if args.checkpoint_steps and args.stage not in ("plan", "train"):
        parser.error("checkpoint-steps requires plan/train")
    if args.validation_every < 0 or args.validation_points_per_phase < 2:
        parser.error("validation-every must be nonnegative and validation-points-per-phase at least 2")
    if args.validation_seeds is None:
        args.validation_seeds = [args.seed, args.seed+1, args.seed+2]
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        parser.error("validation-seeds must be distinct")
    if args.validation_every:
        if args.stage not in ("plan", "train") or not args.target_conditioned or not args.data_manifest:
            parser.error("periodic selection requires target-conditioned manifest plan/train")
        if args.checkpoint_steps:
            parser.error("periodic best/final retention cannot also retain explicit checkpoint-steps")
    if args.resume_training_state is not None:
        if (args.stage not in ("plan", "train") or not args.validation_every
                or args.normalization_source != "base_checkpoint" or args.base_path is None):
            parser.error("resume-training-state requires periodic manifest plan/train and base_checkpoint normalization")
        if args.learning_rate is not None:
            parser.error("resume-training-state preserves the saved learning rate; do not override it")
    if args.prepared_data is not None and (args.data_manifest is None or args.stage not in ("plan", "train", "evaluate")):
        parser.error("prepared-data requires a manifest and plan/train/evaluate")
    if args.target_conditioned and args.stage in ("preflight", "train", "evaluate") and args.data_manifest is None:
        parser.error("--target-conditioned requires --data-manifest")
    if args.data_manifest and any(getattr(args, name) is not None for name in
            ("train_root", "train_evidence", "validation_root", "validation_evidence",
             "train_source_root", "validation_source_root")):
        parser.error("use either --data-manifest or individual data paths")
    if args.stage in ("preflight", "train", "evaluate") and args.data_manifest is None:
        for name in ("train_root", "train_evidence", "validation_root", "validation_evidence"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
    if args.stage in ("smoke", "train") and (args.base_path is None or args.vlm_path is None):
        parser.error("local --base-path and --vlm-path are required")
    if args.stage == "evaluate" and args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.stage != "plan":
        if args.output is None or args.output.exists():
            parser.error("a NEW --output directory is required; never overwrite/resume implicitly")
        output = args.output.resolve()
        if args.data_manifest is not None:
            from lerobot.bamboo_sorting.policy.joint_training import read_data_manifest
            sources = read_data_manifest(args.data_manifest)
            validations = sources["validation"]
            validations = validations if isinstance(validations, list) else [validations]
            protected = [args.data_manifest.resolve(), *[Path(item[key])
                for item in [*sources["train"], *validations] for key in ("root", "evidence_root")]]
            if any(output == p or p in output.parents or output in p.parents for p in protected):
                parser.error("output must be independent of manifest and source roots")
        for path in (args.train_root, args.validation_root, args.train_evidence, args.validation_evidence, args.prepared_data,
                     args.base_path, args.vlm_path, args.checkpoint, args.resume_training_state):
            if path is not None and (output == path.resolve() or path.resolve() in output.parents
                                     or output in path.resolve().parents):
                parser.error("output must be independent of input roots")
    return args


def read_continuation_state(args, prepared):
    """Restore explicit optimizer progress; do not claim to restore data/RNG position."""
    import torch

    path = args.resume_training_state
    state = torch.load(path, map_location="cpu", weights_only=True)
    progress = json.loads((args.base_path / "training_progress.json").read_text())
    step = state["step"]
    if type(step) is not int or step < 1 or progress["optimizer_step"] != step:
        raise ValueError("optimizer state step differs from base checkpoint")
    if json.loads((args.base_path / "data_preflight.json").read_text()) != prepared:
        raise ValueError("continuation data/split/statistics differ from base checkpoint")
    source_plan = json.loads((path.parent / "plan.json").read_text())
    for key in ("batch_size", "seed", "local_demo_fraction", "target_outline_width_px"):
        if source_plan.get(key) != getattr(args, key):
            raise ValueError(f"continuation training setting differs: {key}")
    source_pointer = json.loads((path.parent / "best_checkpoint.json").read_text())
    if (Path(source_pointer["path"]).resolve() != args.base_path.resolve()
            or source_pointer["optimizer_steps"] != step):
        raise ValueError("continuation requires the best checkpoint at the saved optimizer step")
    if state["scheduler"]["last_epoch"] != step:
        raise ValueError("scheduler step differs from optimizer step")
    groups = state["optimizer"]["param_groups"]
    if (not groups or len(groups) != len(state["scheduler"]["_last_lr"])
            or any(g["lr"] != lr or not math.isfinite(lr) or lr <= 0
                   for g, lr in zip(groups, state["scheduler"]["_last_lr"], strict=True))):
        raise ValueError("optimizer/scheduler learning rates differ")
    # Legacy training_state.pt was written only after its full optimization budget.
    state["schedule_steps"] = state.get("schedule_steps", step)
    if type(state["schedule_steps"]) is not int or not 1 <= state["schedule_steps"] <= step:
        raise ValueError("invalid original schedule horizon")
    return state


def training_lr_scale(step, schedule_steps):
    """The original cosine schedule, clamped at its existing terminal rate."""
    warmup = min(1000, max(1, schedule_steps // 10))
    if step < warmup:
        return (step + 1) / warmup
    return .025 + .975 * (1 + math.cos(math.pi * min(
        (step - warmup) / max(1, schedule_steps - warmup), 1))) / 2


def main(argv=None):
    args = parse_args(argv)
    plan = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(fixed_joint_targets_deg={}, state_action_names=["J1", "J2", "J3", "J4", "J5", "J6", "gripper_pos"],
                chunk_size=50, n_action_steps=1, smoke_optimizer_steps=0, policy_execution_authorized=False)
    if args.validation_every:
        from lerobot.bamboo_sorting.policy.joint_checkpoint_selection import SELECTION_RULE
        plan.update(selection_rule=SELECTION_RULE, checkpoint_retention="verified_best_and_final",
                    automatic_checkpoint_deletion="superseded best owned by this output only", early_stopping=False)
    if args.resume_training_state is not None:
        plan.update(step_budget="additional_updates", selection_baseline=str(args.base_path.resolve()),
                    optimizer_resume=True, scheduler_resume=True, data_rng_position_restored=False,
                    learning_rate_policy="preserve original schedule horizon and terminal rate")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.stage == "plan":
        return 0
    from lerobot.bamboo_sorting.policy.joint_training import (
        prepare_training_data, prepare_training_manifest, read_prepared_training_data, training_sampling_plan, write_json,
    )
    prepared = None
    if args.stage != "smoke":
        if args.prepared_data:
            prepared = read_prepared_training_data(args.prepared_data, args.data_manifest,
                                                  target_conditioned=args.target_conditioned,
                                                  target_outline_width_px=args.target_outline_width_px)
        elif args.data_manifest:
            prepared = prepare_training_manifest(args.data_manifest, target_conditioned=args.target_conditioned)
        else:
            prepared = prepare_training_data(args.train_root, args.train_evidence,
                args.validation_root, args.validation_evidence,
                train_source_root=args.train_source_root, validation_source_root=args.validation_source_root)
    sampling = training_sampling_plan(prepared["train"], args.local_demo_fraction) \
        if args.stage in ("preflight", "train") else None
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    if sampling is not None:
        write_json(args.output / "training_sampling.json", {**sampling, "seed": args.seed})
    if prepared is not None:
        if args.target_conditioned:
            from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
            prepared = {**prepared, "contract": target_policy_contract_record(outline_width_px=args.target_outline_width_px)}
        write_json(args.output / "data_preflight.json", prepared)
    if args.stage == "preflight":
        return 0
    # Set offline behavior only in this explicitly invoked model subprocess.
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    run_model(args, prepared)
    return 0


def run_model(args, prepared):
    import gc
    import importlib.metadata
    import math
    import random
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, JOINT_IMAGE_KEYS, JOINT_TASK, joint_contract_record
    from lerobot.bamboo_sorting.policy.joint_training import prediction_metrics, write_json, make_training_sampler
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import (
        make_joint_smolvla_config, make_joint_smolvla_processors, load_joint_smolvla_processors,
    )
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import SmolVLAJointOfflineAdapter
    if args.target_conditioned:
        from lerobot.bamboo_sorting.policy.joint_target_policy import (
            SmolVLAJointTargetOfflineAdapter, SmolVLAJointThinTargetOfflineAdapter, target_policy_contract_record,
        )
        contract_record = partial(target_policy_contract_record, outline_width_px=args.target_outline_width_px)
        adapter_class = (SmolVLAJointThinTargetOfflineAdapter if args.target_outline_width_px == 1
                         else SmolVLAJointTargetOfflineAdapter)
        image_keys = adapter_class.image_keys
    else:
        contract_record, adapter_class, image_keys = joint_contract_record, SmolVLAJointOfflineAdapter, JOINT_IMAGE_KEYS
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    write_json(args.output / "environment.json", {key: importlib.metadata.version(key)
        for key in ("torch", "transformers", "lerobot", "safetensors", "accelerate")})
    images = [f"observation.images.{name}" for name in image_keys]

    def load_checkpoint(path, *, device=None):
        if json.loads((path / "aubo_joint_contract.json").read_text()) != contract_record():
            raise ValueError("checkpoint is not full-joint AUBO")
        cfg = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        device = args.device if device is None else device
        cfg.device = device
        model = SmolVLAPolicy.from_pretrained(path, config=cfg, local_files_only=True, strict=True)
        pre = PolicyProcessorPipeline.from_pretrained(path, "policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(path, "policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        adapter_class(model, pre, post, contract=contract_record())
        return model, pre, post

    def dataset(which):
        from lerobot.bamboo_sorting.policy.joint_training import load_prepared_dataset
        return load_prepared_dataset(prepared[which], video_backend=args.video_backend,
            target_outline_width_px=args.target_outline_width_px)

    def loader(ds, shuffle=False, sampler=None):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=0,
                          generator=torch.Generator().manual_seed(args.seed))

    def observation(raw):
        return {k: raw[k] for k in [*images, "observation.state", "task"]}

    def save_checkpoint(path, step):
        path.parent.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(path)
        pre.save_pretrained(path)
        post.save_pretrained(path)
        write_json(path / "aubo_joint_contract.json", contract_record())
        if prepared is not None:
            write_json(path / "data_preflight.json", prepared)
        write_json(path / "normalization.json", normalization)
        if args.stage == "train":
            write_json(path / "training_sampling.json", json.loads((args.output / "training_sampling.json").read_text()))
        write_json(path / "training_progress.json", {"optimizer_step": step,
            "implicit_resume_supported": False, "policy_execution_authorized": False})

    best_path = best_metrics = monitor_plan = validation_dataset = None

    def verify_saved_best(path):
        from lerobot.bamboo_sorting.policy.joint_checkpoint_selection import isolated_evaluation
        # A second GPU policy is unnecessary for checking saved tensors.
        with isolated_evaluation(policy):
            restored, restored_pre, restored_post = load_checkpoint(path, device="cpu")
            original_state, restored_state = policy.state_dict(), restored.state_dict()
            if original_state.keys() != restored_state.keys():
                raise ValueError("saved best state dictionary keys differ")
            for key, value in original_state.items():
                saved = restored_state[key]
                if value.dtype != saved.dtype or not torch.equal(value.detach().cpu(), saved.detach().cpu()):
                    raise ValueError(f"saved best tensor differs: {key}")
            for original_pipeline, restored_pipeline in ((pre, restored_pre), (post, restored_post)):
                for original_step, restored_step in zip(original_pipeline.steps, restored_pipeline.steps, strict=True):
                    left, right = original_step.state_dict(), restored_step.state_dict()
                    if type(original_step) != type(restored_step) or left.keys() != right.keys():
                        raise ValueError("saved best processor structure differs")
                    if any(left[k].dtype != right[k].dtype or not torch.equal(left[k].cpu(), right[k].cpu()) for k in left):
                        raise ValueError("saved best processor statistics differ")
            del restored_state, original_state, restored, restored_pre, restored_post
            gc.collect()
        return {"strict_reload": True, "state_dict_equal": True, "processors_state_equal": True, "device": "cpu"}

    if args.stage == "evaluate":
        provenance = json.loads((args.checkpoint / "data_preflight.json").read_text())
        if provenance != prepared:
            raise ValueError("evaluation datasets differ from checkpoint provenance")
        policy, pre, post = load_checkpoint(args.checkpoint)
    else:
        if not args.base_path.is_dir() or not args.vlm_path.is_dir():
            raise ValueError("base model and VLM must be existing LOCAL directories")
        cfg = PreTrainedConfig.from_pretrained(args.base_path, local_files_only=True)
        if cfg.type != "smolvla":
            raise ValueError("base model must be SmolVLA")
        joint_cfg = make_joint_smolvla_config(device=args.device, target_conditioned=args.target_conditioned)
        if args.normalization_source == "base_checkpoint" and any(
                getattr(cfg, name) != getattr(joint_cfg, name) for name in (
                    "input_features", "output_features", "normalization_mapping", "action_representation",
                    "adapt_to_pi_aloha", "use_delta_joint_actions_aloha")):
            raise ValueError("base checkpoint must already use the requested absolute 7D AUBO configuration")
        for name in ("input_features", "output_features", "normalization_mapping", "device", "chunk_size",
                     "n_action_steps", "num_steps", "action_representation", "relative_action_stats",
                     "execution_loss_fraction", "adapt_to_pi_aloha", "use_delta_joint_actions_aloha", "push_to_hub"):
            setattr(cfg, name, getattr(joint_cfg, name))
        cfg.load_vlm_weights = False
        cfg.vlm_model_name = str(args.vlm_path.resolve())
        if args.learning_rate is not None:
            cfg.optimizer_lr = args.learning_rate
        if args.stage == "smoke":
            stats = {key: {"mean": [0] * GRIPPER_INDEX + [50], "std": [1] * GRIPPER_INDEX + [50],
                           "min": [-1] * GRIPPER_INDEX + [0], "max": [1] * GRIPPER_INDEX + [100]}
                     for key in ("observation.state", "action")}
            raw = {"observation.state": torch.zeros(1, JOINT_DIM), "action": torch.zeros(1, 50, JOINT_DIM),
                   "action_is_pad": torch.zeros(1, 50, dtype=torch.bool), "task": [JOINT_TASK],
                   **{key: torch.zeros(1, 3, 480, 640) for key in images}}
            raw["action"][:, 1:25, GRIPPER_INDEX] = 100
        else:
            stats = prepared["train_only_stats"]
        # Preserve the new train-only measurements even when they are not the
        # normalization actually used by the continued policy.
        write_json(args.output / "train_only_stats.json", stats)
        if args.normalization_source == "base_checkpoint":
            pre, post, stats = load_joint_smolvla_processors(args.base_path, cfg, contract=contract_record())
        else:
            pre, post = make_joint_smolvla_processors(cfg, stats, target_conditioned=args.target_conditioned)
        normalization = {"source": args.normalization_source, "stats": stats,
                         "base_checkpoint": str(args.base_path.resolve()) if args.normalization_source == "base_checkpoint" else None}
        write_json(args.output / "normalization.json", normalization)
        policy = SmolVLAPolicy.from_pretrained(args.base_path, config=cfg, local_files_only=True, strict=True)
        params = [p for p in policy.parameters() if p.requires_grad]
        policy.train()
        if args.stage == "smoke":
            loss, _ = policy(pre(raw))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite smoke loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 10, error_if_nonfinite=True)
            suction_grad = float(policy.model.action_out_proj.weight.grad[GRIPPER_INDEX].norm())
            if suction_grad <= 0:
                raise ValueError("suction output receives no gradient")
            write_json(args.output / "smoke_backward.json", {"loss": float(loss.detach()),
                "gradient_norm": float(norm), "suction_head_gradient_norm": suction_grad,
                "synthetic_data": True, "optimizer_steps": 0})
            policy.zero_grad(set_to_none=True)
            del loss
        else:
            continuation = read_continuation_state(args, prepared) if args.resume_training_state else None
            start_step = continuation["step"] if continuation else 0
            final_step = start_step + args.steps
            schedule_steps = continuation["schedule_steps"] if continuation else args.steps
            train_dataset = dataset("train")
            if args.validation_every:
                from lerobot.bamboo_sorting.policy.joint_checkpoint_selection import (
                    SELECTION_RULE, build_monitor_plan, evaluate_monitor, publish_best_checkpoint, selection_comparison,
                    load_selection_baseline,
                )
                validation_dataset = dataset("validation")
                monitor_plan = build_monitor_plan(prepared["validation"], points_per_phase=args.validation_points_per_phase)
                write_json(args.output / "validation_plan.json", {
                    **{k: v for k, v in monitor_plan.items() if k != "full_observations"},
                    "full_validation_frames": len(validation_dataset), "seeds": args.validation_seeds,
                    "rule": SELECTION_RULE, "validation_every": args.validation_every,
                    "final_step_always_checked": True, "early_stopping": False})
                if continuation:
                    best_metrics = load_selection_baseline(args.base_path, monitor_plan, args.validation_seeds)
                    best_path = args.base_path.resolve()
                    write_json(args.output / "best_checkpoint.json", dict(path=str(best_path),
                        optimizer_steps=start_step, primary=best_metrics["primary"],
                        primary_name=best_metrics["primary_name"], external_baseline=True))
            sampler = make_training_sampler(prepared["train"], args.local_demo_fraction, seed=args.seed)
            batches = loader(train_dataset, shuffle=sampler is None, sampler=sampler)
            optimizer = cfg.get_optimizer_preset().build(params)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(training_lr_scale, schedule_steps=schedule_steps))
            if continuation:
                optimizer.load_state_dict(continuation["optimizer"])
                scheduler.load_state_dict(continuation["scheduler"])
                write_json(args.output / "continuation.json", dict(base_checkpoint=str(args.base_path.resolve()),
                    training_state=str(args.resume_training_state.resolve()), start_step=start_step,
                    additional_steps=args.steps, final_step=final_step, schedule_steps=schedule_steps,
                    restored_learning_rates=[group["lr"] for group in optimizer.param_groups],
                    optimizer_resume=True, scheduler_resume=True, data_rng_position_restored=False,
                    selection_baseline_primary=best_metrics["primary"]))
                del continuation
            iterator, start = iter(batches), time.monotonic()
            with (args.output / "training.jsonl").open("x") as stream:
                for step in range(start_step + 1, final_step + 1):
                    try:
                        raw = next(iterator)
                    except StopIteration:
                        iterator = iter(batches)
                        raw = next(iterator)
                    if any(task != JOINT_TASK for task in raw["task"]):
                        raise ValueError("unexpected training instruction")
                    loss, _ = policy(pre(raw))
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    record = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                        "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.monotonic() - start}
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if step == start_step + 1 or step % 100 == 0:
                        print(f"step={step} loss={float(loss.detach()):.6f}", flush=True)
                    if step in args.checkpoint_steps:
                        save_checkpoint(args.output / "checkpoints" / f"step_{step:06d}", step)
                        print(f"checkpoint_saved step={step}", flush=True)
                    if args.validation_every and ((step - start_step) % args.validation_every == 0 or step == final_step):
                        metrics = evaluate_monitor(policy, pre, post, validation_dataset, monitor_plan["observations"],
                            batch_size=args.batch_size, seeds=args.validation_seeds)
                        comparison = selection_comparison(metrics, best_metrics)
                        destination = args.output / "validation" / f"step_{step:06d}"
                        destination.mkdir(parents=True, exist_ok=False)
                        write_json(destination / "metrics.json", metrics)
                        if comparison["improved"]:
                            metrics["guard_reference"] = {k: min(v, best_metrics.get("guard_reference", best_metrics["guards"])[k])
                                if best_metrics else v for k, v in metrics["guards"].items()}
                            owned_previous = best_path if best_path is not None and best_path.parent == args.output / "checkpoints" else None
                            best_path = publish_best_checkpoint(args.output, step, metrics, owned_previous,
                                                                save_checkpoint, verify_saved_best)
                            best_metrics = metrics
                        decision = dict(optimizer_step=step, primary=metrics["primary"],
                            primary_seed_std=metrics["primary_seed_std"], validation_loss=metrics["validation_loss"],
                            guards=metrics["guards"], comparison=comparison, best_checkpoint=str(best_path),
                            best_primary=best_metrics["primary"], training_continues=True)
                        write_json(destination / "decision.json", decision)
                        with (args.output / "validation_history.jsonl").open("a") as history:
                            history.write(json.dumps(decision, allow_nan=False) + "\n")
                        print(f"validation step={step} primary={metrics['primary']:.6f} "
                              f"new_best={comparison['improved']} blocked={comparison['blocked_guards']}", flush=True)
            torch.save({"step": final_step, "schedule_steps": schedule_steps,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                       args.output / "training_state.pt")
            del optimizer, scheduler, loss
        checkpoint = args.output / "final"
        save_checkpoint(checkpoint, final_step if args.stage == "train" else 0)
        torch.manual_seed(args.seed)
        before = post(policy.predict_action_chunk(pre(observation(raw)))).detach().cpu()
        del params, policy, pre, post
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        policy, pre, post = load_checkpoint(checkpoint)
        torch.manual_seed(args.seed)
        after = post(policy.predict_action_chunk(pre(observation(raw)))).detach().cpu()
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        if after.shape[-2:] != (50, JOINT_DIM) or not torch.isfinite(after).all():
            raise ValueError("invalid reloaded action chunk")
        write_json(args.output / "reload_check.json", {"strict_load": True, "identical_prediction": True,
            "shape": list(after.shape), "policy_execution_authorized": False})
        if args.stage == "smoke":
            write_json(args.output / "complete.json", {"stage": "synthetic_smoke", "passed": True,
                "optimizer_steps": 0, "physical_success_verified": False})
            return
    states, targets, predictions, episodes, frames, source_indices, losses = [], [], [], [], [], [], []
    full_chunks, expert_chunks, full_padding = [], [], []
    validation_sources = prepared["validation"].get("sources", [prepared["validation"]])
    source_boundaries = np.cumsum([source["frames"] for source in validation_sources])
    observed_frames = 0
    policy.eval()
    if args.validation_every:
        torch.manual_seed(args.seed)
    with torch.inference_mode():
        for raw in loader(validation_dataset if validation_dataset is not None else dataset("validation")):
            loss, _ = policy(pre(raw), reduction="none")
            chunks = post(policy.predict_action_chunk(pre(observation(raw)))).cpu().numpy()
            predicted = chunks[:, 0]
            if args.validation_every:
                full_chunks.extend(chunks.copy()); expert_chunks.extend(raw["action"].numpy().copy())
                full_padding.extend(raw["action_is_pad"].numpy().copy())
            if not torch.isfinite(loss).all() or not np.isfinite(predicted).all():
                raise ValueError("nonfinite held-out results")
            states.extend(raw["observation.state"].numpy().copy())
            targets.extend(raw["action"][:, 0].numpy().copy())
            predictions.extend(predicted.copy())
            episodes.extend(raw["episode_index"].tolist())
            frames.extend(raw["frame_index"].tolist())
            batch_frames = len(raw["episode_index"])
            source_indices.extend(np.searchsorted(source_boundaries,
                np.arange(observed_frames, observed_frames + batch_frames), side="right").tolist())
            observed_frames += batch_frames
            losses.extend(loss.cpu().tolist())
    if observed_frames != prepared["validation"]["frames"]:
        raise ValueError("validation frame count changed")
    np.savez_compressed(args.output / "heldout_predictions.npz", states=states, targets=targets,
        predictions=predictions, episode_index=episodes, frame_index=frames,
        source_index=source_indices)
    metrics = prediction_metrics(states, targets, predictions)
    metrics.update(validation_loss=float(np.mean(losses)), prediction="first_action_from_saved_expert_observations",
                   policy_execution_authorized=False)
    write_json(args.output / "validation.json", metrics)
    source_indices_array = np.asarray(source_indices)
    states_array, targets_array, predictions_array = map(np.asarray, (states, targets, predictions))
    losses_array = np.asarray(losses)
    by_source = []
    for index, source in enumerate(validation_sources):
        mask = source_indices_array == index
        result = prediction_metrics(states_array[mask], targets_array[mask], predictions_array[mask])
        result.update(source_index=index, root=source["root"],
                      demo_scope=source.get("demo_scope", "full"),
                      validation_loss=float(losses_array[mask].mean()),
                      prediction="first_action_from_saved_expert_observations",
                      policy_execution_authorized=False)
        by_source.append(result)
    write_json(args.output / "validation_by_source.json", {"sources": by_source,
        "physical_success_verified": False, "policy_execution_authorized": False})
    by_scope = {}
    for scope in sorted({source.get("demo_scope", "full") for source in validation_sources}):
        indices = [i for i, source in enumerate(validation_sources) if source.get("demo_scope", "full") == scope]
        mask = np.isin(source_indices_array, indices)
        result = prediction_metrics(states_array[mask], targets_array[mask], predictions_array[mask])
        result["validation_loss"] = float(losses_array[mask].mean())
        by_scope[scope] = result
    write_json(args.output / "validation_by_demo_scope.json", {"groups": by_scope,
        "prediction": "first_action_from_saved_expert_observations",
        "physical_success_verified": False, "policy_execution_authorized": False})
    selection_result = {}
    if args.validation_every:
        from lerobot.bamboo_sorting.policy.joint_checkpoint_selection import score_action_chunks
        full_rows = monitor_plan["full_observations"]
        final_selection = score_action_chunks(full_rows, states, expert_chunks, full_chunks, full_padding)
        final_selection.update(seeds=[args.seed], frames=len(full_rows), coverage=[(
            r["demo_scope"], r["scene_id"], r["source_index"], r["episode_index"], r["frame_index"], r["phase"])
            for r in full_rows])
        folder = args.output / "full_validation"
        folder.mkdir()
        write_json(folder / "final.json", final_selection)
        del policy, pre, post
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        policy, pre, post = load_checkpoint(best_path)
        best_selection = evaluate_monitor(policy, pre, post, validation_dataset, full_rows,
                                          batch_size=args.batch_size, seeds=[args.seed])
        write_json(folder / "best.json", best_selection)
        write_json(folder / "comparison.json", dict(
            final_vs_best=selection_comparison(final_selection, best_selection),
            selected_by="fixed monitor, three paired seeds; full validation is a separate diagnostic",
            best_checkpoint=str(best_path), final_checkpoint=str(checkpoint), automatic_deployment=False))
        selection_result = dict(best_checkpoint=str(best_path), final_checkpoint=str(checkpoint),
                                checkpoint_retention="verified_best_and_final", early_stopping=False)
    write_json(args.output / "complete.json", {"stage": args.stage, "completed": True, **selection_result,
        **({"optimizer_steps": final_step, "optimizer_steps_this_run": args.steps} if args.stage == "train" else {}),
        "policy_execution_authorized": False, "physical_success_verified": False})


if __name__ == "__main__":
    raise SystemExit(main())
