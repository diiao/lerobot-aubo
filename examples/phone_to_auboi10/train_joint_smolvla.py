#!/usr/bin/env python
"""Full-joint SmolVLA: plan / preflight / synthetic smoke / train / evaluate.

Default plan is hardware-free, model-free and writes nothing. All model use is
offline. Smoke uses synthetic data and ZERO optimizer steps, never real capture.
"""

import argparse
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
    for name in ("train-root", "train-evidence", "validation-root", "validation-evidence",
                 "train-source-root", "validation-source-root", "base-path", "vlm-path", "output", "checkpoint"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--learning-rate", type=float, help="Override optimizer peak learning rate for an explicit continuation")
    parser.add_argument("--preclose-loss-weight", type=float,
                        help="Training-only future-action weight near expert closure; 1 keeps the uniform objective")
    parser.add_argument("--preclose-frames", type=int, default=10,
                        help="Expert frames before first close to weight, including the close frame")
    parser.add_argument("--holding-loss-coefficient", type=float, default=0.,
                        help="Training-only six-joint clean-action increment loss; 0 disables it")
    parser.add_argument("--holding-tolerance-deg", type=float, default=.01,
                        help="Expert max joint distance from close pose defining its contiguous hold")
    parser.add_argument("--gripper-close-loss-weight", type=float,
                        help="Training-only last-coordinate weight at expert close and following rows")
    parser.add_argument("--gripper-close-frames", type=int, default=3,
                        help="Inclusive close-window length, clipped before the first release")
    parser.add_argument("--phase-balanced-sampling", action="store_true",
                        help="Training-only 4/2/1/1 whole/late-hold/close/release input batches; original loss")
    parser.add_argument("--fit-observations", type=Path,
                        help="Diagnostic only: JSON sample_keys selecting one fixed training batch; original loss")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    for name in ("learning_rate", "preclose_loss_weight", "holding_tolerance_deg", "gripper_close_loss_weight"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.preclose_frames < 0:
        parser.error("preclose-frames must be nonnegative")
    if not math.isfinite(args.holding_loss_coefficient) or args.holding_loss_coefficient < 0:
        parser.error("holding-loss-coefficient must be finite and nonnegative")
    if args.gripper_close_frames < 1:
        parser.error("gripper-close-frames must be positive")
    candidates = [args.preclose_loss_weight is not None, args.holding_loss_coefficient > 0,
                  args.gripper_close_loss_weight is not None, args.phase_balanced_sampling,
                  args.fit_observations is not None]
    if sum(candidates) > 1:
        parser.error("evaluate training objectives, phase sampling and fixed-observation diagnostics separately")
    if any(candidates) and (not args.target_conditioned or args.stage not in ("plan", "train")):
        parser.error("training objectives require target-conditioned plan/train")
    if args.phase_balanced_sampling and args.batch_size != 8:
        parser.error("phase-balanced-sampling requires batch-size 8 for the fixed 4/2/1/1 allocation")
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
                     args.base_path, args.vlm_path, args.checkpoint, args.fit_observations):
            if path is not None and (output == path.resolve() or path.resolve() in output.parents
                                     or output in path.resolve().parents):
                parser.error("output must be independent of input roots")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(fixed_joint_targets_deg={}, state_action_names=["J1", "J2", "J3", "J4", "J5", "J6", "gripper_pos"],
                chunk_size=50, n_action_steps=1, smoke_optimizer_steps=0, policy_execution_authorized=False)
    if args.fit_observations:
        plan["diagnostic_only"] = True
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.stage == "plan":
        return 0
    fit_keys = None
    if args.fit_observations:
        fit_keys = json.loads(args.fit_observations.read_text())["sample_keys"]
        if (not isinstance(fit_keys, list) or len(fit_keys) != args.batch_size
                or not all(isinstance(key, str) for key in fit_keys)
                or len(set(fit_keys)) != len(fit_keys)):
            raise ValueError("fit-observations requires unique sample_keys with count equal to batch-size")
    from lerobot.bamboo_sorting.policy.joint_training import (
        prepare_training_data, prepare_training_manifest, read_prepared_training_data, write_json,
    )
    prepared = None
    if args.stage != "smoke":
        if args.prepared_data:
            prepared = read_prepared_training_data(args.prepared_data, args.data_manifest,
                                                  target_conditioned=args.target_conditioned)
        elif args.data_manifest:
            prepared = prepare_training_manifest(args.data_manifest, target_conditioned=args.target_conditioned)
        else:
            prepared = prepare_training_data(args.train_root, args.train_evidence,
                args.validation_root, args.validation_evidence,
                train_source_root=args.train_source_root, validation_source_root=args.validation_source_root)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    if prepared is not None:
        write_json(args.output / "data_preflight.json", prepared)
    if args.stage == "preflight":
        return 0
    # Set offline behavior only in this explicitly invoked model subprocess.
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    run_model(args, prepared, fit_keys=fit_keys)
    return 0


def run_model(args, prepared, *, fit_keys=None):
    import gc
    import importlib.metadata
    import math
    import random
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, JOINT_IMAGE_KEYS, JOINT_TASK, joint_contract_record
    from lerobot.bamboo_sorting.policy.joint_training import prediction_metrics, write_json
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import make_joint_smolvla_config, make_joint_smolvla_processors
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import SmolVLAJointOfflineAdapter
    if args.target_conditioned:
        from lerobot.bamboo_sorting.policy.joint_target_policy import (
            SmolVLAJointTargetOfflineAdapter, target_policy_contract_record,
        )
        contract_record = target_policy_contract_record
        adapter_class = SmolVLAJointTargetOfflineAdapter
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

    def load_checkpoint(path):
        if json.loads((path / "aubo_joint_contract.json").read_text()) != contract_record():
            raise ValueError("checkpoint is not full-joint AUBO")
        cfg = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        cfg.device = args.device
        model = SmolVLAPolicy.from_pretrained(path, config=cfg, local_files_only=True, strict=True)
        pre = PolicyProcessorPipeline.from_pretrained(path, "policy_preprocessor.json",
            overrides={"device_processor": {"device": args.device}},
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(path, "policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        adapter_class(model, pre, post, contract=contract_record())
        return model, pre, post

    def dataset(which):
        from lerobot.bamboo_sorting.policy.joint_training import load_prepared_dataset
        return load_prepared_dataset(prepared[which], video_backend=args.video_backend,
            preclose_loss_weight=args.preclose_loss_weight if which == "train" else None,
            preclose_frames=args.preclose_frames,
            holding_loss_coefficient=args.holding_loss_coefficient if which == "train" else 0.,
            holding_tolerance_deg=args.holding_tolerance_deg,
            gripper_close_loss_weight=args.gripper_close_loss_weight if which == "train" else None,
            gripper_close_frames=args.gripper_close_frames,
            phase_balanced_sampling=args.phase_balanced_sampling and which == "train")

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                          generator=torch.Generator().manual_seed(args.seed))

    def observation(raw):
        return {k: raw[k] for k in [*images, "observation.state", "task"]}

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
        for name in ("input_features", "output_features", "normalization_mapping", "device", "chunk_size",
                     "n_action_steps", "num_steps", "action_representation", "relative_action_stats",
                     "execution_loss_fraction", "adapt_to_pi_aloha", "use_delta_joint_actions_aloha", "push_to_hub"):
            setattr(cfg, name, getattr(joint_cfg, name))
        cfg.load_vlm_weights = False
        cfg.vlm_model_name = str(args.vlm_path.resolve())
        if args.learning_rate is not None:
            cfg.optimizer_lr = args.learning_rate
        policy = SmolVLAPolicy.from_pretrained(args.base_path, config=cfg, local_files_only=True, strict=True)
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
        pre, post = make_joint_smolvla_processors(cfg, stats, target_conditioned=args.target_conditioned)
        write_json(args.output / "train_only_stats.json", stats)
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
            train_dataset = dataset("train")
            if args.fit_observations:
                from lerobot.bamboo_sorting.policy.joint_training import FixedObservationDataset
                train_dataset = FixedObservationDataset(train_dataset, fit_keys)
                write_json(args.output / "fit_observations.json", train_dataset.report)
            if args.holding_loss_coefficient > 0 or args.gripper_close_loss_weight is not None or args.phase_balanced_sampling:
                from lerobot.bamboo_sorting.policy.joint_training import training_supervision_record
                write_json(args.output / "training_supervision.json", training_supervision_record(train_dataset))
            if args.phase_balanced_sampling:
                from lerobot.bamboo_sorting.policy.joint_training import PhaseBalancedBatchSampler, phase_sampling_inputs
                sampler = PhaseBalancedBatchSampler(*phase_sampling_inputs(train_dataset), steps=args.steps, seed=args.seed)
                write_json(args.output / "sampling_plan.json", sampler.report)
                batches = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0,
                                     generator=torch.Generator().manual_seed(args.seed))
            else:
                batches = loader(train_dataset, args.fit_observations is None)
            optimizer = cfg.get_optimizer_preset().build(params)
            warmup = min(1000, max(1, args.steps // 10))
            def scale(step):
                if step < warmup:
                    return (step + 1) / warmup
                return .025 + .975 * (1 + math.cos(math.pi * min((step - warmup) / max(1, args.steps - warmup), 1))) / 2
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
            iterator, start = iter(batches), time.monotonic()
            with (args.output / "training.jsonl").open("x") as stream:
                for step in range(1, args.steps + 1):
                    try:
                        raw = next(iterator)
                    except StopIteration:
                        iterator = iter(batches)
                        raw = next(iterator)
                    if any(task != JOINT_TASK for task in raw["task"]):
                        raise ValueError("unexpected training instruction")
                    if args.fit_observations and raw["training_sample_key"] != train_dataset.sample_keys:
                        raise ValueError("fixed diagnostic batch differs from the selected observations")
                    if args.phase_balanced_sampling and raw["training_sample_key"] != sampler.report["batches"][step-1]["sample_keys"]:
                        raise ValueError("loaded input identities differ from the frozen phase sampling plan")
                    weights = raw.get("action_loss_weights")
                    extra = {} if weights is None else {"action_loss_weights": weights}
                    if args.holding_loss_coefficient > 0:
                        extra.update(holding_pair_mask=raw["holding_pair_mask"],
                                     holding_loss_coefficient=args.holding_loss_coefficient)
                    if args.gripper_close_loss_weight is not None:
                        extra["gripper_loss_weights"] = raw["gripper_loss_weights"]
                    loss, loss_info = policy(pre(raw), **extra)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    record = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                        "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.monotonic() - start}
                    if args.fit_observations:
                        record.update(diagnostic_only=True, sample_keys=raw["training_sample_key"])
                    if args.phase_balanced_sampling:
                        record.update(uniform_valid_loss=record["loss"],
                                      sample_keys=raw["training_sample_key"], sampling_groups=sampler.groups[step-1])
                    if weights is not None:
                        record.update(uniform_valid_loss=loss_info["uniform_valid_loss"],
                            sample_keys=raw["training_sample_key"],
                            weighted_valid_timesteps=int(((weights > 1) & ~raw["action_is_pad"]).sum()))
                    if args.holding_loss_coefficient > 0 or args.gripper_close_loss_weight is not None:
                        record.update(uniform_valid_loss=loss_info["uniform_valid_loss"],
                                      sample_keys=raw["training_sample_key"])
                        if args.holding_loss_coefficient > 0:
                            record.update(holding_loss=loss_info["holding_loss"],
                                          scaled_holding_loss=loss_info["scaled_holding_loss"],
                                          holding_valid_pairs=int(raw["holding_pair_mask"].sum()))
                        else:
                            record.update(gripper_weighted_loss=loss_info["gripper_weighted_loss"],
                                gripper_weighted_valid_timesteps=int(
                                    ((raw["gripper_loss_weights"] != 1) & ~raw["action_is_pad"]).sum()))
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if step == 1 or step % 100 == 0:
                        print(f"step={step} loss={float(loss.detach()):.6f}", flush=True)
            torch.save({"step": args.steps, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                       args.output / "training_state.pt")
            del optimizer, scheduler, loss
        checkpoint = args.output / "final"
        policy.save_pretrained(checkpoint)
        pre.save_pretrained(checkpoint)
        post.save_pretrained(checkpoint)
        write_json(checkpoint / "aubo_joint_contract.json", contract_record())
        if prepared is not None:
            write_json(checkpoint / "data_preflight.json", prepared)
        if args.fit_observations:
            write_json(checkpoint / "fit_observations.json", train_dataset.report)
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
    validation_sources = prepared["validation"].get("sources", [prepared["validation"]])
    source_boundaries = np.cumsum([source["frames"] for source in validation_sources])
    observed_frames = 0
    policy.eval()
    with torch.inference_mode():
        for raw in loader(dataset("validation")):
            loss, _ = policy(pre(raw), reduction="none")
            predicted = post(policy.predict_action_chunk(pre(observation(raw))))[:, 0].cpu().numpy()
            if not torch.isfinite(loss).all() or not np.isfinite(predicted).all():
                raise ValueError("nonfinite held-out results")
            states.extend(raw["observation.state"].numpy())
            targets.extend(raw["action"][:, 0].numpy())
            predictions.extend(predicted)
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
                      validation_loss=float(losses_array[mask].mean()),
                      prediction="first_action_from_saved_expert_observations",
                      policy_execution_authorized=False)
        by_source.append(result)
    write_json(args.output / "validation_by_source.json", {"sources": by_source,
        "physical_success_verified": False, "policy_execution_authorized": False})
    write_json(args.output / "complete.json", {"stage": args.stage, "completed": True,
        "diagnostic_only": args.fit_observations is not None or (
            args.checkpoint is not None and (args.checkpoint / "fit_observations.json").is_file()),
        "policy_execution_authorized": False, "physical_success_verified": False})


if __name__ == "__main__":
    raise SystemExit(main())
