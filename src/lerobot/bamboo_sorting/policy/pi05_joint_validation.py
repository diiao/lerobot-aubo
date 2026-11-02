"""Bounded paired precision experiment; training data only, no robot/deployment."""

import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Subset

from . import pi05_joint_training as training
from .pi05_joint_adapter import (
    FP32_EXPERT_PRECISION, configure_joint_pi05_precision, load_joint_pi05_weights,
    make_joint_pi05_processors, predict_joint_pi05_chunk,
)


def precision_training_indices(record, onsets):
    """Fixed onset/approach/close observations in three TRAIN sources, never validation."""
    sources = record.get("sources", [record])
    chosen = {0, len(sources)//2, len(sources)-1}
    indices, offset = [], 0
    for source_index, source in enumerate(sources):
        if source_index in chosen:
            episode = source["episodes"][0]
            rows = [(i, f) for i, f in enumerate(source["target_frames"]) if f["episode_index"] == episode]
            onset = onsets[(source_index, episode)]
            close = next(f["frame_index"] for _, f in rows if f["transport"])
            wanted = {onset, onset+25, onset+75, close-25, close-10, close-5, close-1, close+25}
            indices.extend(offset+i for i, frame in rows if frame["frame_index"] in wanted)
        offset += source["frames"]
    if not indices:
        raise ValueError("no training observations selected for precision validation")
    return sorted(set(indices))


def _dtype_counts(policy):
    counts = {}
    for p in policy.parameters():
        if p.requires_grad:
            counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
    return counts


def run_precision_validation(args, plan, prepared, provenance, config):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if not 1 <= args.steps <= 256:
        raise ValueError("precision validation is limited to 256 optimizer steps per control")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    training.write_json(args.output/"plan.json", plan)
    stats = json.loads((args.preflight/"train_quantile_stats.json").read_text())
    pre, post = make_joint_pi05_processors(config, stats, provenance["tokenizer_path"])
    train = training.load_prepared_dataset(prepared["train"])
    validation = training.load_prepared_dataset(prepared["validation"])
    train.aubo_motion_onsets = training.motion_onset_frames(prepared["train"])
    validation.aubo_motion_onsets = training.motion_onset_frames(prepared["validation"])
    train_indices = precision_training_indices(prepared["train"], train.aubo_motion_onsets)
    validation_indices = training.monitoring_indices(prepared["validation"], args.eval_frames_per_episode,
                                                    onsets=validation.aubo_motion_onsets)
    fitting, heldout = Subset(train, train_indices), Subset(validation, validation_indices)
    training.write_json(args.output/"sample_indices.json", dict(train=train_indices, validation=validation_indices,
        train_frames=len(fitting), validation_frames=len(heldout),
        rule="Only training observations receive updates. Held-out observations are evaluation only."))
    results = {}
    started = time.monotonic()
    for name in ("checkpoint_storage", "fp32_expert"):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        folder = args.output/name
        folder.mkdir()
        policy = PI05Policy(config)
        load_joint_pi05_weights(policy, args.base_path/"model.safetensors")
        if getattr(policy, "aubo_joint_precision", None) is not None:
            raise ValueError("paired comparison must start from a legacy checkpoint/base, not an already FP32 candidate")
        if name == "fp32_expert":
            configure_joint_pi05_precision(policy)
        # Same compute, optimizer, batches and RNG; only expert storage differs.
        policy.aubo_validation_bf16_compute = True
        dtype_counts = _dtype_counts(policy)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
            lr=config.optimizer_lr, betas=config.optimizer_betas, eps=config.optimizer_eps,
            weight_decay=config.optimizer_weight_decay, foreach=False)
        torch.manual_seed(args.seed)
        before = {split:training._evaluate(policy, pre, post, dataset, args, stats, output=folder/(split+"_before"))
                  for split, dataset in [("training", fitting), ("heldout", heldout)]}
        batches = training._loader(fitting, args, shuffle=True)
        iterator = iter(batches)
        records = []
        policy.train()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed)
        with (folder/"training.jsonl").open("x") as stream:
            for step in range(1, args.steps+1):
                try:
                    raw = next(iterator)
                except StopIteration:
                    iterator = iter(batches)
                    raw = next(iterator)
                record = training.backward_batch(policy, pre, raw, clip_norm=config.optimizer_grad_clip_norm,
                    optimizer=optimizer, inspect_updates=step == 1 or step % 32 == 0 or step == args.steps)
                record.update(step=step, control=name)
                stream.write(json.dumps(record)+"\n")
                stream.flush()
                records.append(record)
                if step == 1 or step % 32 == 0 or step == args.steps:
                    print(json.dumps(record), flush=True)
        after = {split:training._evaluate(policy, pre, post, dataset, args, stats, output=folder/(split+"_after"))
                 for split, dataset in [("training", fitting), ("heldout", heldout)]}
        results[name] = dict(before=before, after=after, optimizer_steps=args.steps,
            trainable_dtype_counts=dtype_counts, mean_training_loss=float(np.mean([r["loss"] for r in records])),
            update_probes=[r for r in records if "parameter_updates" in r],
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/training.GIB)
        del optimizer, batches, iterator
        if name == "fp32_expert":
            policy.eval()
            checkpoint = args.output/"candidate"
            training._save_checkpoint(policy, checkpoint, config, stats, provenance, args.steps, plan)
            observation = training._observation(next(iter(training._loader(fitting, args, shuffle=False))))
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                expected = post(predict_joint_pi05_chunk(policy, pre(observation))).cpu()
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    # Exercise the same loader and adapter path used by the inference worker.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK
    from .joint_target_policy import target_policy_contract_record
    from .pi05_joint_adapter import PI05JointTargetOfflineAdapter
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action,
    )
    policy = PI05Policy(PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True))
    load_joint_pi05_weights(policy, checkpoint/"model.safetensors")
    if policy.aubo_joint_precision != FP32_EXPERT_PRECISION or set(_dtype_counts(policy)) != {"torch.float32"}:
        raise ValueError("FP32 expert precision was not preserved by strict reload")
    pre = PolicyProcessorPipeline.from_pretrained(checkpoint, "policy_preprocessor.json", local_files_only=True,
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(checkpoint, "policy_postprocessor.json", local_files_only=True,
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    policy.eval()
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        actual = post(predict_joint_pi05_chunk(policy, pre(observation))).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sample = fitting[0]
    inputs = training._observation(sample)
    frame = {'observation.state':sample['observation.state'].numpy().astype(np.float32),
             'task':JOINT_TASK, 'target_contract':target_policy_contract_record(),
             'target_input':dict(target_id='saved_training_target',frame_id='precision_validation',phase='approach')}
    for key in config.image_features:
        pixels = sample[key].mul(255).round().to(torch.uint8).permute(1,2,0).numpy()
        frame[key] = pixels
        inputs[key] = torch.from_numpy(pixels.transpose(2,0,1).copy()).float()/255
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        direct = post(predict_joint_pi05_chunk(policy,pre(inputs)))[0].cpu().numpy()
    adapter = PI05JointTargetOfflineAdapter(policy,pre,post,contract=target_policy_contract_record())
    torch.manual_seed(args.seed)
    decoded = adapter(frame)
    np.testing.assert_array_equal(adapter.raw_actions,direct)
    np.testing.assert_array_equal(decoded[:,:6],direct[:,:6])
    results.update(completed=True, steps_per_control=args.steps, total_optimizer_steps=2*args.steps,
        same_bf16_compute_both_controls=True, same_initial_checkpoint=str(args.base_path),
        checkpoint=str(checkpoint), strict_reload_equal=True, runtime_adapter_raw_equal=True,
        saved_trainable_dtype="float32",
        elapsed_seconds=time.monotonic()-started, hardware_access=False, deployed=False,
        physical_grasp_verified=False, target_obedience_verified=False,
        limits=["Small training-observation learnability comparison, not a full training or grasp acceptance.",
                "Both controls use BF16 autocast; the previous live model used its original no-autocast path.",
                "One seed, existing three-strip development validation, no independent four-strip test."])
    training.write_json(args.output/"result.json", results)
    print(json.dumps(dict(completed=True, result=str(args.output/"result.json"), strict_reload_equal=True)), flush=True)
