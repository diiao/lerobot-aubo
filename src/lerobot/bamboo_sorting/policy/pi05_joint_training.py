"""Offline PI05 expert training using the audited AUBO absolute-joint dataset."""

import gc
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..aubo_joint_contract import JOINT_TASK
from .joint_training import load_prepared_dataset, prediction_metrics, read_prepared_training_data, write_json
from .pi05_joint_adapter import (
    FP32_EXPERT_PRECISION, PRECISION_FILE, configure_joint_pi05_precision, joint_pi05_loss, load_joint_pi05_weights,
    make_joint_pi05_config, make_joint_pi05_processors, predict_joint_pi05_chunk,
)
from .joint_chunk_metrics import (
    EVALUATION_PHASES, EvaluationGeometry, action_chunk_metrics, evaluation_phase, grouped_chunk_metrics,
)

GIB = 2**30
EARLY_STOP_MIN_STEP = 5000
EARLY_STOP_PATIENCE = 3
EARLY_STOP_RELATIVE_DEGRADATION = 0.20
SELECTION_MAX_RELATIVE_REGRESSION = 0.20
SELECTION_RULE = "phase-balanced TCP position; movement and timing guards; close and release event counts diagnostic only; offline only"


def storage_budget(smoke, checkpoint_count, *, fp32_expert=False, save_optimizer=True):
    """Conservative budget: full weights per save, one final AdamW state, 4 GiB reserve."""
    checkpoint_bytes = int(smoke["checkpoint_bytes"]) + 64 * 2**20
    if fp32_expert:
        checkpoint_bytes += int(smoke["trainable_parameters"]) * 2
    optimizer_bytes = int(smoke["trainable_parameters"]) * 8 if save_optimizer else 0
    required = checkpoint_count * checkpoint_bytes + optimizer_bytes + 512 * 2**20 + 4 * GIB
    return {"checkpoint_count": checkpoint_count, "checkpoint_bytes_budget": checkpoint_bytes,
            "optimizer_bytes_upper_bound": optimizer_bytes, "reserve_bytes": 4 * GIB,
            "required_free_bytes": required}


def require_new_output(output, protected):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output must be a NEW directory; no implicit overwrite or resume")
    for path in protected:
        path = Path(path).resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError(f"output overlaps input: {path}")


def build_plan(args):
    if args.stage in ("train", "plan") and args.evaluation_geometry is None:
        raise ValueError("formal training requires --evaluation-geometry for TCP-based model selection")
    if args.evaluation_geometry is not None:
        EvaluationGeometry(json.loads(args.evaluation_geometry.read_text()))
    provenance = json.loads((args.preflight / "result.json").read_text())
    smoke = json.loads(args.smoke_report.read_text())
    if (provenance["stage"] != "preflight" or smoke["optimizer_steps"] != 0
            or not smoke["strict_base_load"] or not smoke["strict_reload"]
            or Path(smoke["preflight"]).resolve() != args.preflight.resolve()):
        raise ValueError("matching completed preflight and strict zero-update smoke reports required")
    prepared = read_prepared_training_data(provenance["prepared_data"], provenance["data_manifest"],
                                          target_conditioned=True)
    if provenance["contract"] != prepared["contract"]:
        raise ValueError("preflight and prepared data contracts differ")
    protected = [args.preflight, args.base_path, args.smoke_report,
                 provenance["prepared_data"], provenance["data_manifest"], provenance["tokenizer_path"]]
    for split in ("train", "validation"):
        for source in prepared[split].get("sources", [prepared[split]]):
            protected.extend(source[key] for key in ("root", "evidence_root", "target_annotations"))
    require_new_output(args.output, protected)
    if not (args.base_path / "model.safetensors").is_file():
        raise FileNotFoundError(args.base_path / "model.safetensors")
    config = make_joint_pi05_config(device="cuda")
    if args.learning_rate is not None:
        config.optimizer_lr = args.learning_rate
        config.scheduler_decay_lr = args.learning_rate / 10
    # Two weight copies cover best+final and old+new best during atomic replacement.
    validation_run = args.stage == "validate"
    checkpoint_count = 1 if validation_run else 0 if args.stage == "probe" else len(args.checkpoint_steps) + 2
    budget = storage_budget(smoke, checkpoint_count, fp32_expert=args.precision == "fp32-expert",
                            save_optimizer=args.stage in ("train", "plan"))
    parent = args.output.resolve().parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    plan = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    plan.update(data_manifest=provenance["data_manifest"], prepared_data=provenance["prepared_data"],
                contract=prepared["contract"], split_counts=provenance["split_counts"],
                optimizer="AdamW, foreach=False", learning_rate=config.optimizer_lr,
                normalization="train-only q01/q99", train_expert_only=True, dtype=config.dtype,
                precision_contract=FP32_EXPERT_PRECISION if args.precision == "fp32-expert" else None,
                gradient_checkpointing=True, sampling="uniform shuffled frames without replacement each epoch",
                effective_frame_epochs=args.steps * args.batch_size / prepared["train"]["frames"],
                storage={**budget, "available_bytes": free, "fits": free >= budget["required_free_bytes"]},
                validation_limitation=provenance["validation_limitation"],
                evaluation={"interval": args.eval_interval,
                    "early_stopping_enabled": args.early_stopping,
                    "frames_per_episode": args.eval_frames_per_episode,
                    "sampling": "fixed uniform and per-phase observations, event neighbors and overlapping-plan pairs",
                    "selection_metric": "episode_phase_macro_precontact_tcp_error_mm",
                    "selection_rule": SELECTION_RULE,
                    "selection_max_relative_regression": SELECTION_MAX_RELATIVE_REGRESSION,
                    "guard_reference": "stricter of initial baseline and previous saved best; missing coverage is not zero error",
                    "full_validation": "strictly loaded best and final; all validation frames; seeds seed, seed+1, seed+2",
                    "early_stop_min_step": EARLY_STOP_MIN_STEP,
                    "early_stop_patience": EARLY_STOP_PATIENCE,
                    "early_stop_relative_degradation": EARLY_STOP_RELATIVE_DEGRADATION,
                    "early_stop_rule": "both selection primary and loss regress versus best; three-strip only"},
                checkpoint_retention="one FP32 candidate; no optimizer save" if validation_run else "best and final, plus explicitly requested steps",
                automatic_checkpoint_deletion="superseded best created by this run only",
                implicit_resume_supported=False,
                data_preflight_copied=False, physical_grasp_verified=False, policy_execution_authorized=False)
    if validation_run:
        plan.update(sampling="fixed onset/approach/close observations from three training sources; identical shuffled batches in both controls",
                    effective_frame_epochs=None, comparison_steps_per_control=args.steps,
                    comparison_compute="BF16 autocast in both controls", save_optimizer_state=False)
    return plan, prepared, provenance, config


def backward_batch(policy, preprocessor, raw, *, clip_norm, optimizer=None, inspect_updates=False):
    """Same loss/gradient path for probe and train; only train supplies an optimizer."""
    if any(task != JOINT_TASK for task in raw["task"]):
        raise ValueError("unexpected AUBO training instruction")
    policy.zero_grad(set_to_none=True)
    loss = joint_pi05_loss(policy, preprocessor(raw))
    if not torch.isfinite(loss):
        raise ValueError("nonfinite PI05 loss; optimizer was not updated")
    loss.backward()
    params = [p for p in policy.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in params):
        raise ValueError("no trainable gradients")
    if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
        raise ValueError("frozen parameters received gradients")
    norm = torch.nn.utils.clip_grad_norm_(params, clip_norm, error_if_nonfinite=True)
    if norm == 0:
        raise ValueError("zero action-expert gradient")
    snapshots = []
    if inspect_updates and optimizer is not None:
        for name, p in policy.named_parameters():
            wanted = any(part in name for part in ("layers.0.self_attn.q_proj.weight", "layers.9.mlp.down_proj.weight", "layers.17.self_attn.o_proj.weight"))
            if p.grad is not None and (wanted or (not snapshots and p.ndim >= 2)):
                stride = max(1, p.numel() // 65536)
                snapshots.append((name, p, stride, p.detach().view(-1)[::stride].clone()))
    if optimizer is not None:
        optimizer.step()
    result = {"loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
              "optimizer_steps": int(optimizer is not None)}
    if snapshots:
        result["parameter_updates"] = [dict(name=name, sampled_elements=before.numel(),
            changed_elements=int(torch.count_nonzero(p.detach().view(-1)[::stride]-before)),
            parameter_dtype=str(p.dtype), gradient_dtype=str(p.grad.dtype),
            adam_moment_dtype=str(optimizer.state[p]["exp_avg"].dtype)) for name, p, stride, before in snapshots]
    policy.zero_grad(set_to_none=True)
    return result


def _loader(dataset, args, *, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))


def _observation(raw):
    return {key: value for key, value in raw.items() if key.startswith("observation.") or key == "task"}


def _save_checkpoint(policy, path, config, stats, provenance, step, plan):
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"checkpoint or unfinished checkpoint already exists: {path}")
    if shutil.disk_usage(path.parent).free < plan["storage"]["checkpoint_bytes_budget"] + 4 * GIB:
        raise RuntimeError("not enough space to save a full checkpoint with 4 GiB reserve")
    # New directories only. A crash leaves an explicit .incomplete directory, never a false final.
    temporary.mkdir(exist_ok=False)
    shutil.copytree(provenance["tokenizer_path"], temporary / "tokenizer")
    pre, post = make_joint_pi05_processors(config, stats, temporary / "tokenizer")
    # Tokenizer is already prepared in the temporary directory; its saved path must survive rename.
    for processor in pre.steps:
        if hasattr(processor, "tokenizer_name"):
            processor.tokenizer_name = str(path / "tokenizer")
    pre.save_pretrained(temporary)
    post.save_pretrained(temporary)
    policy.save_pretrained(temporary)
    if getattr(policy, "aubo_joint_precision", None) is not None:
        write_json(temporary / PRECISION_FILE, policy.aubo_joint_precision)
    write_json(temporary / "aubo_joint_contract.json", provenance["contract"])
    write_json(temporary / "training_progress.json", {"optimizer_steps": step,
        "preflight": plan["preflight"], "data_manifest": provenance["data_manifest"],
        "prepared_data": provenance["prepared_data"], "implicit_resume_supported": False,
        "physical_grasp_verified": False, "policy_execution_authorized": False})
    temporary.rename(path)


def episode_event_frames(states, actions, frames):
    states, actions, frames = map(np.asarray, (states, actions, frames))
    if (states.shape != (len(frames), 7) or actions.shape != states.shape or not len(frames)
            or np.any(np.diff(frames) != 1) or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError("evaluation requires finite states/actions and contiguous ordered expert frames")
    moving = np.flatnonzero(np.max(np.abs(states[:, :6]-states[0, :6]), axis=1) > .2)
    transitions = np.diff(np.r_[states[0, 6] >= 50, actions[:, 6] >= 50].astype(int))
    closes = np.flatnonzero(transitions == 1)
    releases = np.flatnonzero(transitions == -1)
    close = int(closes[0]) if len(closes) else None
    releases = releases[releases > close] if close is not None else releases
    return {"onset": int(frames[moving[0]]) if len(moving) else None,
            "close": int(frames[close]) if close is not None else None,
            "release": int(frames[releases[0]]) if len(releases) else None}


def evaluation_episode_events(record):
    """Read expert onset/close/release identities without decoding video."""
    import pyarrow.parquet as pq
    events = {}
    for source_index, source in enumerate(record.get("sources", [record])):
        data = pq.read_table(sorted((Path(source["root"])/"data").glob("chunk-*/*.parquet")),
                             columns=["episode_index", "frame_index", "observation.state", "action"]).to_pydict()
        for episode in source["episodes"]:
            selected = np.asarray(data["episode_index"]) == episode
            states = np.asarray(data["observation.state"])[selected]
            actions = np.asarray(data["action"])[selected]
            frames = np.asarray(data["frame_index"])[selected]
            events[(source_index, episode)] = episode_event_frames(states, actions, frames)
    return events


def motion_onset_frames(record):
    return {key: value["onset"] for key, value in evaluation_episode_events(record).items()
            if value["onset"] is not None}


def monitoring_indices(record, frames_per_episode, *, onsets=None, events=None):
    """Select stable identities, including close-boundary frames, without reading video."""
    indices, offset = [], 0
    for source_index, source in enumerate(record.get("sources", [record])):
        for episode in source["episodes"]:
            rows = [i for i, frame in enumerate(source["target_frames"])
                    if frame["episode_index"] == episode]
            if not rows:
                raise ValueError("monitor episode has no frames")
            selected = set(np.linspace(0, len(rows)-1, min(frames_per_episode, len(rows)), dtype=int))
            if events is not None:
                episode_events = events[(source_index, episode)]
                labels = [evaluation_phase(source["target_frames"][row]["frame_index"], episode_events) for row in rows]
                for phase in EVALUATION_PHASES:
                    positions = [i for i, label in enumerate(labels) if label == phase]
                    if positions:
                        quota = min(len(positions), max(2, frames_per_episode//len(EVALUATION_PHASES)))
                        selected.update(positions[i] for i in np.linspace(0, len(positions)-1, quota, dtype=int))
                        # About 320 ms at 25 Hz: useful overlap at observed replanning delays.
                        anchor = positions[len(positions)//2]
                        selected.add(anchor)
                        if anchor+8 in positions:
                            selected.add(anchor+8)
            if onsets is not None and (source_index, episode) in onsets:
                onset = onsets[(source_index, episode)]
                selected.update(i for i, row in enumerate(rows)
                                if source["target_frames"][row]["frame_index"] in (onset, onset+25, onset+75))
            for position in range(1, len(rows)):
                if source["target_frames"][rows[position]]["transport"] != source["target_frames"][rows[position-1]]["transport"]:
                    selected.update(range(max(0, position-2), min(len(rows), position+3)))
            indices.extend(offset + rows[i] for i in sorted(selected))
        offset += source["frames"]
    if len(set(indices)) != len(indices) or offset != record["frames"]:
        raise ValueError("monitor frame identities overlap or counts changed")
    return sorted(indices)


def selection_comparison(metrics, best):
    """Relative screening budget, not a physical grasp acceptance threshold."""
    if "offline_selection" not in metrics:
        # Historical callers keep their original explicitly offline scoring.
        primary = "execution_window_normalized_action_mae" if "execution_window_normalized_action_mae" in metrics else "normalized_action_mae"
        return {"improved": best is None or metrics[primary] < best[primary],
                "primary": metrics[primary], "rule": "legacy action-error proxy", "blocked_guards": []}
    candidate = metrics["offline_selection"]
    if candidate is None or not np.isfinite(candidate["primary"]) or candidate["primary"] < 0:
        raise ValueError("TCP selection requires geometry and preclose observation coverage")
    guards = candidate["guards"]
    if any(value is not None and (not np.isfinite(value) or value < 0) for value in guards.values()):
        raise ValueError("nonfinite or negative selection guard")
    blocked = []
    if best is not None:
        previous = best["offline_selection"]
        if (candidate["primary_name"] != previous["primary_name"]
                or candidate["available_precontact_phases"] != previous["available_precontact_phases"]
                or guards.keys() != previous["guards"].keys()):
            raise ValueError("selection coverage or metric contract changed")
        for key, value in guards.items():
            reference = previous["guards"][key]
            if (value is None) != (reference is None):
                raise ValueError(f"selection guard coverage changed: {key}")
            # Full-block event counts remain diagnostics; online replanning can
            # replace the block tail. Event timing remains a separate guard.
            if key in {"close_event_errors_per_window", "release_event_errors_per_window"}:
                continue
            if reference is not None:
                # Repeated 19% regressions must not compound beyond the baseline budget.
                reference = min(reference, previous.get("guard_reference", previous["guards"])[key])
            if value is not None and value > max(reference, 1e-12)*(1+SELECTION_MAX_RELATIVE_REGRESSION):
                blocked.append(key)
    return {"improved": best is None or (candidate["primary"] < previous["primary"] and not blocked),
            "primary": candidate["primary"], "rule": SELECTION_RULE, "blocked_guards": blocked,
            "diagnostic_only_guards": [key for key in guards if key in {
                "close_event_errors_per_window", "release_event_errors_per_window"}],
            "max_relative_regression": SELECTION_MAX_RELATIVE_REGRESSION}


def validation_decision(metrics, best, bad_checks, step, *, early_stopping=True):
    """Stop persistent regression, not one noisy loss; this does not diagnose its cause."""
    comparison = selection_comparison(metrics, best)
    if comparison["improved"]:
        if "offline_selection" in metrics:
            previous = best["offline_selection"] if best is not None else metrics["offline_selection"]
            reference = dict(previous.get("guard_reference", previous["guards"]))
            metrics = {**metrics, "offline_selection": {**metrics["offline_selection"], "guard_reference": reference}}
        return metrics, 0, False, True
    threshold = 1 + EARLY_STOP_RELATIVE_DEGRADATION
    regressed = (step >= EARLY_STOP_MIN_STEP
                 and comparison["primary"] > max(selection_comparison(best, None)["primary"], 1e-12)*threshold
                 and metrics["validation_loss"] > max(best["validation_loss"], 1e-12)*threshold)
    bad_checks = bad_checks + 1 if regressed else 0
    return best, bad_checks, early_stopping and bad_checks >= EARLY_STOP_PATIENCE, False


def _save_best(policy, path, previous, config, stats, provenance, step, plan, metrics):
    """Publish a complete new best before removing only this run's preceding best."""
    if previous is not None:
        if (previous.parent != path.parent or not previous.name.startswith("best_step_")
                or previous.is_symlink() or previous == path):
            raise ValueError("refusing to replace a checkpoint outside this run's best sequence")
    _save_checkpoint(policy, path, config, stats, provenance, step, plan)
    pointer = path.parent.parent / "best_checkpoint.json"
    temporary = pointer.with_suffix(".incomplete")
    write_json(temporary, {"path": str(path), "optimizer_steps": step, "metrics": metrics})
    temporary.replace(pointer)
    if previous is not None:
        shutil.rmtree(previous)


def _evaluate(policy, pre, post, dataset, args, stats, *, output):
    states, targets, predictions, episodes, frames, sources = [], [], [], [], [], []
    chunks, expert_chunks, padding = [], [], []
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    identities = np.asarray(dataset.indices if isinstance(dataset, Subset) else np.arange(len(dataset)))
    boundaries = getattr(base, "cumulative_sizes", [len(base)])
    weighted_loss = valid_steps = 0
    was_training = policy.training
    try:
        policy.eval()
        # Evaluations use identical flow noise without changing subsequent training RNG.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.inference_mode():
            torch.manual_seed(args.seed)
            for raw in _loader(dataset, args, shuffle=False):
                batch = pre(raw)
                loss = joint_pi05_loss(policy, batch)
                chunk = post(predict_joint_pi05_chunk(policy, pre(_observation(raw)))).float().cpu()
                prediction = chunk[:, 0]
                if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                    raise ValueError("nonfinite validation result")
                count = int((~raw["action_is_pad"]).sum())
                weighted_loss += float(loss) * count
                valid_steps += count
                sources.extend(np.searchsorted(boundaries, identities[len(states):len(states)+len(prediction)], side="right"))
                states.extend(raw["observation.state"].numpy())
                targets.extend(raw["action"][:, 0].numpy())
                predictions.extend(prediction.numpy())
                chunks.extend(chunk.numpy())
                expert_chunks.extend(raw["action"].numpy())
                padding.extend(raw["action_is_pad"].numpy())
                episodes.extend(raw["episode_index"].tolist())
                frames.extend(raw["frame_index"].tolist())
    finally:
        policy.train(was_training)
    if len(states) != len(dataset):
        raise ValueError("validation frame coverage mismatch")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "heldout_predictions.npz", states=states, targets=targets,
                        predictions=predictions, episode_index=episodes, frame_index=frames, source_index=sources,
                        action_chunks=chunks, expert_action_chunks=expert_chunks, action_is_pad=padding)
    metrics = prediction_metrics(states, targets, predictions)
    action_range = np.asarray(stats["action"]["q99"]) - np.asarray(stats["action"]["q01"])
    metrics.update(validation_loss=weighted_loss / valid_steps,
                   normalized_action_mae=float((2*np.abs(np.asarray(predictions)-targets)/action_range).mean()),
                   prediction="first action from saved expert observations",
                   validation_limitation="three-strip only; not physical grasp success")
    geometry_path = getattr(args, "evaluation_geometry", None)
    geometry = EvaluationGeometry(json.loads(Path(geometry_path).read_text())) if geometry_path else None
    onsets = getattr(base, "aubo_motion_onsets", {})
    onset_flags = [onsets.get((int(source), int(ep))) == int(frame)
                   for source, ep, frame in zip(sources, episodes, frames, strict=True)]
    metrics.update(action_chunk_metrics(states, expert_chunks, chunks, np.asarray(padding), stats,
                                       geometry=geometry, motion_onset=onset_flags))
    events = getattr(base, "aubo_evaluation_events", None)
    if events is not None:
        phases = [evaluation_phase(int(frame), events[(int(source), int(episode))])
                  for source, episode, frame in zip(sources, episodes, frames, strict=True)]
        metrics.update(grouped_chunk_metrics(states, expert_chunks, chunks, padding, stats,
            {"source_index": sources, "episode_index": episodes, "frame_index": frames, "phase": phases}, geometry=geometry))
        metrics.update(evaluation_seed=args.seed, selection_scope="fixed expert observations; offline only")
    write_json(output / "validation.json", metrics)
    return metrics


def _load_saved_policy(checkpoint):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action,
    )
    policy = PI05Policy(PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True))
    load_joint_pi05_weights(policy, checkpoint / "model.safetensors")
    pre = PolicyProcessorPipeline.from_pretrained(checkpoint, "policy_preprocessor.json", local_files_only=True,
            to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(checkpoint, "policy_postprocessor.json", local_files_only=True,
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return policy.eval().requires_grad_(False), pre, post


def _review_saved_checkpoints(args, stats, validation, monitor, best_path, raw, final_prediction):
    """Sequential loads; no extra weight copies, training or model promotion."""
    seeds = [args.seed+i for i in range(3)]
    results, summaries = {}, {}
    for label, checkpoint in (("final", args.output/"final"), ("best", best_path)):
        policy, pre, post = _load_saved_policy(checkpoint)
        try:
            if label == "final":
                torch.manual_seed(args.seed)
                with torch.inference_mode():
                    after = post(predict_joint_pi05_chunk(policy, pre(_observation(raw)))).cpu()
                torch.testing.assert_close(final_prediction, after, rtol=0, atol=0)
                if after.shape != (len(raw["action"]), 50, 7) or not torch.isfinite(after).all():
                    raise ValueError("invalid saved action chunk")
                write_json(args.output/"reload_check.json", {"strict_reload": True, "same_seed_equal": True})
            else:
                pointer = json.loads((args.output/"best_checkpoint.json").read_text())
                expected = args.output/"validation"/f"step_{pointer['optimizer_steps']:06d}"/"heldout_predictions.npz"
                destination = args.output/"full_validation"/"best_reload_monitor"
                _evaluate(policy, pre, post, monitor, args, stats, output=destination)
                with np.load(expected) as before, np.load(destination/"heldout_predictions.npz") as after:
                    for key in ("source_index", "episode_index", "frame_index", "action_chunks"):
                        np.testing.assert_array_equal(before[key], after[key])
                write_json(destination/"reload_check.json", {"strict_reload": True, "monitor_predictions_equal": True})
            runs = []
            for seed in seeds:
                seed_args = SimpleNamespace(**{**vars(args), "seed": seed})
                # Preserve the existing root final-evaluation interface for the first seed.
                destination = args.output if label == "final" and seed == args.seed else args.output/"full_validation"/label/f"seed_{seed}"
                metrics = _evaluate(policy, pre, post, validation, seed_args, stats, output=destination)
                if metrics.get("offline_selection") is None:
                    raise ValueError("full validation lacks TCP selection coverage")
                runs.append(metrics)
            results[label] = runs
            values = [r["offline_selection"]["primary"] for r in runs]
            summaries[label] = {"checkpoint": str(checkpoint), "seeds": seeds, "frames_per_seed": len(validation),
                "primary_name": runs[0]["offline_selection"]["primary_name"],
                "primary_by_seed": values, "primary_mean": float(np.mean(values)),
                "primary_std": float(np.std(values)), "primary_min": float(min(values)), "primary_max": float(max(values))}
            write_json(args.output/"full_validation"/label/"summary.json", summaries[label])
        finally:
            del policy, pre, post
            gc.collect()
            torch.cuda.empty_cache()
    final_wins = [selection_comparison(f, b)["improved"] for f, b in zip(results["final"], results["best"], strict=True)]
    best_wins = [selection_comparison(b, f)["improved"] for f, b in zip(results["final"], results["best"], strict=True)]
    preference = "final" if all(final_wins) else "best" if all(best_wins) else "inconclusive"
    report = {"checkpoints": summaries, "final_preferred_by_seed": final_wins, "best_preferred_by_seed": best_wins,
              "consistent_offline_preference": preference,
              "monitor_best_checkpoint_unchanged": str(best_path), "automatic_deployment": False,
              "physical_grasp_verified": False,
              "limitation": "same three-strip development validation; seed spread is not a grasp success rate"}
    write_json(args.output/"full_validation"/"comparison.json", report)
    return report


def run(args, plan, prepared, provenance, config):
    if not torch.cuda.is_available():
        raise RuntimeError("PI05 probe/train requires CUDA")
    if not plan["storage"]["fits"]:
        raise RuntimeError("planned checkpoint/optimizer budget exceeds available disk")
    if args.stage == "validate":
        from .pi05_joint_validation import run_precision_validation
        return run_precision_validation(args, plan, prepared, provenance, config)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    stats = json.loads((args.preflight / "train_quantile_stats.json").read_text())
    pre, post = make_joint_pi05_processors(config, stats, provenance["tokenizer_path"])
    train = load_prepared_dataset(prepared["train"])
    validation = load_prepared_dataset(prepared["validation"])
    print(f"loaded train={len(train)} validation={len(validation)}; strictly loading PI05", flush=True)
    policy = PI05Policy(config)
    if args.precision == "fp32-expert":
        # Official base experts are FP32: set destination storage before loading
        # so BF16 construction cannot round their values during the copy.
        configure_joint_pi05_precision(policy)
    load_joint_pi05_weights(policy, args.base_path / "model.safetensors")
    if any(p.requires_grad for p in policy.model.paligemma_with_expert.paligemma.parameters()):
        raise ValueError("visual-language backbone must be frozen")
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = scheduler = None
    if args.stage == "train":
        optimizer = torch.optim.AdamW(params, lr=config.optimizer_lr, betas=config.optimizer_betas,
                                     eps=config.optimizer_eps, weight_decay=config.optimizer_weight_decay,
                                     foreach=False)
        scheduler = config.get_scheduler_preset().build(optimizer, args.steps)
    best = best_path = None
    bad_checks = 0
    stopped_early = False
    if args.stage == "train":
        validation.aubo_evaluation_events = evaluation_episode_events(prepared["validation"])
        validation.aubo_motion_onsets = {key: event["onset"] for key, event in validation.aubo_evaluation_events.items()
                                        if event["onset"] is not None}
        indices = monitoring_indices(prepared["validation"], args.eval_frames_per_episode,
                                     onsets=validation.aubo_motion_onsets, events=validation.aubo_evaluation_events)
        monitor = Subset(validation, indices)
        write_json(args.output / "validation_samples.json", {"indices": indices,
                   "frames": len(indices), "seed": args.seed,
                   "validation_limitation": provenance["validation_limitation"]})
        best = _evaluate(policy, pre, post, monitor, args, stats,
                         output=args.output / "validation" / "step_000000")
        selection_comparison(best, None)  # Fail before updates if phase/TCP coverage is absent.
        parent = args.output / "checkpoints"
        parent.mkdir()
        best_path = parent / "best_step_000000"
        _save_best(policy, best_path, None, config, stats, provenance, 0, plan, best)
        print(json.dumps({"validation_step": 0, **best}), flush=True)
    batches = _loader(train, args, shuffle=True)
    iterator = iter(batches)
    steps = args.probe_batches if args.stage == "probe" else args.steps
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    records = []
    window_losses = []
    with (args.output / "training.jsonl").open("x") as stream:
        for step in range(1, steps + 1):
            data_started = time.monotonic()
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                raw = next(iterator)
            data_seconds = time.monotonic() - data_started
            torch.cuda.synchronize()
            compute_started = time.monotonic()
            record = backward_batch(policy, pre, raw, clip_norm=config.optimizer_grad_clip_norm,
                                    optimizer=optimizer, inspect_updates=step == 1 or step % 100 == 0 or step == steps)
            torch.cuda.synchronize()
            record.update(batch=step, learning_rate=optimizer.param_groups[0]["lr"] if optimizer else None,
                          data_seconds=data_seconds, compute_seconds=time.monotonic() - compute_started,
                          elapsed_seconds=time.monotonic() - started)
            if scheduler is not None:
                scheduler.step()
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if args.stage == "probe":
                records.append(record)
            else:
                window_losses.append(record["loss"])
            if step == 1 or step % 100 == 0 or args.stage == "probe":
                print(json.dumps(record), flush=True)
            if args.stage == "train" and step in args.checkpoint_steps:
                parent = args.output / "checkpoints"
                parent.mkdir(exist_ok=True)
                _save_checkpoint(policy, parent / f"step_{step:06d}", config, stats, provenance, step, plan)
            if args.stage == "train" and (step % args.eval_interval == 0 or step == steps):
                evaluation_started = time.monotonic()
                metrics = _evaluate(policy, pre, post, monitor, args, stats,
                                    output=args.output / "validation" / f"step_{step:06d}")
                selection = selection_comparison(metrics, best)
                best, bad_checks, stopped_early, improved = validation_decision(
                    metrics, best, bad_checks, step, early_stopping=args.early_stopping)
                if improved:
                    path = args.output / "checkpoints" / f"best_step_{step:06d}"
                    _save_best(policy, path, best_path, config, stats, provenance, step, plan, metrics)
                    best_path = path
                evaluation_record = {"optimizer_steps": step, "metrics": metrics,
                    "mean_training_loss_since_last_eval": float(np.mean(window_losses)),
                    "best_checkpoint": str(best_path), "best_execution_window_normalized_action_mae": best["execution_window_normalized_action_mae"],
                    "selection_comparison": selection, "best_tcp_selection": best["offline_selection"],
                    "consecutive_regressions": bad_checks, "stop_requested": stopped_early,
                    "regression_warning": bad_checks >= EARLY_STOP_PATIENCE,
                    "early_stopping_enabled": args.early_stopping,
                    "evaluation_and_save_seconds": time.monotonic()-evaluation_started}
                with (args.output / "validation_history.jsonl").open("a") as history:
                    history.write(json.dumps(evaluation_record) + "\n")
                print(json.dumps({"validation_step": step, **evaluation_record}), flush=True)
                window_losses.clear()
                if stopped_early:
                    print("persistent three-strip validation regression; saving final at current step", flush=True)
                    break
    resources = {"peak_allocated_GiB": torch.cuda.max_memory_allocated()/GIB,
                 "peak_reserved_GiB": torch.cuda.max_memory_reserved()/GIB,
                 "optimizer_memory_included": optimizer is not None,
                 "optimizer_state_GiB_upper_bound": plan["storage"]["optimizer_bytes_upper_bound"]/GIB,
                 "free_disk_GiB": shutil.disk_usage(args.output).free/GIB}
    if args.stage == "probe":
        # The full held-out dataset was constructed; sample each source without changing weights.
        boundaries = np.cumsum([s["frames"] for s in prepared["validation"].get("sources", [prepared["validation"]])])
        identities = []
        for index in [0, *boundaries[:-1]]:
            item = validation[int(index)]
            identities.append({"episode": int(item["episode_index"]), "frame": int(item["frame_index"])})
        write_json(args.output / "result.json", {"stage": "probe", "batch_size": args.batch_size,
            "optimizer_steps": 0, "batches": records, "validation_source_samples": identities,
            "data_frames": {"train": len(train), "validation": len(validation)}, "resources": resources,
            "frozen_backbone_has_gradients": False, "physical_grasp_verified": False})
        return
    state_budget = plan["storage"]["optimizer_bytes_upper_bound"] + plan["storage"]["checkpoint_bytes_budget"] + 4*GIB
    if shutil.disk_usage(args.output).free < state_budget:
        raise RuntimeError("insufficient disk for final weights and optimizer state")
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "implicit_resume_supported": False}, args.output / "training_state.pt")
    checkpoint = args.output / "final"
    policy.eval()
    _save_checkpoint(policy, checkpoint, config, stats, provenance, step, plan)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        before = post(predict_joint_pi05_chunk(policy, pre(_observation(raw)))).cpu()
    del optimizer, scheduler, params, policy
    gc.collect()
    torch.cuda.empty_cache()
    full_review = _review_saved_checkpoints(args, stats, validation, monitor, best_path, raw, before)
    write_json(args.output / "result.json", {"stage": "train", "optimizer_steps": step,
        "planned_steps": args.steps, "completed": step == args.steps, "run_finished": True,
        "early_stopped": stopped_early,
        "stop_reason": "persistent_three_strip_validation_regression" if stopped_early else "step_budget_reached",
        "best_checkpoint": str(best_path), "resources": resources, "physical_grasp_verified": False,
        "full_validation_completed": True, "consistent_offline_preference": full_review["consistent_offline_preference"],
        "policy_execution_authorized": False})
