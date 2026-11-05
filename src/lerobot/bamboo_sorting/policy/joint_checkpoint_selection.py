"""Fixed held-out SmolVLA scoring and retention of this run's best checkpoint.

No robot, model construction, or GPU work occurs on import. Scores are offline
action agreement, never physical grasp success. Model use requires a caller.
"""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import random
import re
import shutil

import numpy as np

from .joint_chunk_metrics import EVALUATION_PHASES, PRECONTACT_PHASES, _event_metrics, evaluation_phase


SELECTION_RULE = {
    "primary": "scope_scene_episode_phase_macro_precontact_joint_mae_deg",
    "future_rows": [0, 19],
    "aggregation": "equal scopes; equal scenes within scope; equal episodes within scene; equal observed precontact phases",
    "minimum_improvement_deg": 1e-9,
    "max_relative_guard_regression": .20,
    "joint_guard_absolute_tolerance_deg": .05,
    "event_timing_absolute_tolerance_steps": 1.,
    "guard_reference": "best value among accepted best checkpoints; regressions cannot accumulate",
    "diagnostic_only": ["validation_loss", "close_event_errors_per_window", "release_event_errors_per_window"],
    "retention": "verified best and final only; all validation reports retained",
    "physical_success_verified": False,
}


def build_monitor_plan(record, *, points_per_phase=8):
    """Read expert metadata/Parquet once, no video decoding or repeated hashing."""
    import pyarrow.parquet as pq

    if type(points_per_phase) is not int or points_per_phase < 2:
        raise ValueError("at least two fixed observations per phase required")
    observations, indices, offset = [], [], 0
    for si, source in enumerate(record.get("sources", [record])):
        data = pq.read_table(sorted((Path(source["root"]) / "data").glob("chunk-*/*.parquet")),
                             columns=["episode_index", "frame_index", "observation.state", "action"]).to_pydict()
        identities = {(int(ep), int(frame)): i for i, (ep, frame) in
                      enumerate(zip(data["episode_index"], data["frame_index"], strict=True))}
        if len(identities) != len(data["frame_index"]):
            raise ValueError("duplicate expert frame identity")
        metadata = {e["episode_index"]: e for e in source["target_episode_metadata"]}
        positions = {}
        for i, row in enumerate(source["target_frames"]):
            positions.setdefault(row["episode_index"], []).append(i)
        if set(positions) != set(source["episodes"]) or len(source["target_frames"]) != source["frames"]:
            raise ValueError("selected episode/frame coverage changed")
        for ep in source["episodes"]:
            rows = positions[ep]
            frames = np.asarray([source["target_frames"][i]["frame_index"] for i in rows])
            selected = [identities[(ep, int(f))] for f in frames]
            states = np.asarray(data["observation.state"], dtype=float)[selected]
            actions = np.asarray(data["action"], dtype=float)[selected]
            if (not len(frames) or frames[0] != 0 or np.any(np.diff(frames) != 1)
                    or states.shape != (len(frames), 7) or actions.shape != states.shape
                    or not np.isfinite(states).all() or not np.isfinite(actions).all()):
                raise ValueError("ordered finite seven-dimensional expert episodes required")
            transitions = np.diff(np.r_[states[0, 6] >= 50, actions[:, 6] >= 50].astype(int))
            close, release = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
            if len(close) != 1 or len(release) != 1 or close[0] >= release[0]:
                raise ValueError("selection requires one expert close then one release per episode")
            moving = np.flatnonzero(np.max(np.abs(states[:, :6] - states[0, :6]), axis=1) > .2)
            events = dict(onset=int(moving[0]) if len(moving) else None,
                          close=int(close[0]), release=int(release[0]))
            labels = [evaluation_phase(int(f), events) for f in frames]
            chosen = set()
            for phase in EVALUATION_PHASES:
                candidates = np.flatnonzero(np.asarray(labels) == phase)
                if len(candidates):
                    chosen.update(candidates[np.linspace(0, len(candidates)-1,
                                  min(points_per_phase, len(candidates)), dtype=int)].tolist())
            for event in (events["close"], events["release"]):
                chosen.update(f for d in (-19, -10, -1, 0, 1, 10)
                              if 0 <= (f := event + d) < len(frames))
            for local, frame in enumerate(frames):
                observations.append(dict(dataset_index=offset+rows[local], source_index=si,
                    episode_index=int(ep), frame_index=int(frame), scene_id=metadata[ep]["scene_id"],
                    demo_scope=source.get("demo_scope", "full"), phase=labels[local]))
                if local in chosen:
                    indices.append(offset+rows[local])
        offset += source["frames"]
    observations.sort(key=lambda r: r["dataset_index"])
    if [r["dataset_index"] for r in observations] != list(range(record["frames"])):
        raise ValueError("monitor dataset ordering/count mismatch")
    selected = [observations[i] for i in sorted(indices)]
    if {r["scene_id"] for r in selected} != set(record["scenes"]):
        raise ValueError("monitor must cover every held-out scene")
    return dict(indices=sorted(indices), observations=selected, full_observations=observations,
                points_per_phase=points_per_phase, frames=len(selected), episodes=len({
                    (r["source_index"], r["episode_index"]) for r in selected}),
                scenes=sorted({r["scene_id"] for r in selected}))


def score_action_chunks(observations, states, targets, predictions, padding):
    """Use real rows 0..19; macro averaging prevents long easy scenes dominating."""
    states, targets, predictions, padding = map(np.asarray, (states, targets, predictions, padding))
    n = len(observations)
    if (not n or states.shape != (n, 7) or targets.shape != (n, 50, 7)
            or predictions.shape != targets.shape or padding.shape != (n, 50)
            or padding.dtype != np.bool_ or not all(np.isfinite(a).all() for a in (states, targets, predictions))):
        raise ValueError("finite matching [N,50,7] chunks and boolean padding required")
    valid = ~padding
    window = valid.copy(); window[:, 20:] = False
    available = window.any(axis=1)
    error = np.abs(predictions-targets)
    rows = []
    keys = sorted({(r["demo_scope"], r["scene_id"], r["source_index"], r["episode_index"], r["phase"])
                   for r in observations})
    for scope, scene, source, episode, phase in keys:
        take = np.asarray([(r["demo_scope"], r["scene_id"], r["source_index"], r["episode_index"], r["phase"])
                           == (scope, scene, source, episode, phase) for r in observations]) & available
        if not take.any():
            continue
        mask = window[take]
        row = dict(demo_scope=scope, scene_id=scene, source_index=source, episode_index=episode,
                   phase=phase, joint_mae_deg=float(error[take, :, :6][mask].mean()),
                   j6_mae_deg=float(error[take, :, 5][mask].mean()))
        for name, closing in (("close", True), ("release", False)):
            events = _event_metrics(states[take], targets[take], predictions[take], valid[take], closing=closing)
            row[name+"_timing_steps"] = events["timing_error_with_misses_steps"]
        rows.append(row)

    def macro(field, phases, scope=None):
        scopes = sorted({r["demo_scope"] for r in rows}) if scope is None else [scope]
        scope_scores = []
        for group in scopes:
            selected = [r for r in rows if r["demo_scope"] == group and r["phase"] in phases
                        and r[field] is not None]
            scene_scores = []
            for scene in sorted({r["scene_id"] for r in selected}):
                scene_rows = [r for r in selected if r["scene_id"] == scene]
                episode_scores = []
                for key in sorted({(r["source_index"], r["episode_index"]) for r in scene_rows}):
                    values = [r[field] for r in scene_rows if (r["source_index"], r["episode_index"]) == key]
                    episode_scores.append(float(np.mean(values)))
                scene_scores.append(float(np.mean(episode_scores)))
            if scene_scores:
                scope_scores.append(float(np.mean(scene_scores)))
        return float(np.mean(scope_scores)) if scope_scores else None

    primary = macro("joint_mae_deg", PRECONTACT_PHASES)
    preclose_scenes = [float(np.mean([r["joint_mae_deg"] for r in rows
                                     if r["scene_id"] == scene and r["phase"] == "preclose"]))
                       for scene in sorted({r["scene_id"] for r in rows if r["phase"] == "preclose"})]
    guards = {"preclose_joint_mae_deg": macro("joint_mae_deg", ["preclose"]),
              "worst_scene_preclose_joint_mae_deg": max(preclose_scenes) if preclose_scenes else None,
              "preclose_j6_mae_deg": macro("j6_mae_deg", ["preclose"]),
              "transport_joint_mae_deg": macro("joint_mae_deg", ["transport", "release"]),
              "close_timing_steps": macro("close_timing_steps", EVALUATION_PHASES),
              "release_timing_steps": macro("release_timing_steps", EVALUATION_PHASES)}
    for scope in sorted({r["demo_scope"] for r in rows}):
        guards[scope+"_precontact_joint_mae_deg"] = macro("joint_mae_deg", PRECONTACT_PHASES, scope)
    if primary is None or any(v is None for v in guards.values()):
        raise ValueError("selection requires preclose, transport, close and release coverage in every scope")
    diagnostics = {name+"_events": _event_metrics(states, targets, predictions, valid, closing=closing)
                   for name, closing in (("close", True), ("release", False))}
    return dict(primary=primary, primary_name=SELECTION_RULE["primary"], guards=guards,
                episode_phase_metrics=rows, diagnostics=diagnostics,
                physical_success_verified=False)


@contextmanager
def isolated_evaluation(policy):
    """Do not change training randomness, mode or the policy's inference queues."""
    import torch
    py_state, np_state, mode = random.getstate(), np.random.get_state(), policy.training
    queues = copy.deepcopy(policy._queues) if hasattr(policy, "_queues") else None
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state); policy.train(mode)
        if queues is not None:
            policy._queues = queues


def collect_action_chunks(policy, pre, post, dataset, observations, *, batch_size, seed, progress=None):
    """Copy compact numeric results into owned arrays, never retain batch storage.

    The caller controls model/global RNG state. This allows process-isolated,
    batch-aligned partitions to continue the same prediction RNG sequence.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    count = len(observations)
    if not count or batch_size < 1:
        raise ValueError("nonempty observations and positive batch size required")
    states = np.empty((count, 7), dtype=np.float32)
    targets = np.empty((count, 50, 7), dtype=np.float32)
    predictions = np.empty_like(targets)
    padding = np.empty((count, 50), dtype=np.bool_)
    losses = np.empty(count, dtype=np.float64)
    batches = DataLoader(Subset(dataset, [r["dataset_index"] for r in observations]),
        batch_size=batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(seed))
    seen = 0
    with torch.inference_mode():
        for raw in batches:
            loss, _ = policy(pre(raw), reduction="none")
            obs = {k: v for k, v in raw.items() if k.startswith("observation.") or k == "task"}
            chunk = post(policy.predict_action_chunk(pre(obs))).detach().float().cpu().numpy()
            size = len(raw["episode_index"])
            for i, identity in enumerate(observations[seen:seen+size]):
                if (int(raw["episode_index"][i]), int(raw["frame_index"][i])) != (
                        identity["episode_index"], identity["frame_index"]):
                    raise ValueError("validation observation identity/order changed")
            end = seen + size
            states[seen:end] = raw["observation.state"].numpy()
            targets[seen:end] = raw["action"].numpy()
            predictions[seen:end] = chunk
            padding[seen:end] = raw["action_is_pad"].numpy()
            losses[seen:end] = loss.detach().float().cpu().reshape(-1).numpy()
            seen = end
            if progress is not None:
                progress(seen, count)
    if seen != count or not all(np.isfinite(a).all() for a in (states, targets, predictions, losses)):
        raise ValueError("incomplete/nonfinite validation")
    return states, targets, predictions, padding, losses


def evaluate_monitor(policy, pre, post, dataset, observations, *, batch_size, seeds):
    """Predict fixed expert observations with paired noise; no optimizer updates."""
    import torch

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("distinct fixed validation seeds required")
    runs = []
    with isolated_evaluation(policy), torch.inference_mode():
        policy.eval()
        for seed in seeds:
            torch.manual_seed(seed)
            states, targets, predictions, padding, losses = collect_action_chunks(
                policy, pre, post, dataset, observations, batch_size=batch_size, seed=seed)
            result = score_action_chunks(observations, states, targets, predictions, padding)
            runs.append(dict(seed=seed, **result, validation_loss=float(np.mean(losses))))
    coverage = [(r["demo_scope"], r["scene_id"], r["source_index"], r["episode_index"], r["frame_index"], r["phase"])
                for r in observations]
    return dict(primary_name=SELECTION_RULE["primary"], primary=float(np.mean([r["primary"] for r in runs])),
                guards={k: float(np.mean([r["guards"][k] for r in runs])) for k in runs[0]["guards"]},
                validation_loss=float(np.mean([r["validation_loss"] for r in runs])),
                seeds=list(seeds), coverage=coverage, frames=len(observations), runs=runs,
                primary_seed_std=float(np.std([r["primary"] for r in runs])), physical_success_verified=False)


def selection_comparison(candidate, best):
    values = [candidate["primary"], *candidate["guards"].values()]
    if candidate["primary_name"] != SELECTION_RULE["primary"] or any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("nonfinite or incompatible selection metrics")
    blocked, limits = [], {}
    if best is not None:
        if any(candidate[k] != best[k] for k in ("primary_name", "seeds", "coverage")) or candidate["guards"].keys() != best["guards"].keys():
            raise ValueError("selection metric/seed/coverage changed")
        for key, value in candidate["guards"].items():
            reference = min(best["guards"][key], best.get("guard_reference", best["guards"])[key])
            tolerance = (SELECTION_RULE["event_timing_absolute_tolerance_steps"] if key.endswith("timing_steps")
                         else SELECTION_RULE["joint_guard_absolute_tolerance_deg"])
            limits[key] = max(reference * (1 + SELECTION_RULE["max_relative_guard_regression"]), reference + tolerance)
            if value > limits[key]:
                blocked.append(key)
    lower = best is None or candidate["primary"] < best["primary"] - SELECTION_RULE["minimum_improvement_deg"]
    return dict(improved=lower and not blocked, primary=candidate["primary"], lower_primary=lower,
                blocked_guards=blocked, guard_limits=limits, rule=SELECTION_RULE)


def publish_best_checkpoint(output, step, metrics, previous, save, verify):
    """Save + strict-verify + atomic pointer, then delete the preceding owned best."""
    output = Path(output)
    parent = output / "checkpoints"
    if output.is_symlink() or parent.is_symlink():
        raise ValueError("checkpoint directories must not be symlinks")
    parent.mkdir(exist_ok=True)
    path = parent / f"best_step_{step:06d}"
    temporary = path.with_name(path.name + ".incomplete")

    def owned(old):
        if (old.is_symlink() or old.parent != parent or not re.fullmatch(r"best_step_\d{6,}", old.name)
                or not old.is_dir() or old == path):
            raise ValueError("refusing deletion outside this run's preceding best")
        owner = json.loads((old / "checkpoint_owner.json").read_text())
        if owner != {"output": str(output.resolve()), "kind": "best", "step": int(old.name.rsplit('_', 1)[1])}:
            raise ValueError("checkpoint ownership mismatch")

    if previous is not None:
        previous = Path(previous); owned(previous)
        pointer = json.loads((output / "best_checkpoint.json").read_text())
        if pointer["path"] != str(previous.resolve()):
            raise ValueError("previous checkpoint is not this run's current best")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError("candidate checkpoint or unfinished save already exists")
    save(temporary, step)
    owner = {"output": str(output.resolve()), "kind": "best", "step": step}
    (temporary / "checkpoint_owner.json").write_text(json.dumps(owner, indent=2) + '\n')
    (temporary / "selection_metrics.json").write_text(json.dumps(metrics, allow_nan=False, indent=2) + '\n')
    checked = verify(temporary)
    if not checked.get("strict_reload") or not checked.get("state_dict_equal"):
        raise ValueError("new best failed strict reload/weight equality")
    (temporary / "reload_check.json").write_text(json.dumps(checked, indent=2) + '\n')
    temporary.rename(path)
    pointer = output / "best_checkpoint.json"
    staging = pointer.with_suffix('.incomplete')
    with staging.open('x') as stream:
        json.dump(dict(path=str(path.resolve()), optimizer_steps=step, primary=metrics["primary"],
                       primary_name=metrics["primary_name"], superseded_checkpoint=str(previous) if previous else None),
                  stream, allow_nan=False, indent=2)
    staging.replace(pointer)
    if previous is not None:
        owned(previous)
        shutil.rmtree(previous)
    return path
