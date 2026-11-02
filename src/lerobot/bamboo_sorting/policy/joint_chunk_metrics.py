"""Offline action-block metrics and nominal geometry; no control or hardware APIs."""

import numpy as np


class EvaluationGeometry:
    """Frozen diagnostic transform, exported from an already checked URDF fit."""

    def __init__(self, record):
        if record["purpose"] != "offline evaluation only" or len(record["joints"]) != 6:
            raise ValueError("six-joint offline evaluation geometry required")
        self.joints = [(np.asarray(j["origin"], dtype=float), np.asarray(j["axis"], dtype=float))
                       for j in record["joints"]]
        self.tool = np.asarray(record["tool_offset_m"], dtype=float)
        if self.tool.shape != (3,) or not np.isfinite(self.tool).all():
            raise ValueError("finite three-dimensional tool offset required")
        for origin, axis in self.joints:
            if origin.shape != (4, 4) or axis.shape != (3,) or not np.isfinite(origin).all() or not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError("finite homogeneous origins and unit axes required")

    def __call__(self, values):
        return self.pose(values)[0]

    def pose(self, values):
        """Nominal TCP position and rotation; not a control calibration."""
        values = np.asarray(values, dtype=float)
        shape = values.shape[:-1]
        if values.shape[-1] < 6 or not np.isfinite(values).all():
            raise ValueError("finite absolute joint angles required")
        transform = np.broadcast_to(np.eye(4), (*shape, 4, 4)).copy()
        for index, (origin, axis) in enumerate(self.joints):
            x, y, z = axis
            cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            angle = np.deg2rad(values[..., index])[..., None, None]
            turn = np.broadcast_to(np.eye(4), (*shape, 4, 4)).copy()
            turn[..., :3, :3] = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)
            transform = transform @ origin @ turn
        rotation = transform[..., :3, :3]
        return (rotation @ self.tool[..., None])[..., 0] + transform[..., :3, 3], rotation


def _event_metrics(states, targets, predictions, valid, *, closing):
    eligible = states[:, 6] < 50 if closing else states[:, 6] >= 50
    missing = false_events = expected = extra_switches = 0
    errors, penalized = [], []
    for state, truth, pred, mask in zip(states[eligible], targets[eligible], predictions[eligible], valid[eligible], strict=True):
        initial = state[6] >= 50
        def event(values):
            active = np.r_[initial, values[mask, 6] >= 50]
            transitions = np.flatnonzero(np.diff(active.astype(int)) == (1 if closing else -1))
            return int(transitions[0]) if len(transitions) else None
        actual, predicted = event(truth), event(pred)
        def switches(values):
            active = np.r_[initial, values[mask, 6] >= 50]
            return int(np.count_nonzero(np.diff(active.astype(int))))
        extra_switches += max(0, switches(pred)-switches(truth))
        if actual is None:
            false_events += int(predicted is not None)
        else:
            expected += 1
            if predicted is None:
                missing += 1
                # A missed transition cannot improve the reported timing mean.
                penalized.append(int(mask.sum()))
            else:
                errors.append(abs(predicted-actual))
                penalized.append(abs(predicted-actual))
    return dict(expected=expected, missing=missing, false_events=false_events,
                mean_absolute_timing_error_steps=float(np.mean(errors)) if errors else None,
                timing_error_with_misses_steps=float(np.mean(penalized)) if penalized else None,
                eligible_windows=int(eligible.sum()),
                failure_rate=float((missing+false_events)/eligible.sum()) if eligible.any() else None,
                extra_switches_per_window=float(extra_switches/eligible.sum()) if eligible.any() else None,
                event_errors_per_window=float((missing+false_events+extra_switches)/eligible.sum()) if eligible.any() else None,
                within_2_steps=int(sum(error <= 2 for error in errors)))


def action_chunk_metrics(states, targets, predictions, action_is_pad, stats, *, geometry=None, motion_onset=None):
    """Score real rows 8..19 and all real rows; never count padded action tails."""
    states, targets, predictions = [np.asarray(a) for a in (states, targets, predictions)]
    padding = np.asarray(action_is_pad)
    if (targets.ndim != 3 or targets.shape[-1] != 7 or targets.shape[1] < 20
            or predictions.shape != targets.shape or states.shape != (len(targets), 7)
            or padding.dtype != np.bool_ or padding.shape != targets.shape[:2]
            or not all(np.isfinite(a).all() for a in (states, targets, predictions))):
        raise ValueError("matching finite [N,T>=20,7] blocks, [N,7] state and boolean padding required")
    valid = ~padding
    window = valid.copy()
    window[:, :8] = False
    window[:, 20:] = False
    if not window.any():
        raise ValueError("evaluation has no real execution-window actions")
    span = np.asarray(stats["action"]["q99"]) - stats["action"]["q01"]
    if span.shape != (7,) or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("finite nondegenerate action quantiles required")
    absolute = np.abs(predictions-targets)
    result = dict(execution_window_rows=[8, 19], execution_window_valid_steps=int(window.sum()),
        execution_window_normalized_action_mae=float((2*absolute/span)[window].mean()),
        execution_window_joint_mae_deg=float(absolute[..., :6][window].mean()),
        execution_window_joint_mae_by_dimension_deg=absolute[..., :6][window].mean(0).tolist(),
        execution_window_joint_p95_deg=float(np.quantile(absolute[..., :6][window], .95)),
        execution_window_normalized_joint_mae=float((2*absolute[..., :6]/span[:6])[window].mean()),
        execution_window_gripper_mae=float(absolute[..., 6][window].mean()),
        full_chunk_normalized_action_mae=float((2*absolute/span)[valid].mean()),
        full_chunk_joint_mae_deg=float(absolute[..., :6][valid].mean()),
        close_events=_event_metrics(states, targets, predictions, valid, closing=True),
        release_events=_event_metrics(states, targets, predictions, valid, closing=False),
        execution_window_tcp_error_mm=None, execution_window_tcp_p95_mm=None,
        execution_window_orientation_error_deg=None, execution_window_orientation_p95_deg=None,
        onset_observations=0,
        onset_expert_lift_mm=None, onset_predicted_lift_mm=None,
        physical_success_verified=False, target_obedience_verified=False)
    if geometry is not None:
        truth_xyz, truth_rotation = geometry.pose(targets)
        pred_xyz, pred_rotation = geometry.pose(predictions)
        start_xyz = geometry(states)
        tcp_error = np.linalg.norm(pred_xyz-truth_xyz, axis=-1)[window]*1000
        relative_rotation = np.swapaxes(truth_rotation, -1, -2) @ pred_rotation
        angle = np.rad2deg(np.arccos(np.clip((np.trace(relative_rotation, axis1=-2, axis2=-1)-1)/2, -1, 1)))[window]
        result.update(execution_window_tcp_error_mm=float(tcp_error.mean()),
                      execution_window_tcp_p95_mm=float(np.quantile(tcp_error, .95)),
                      execution_window_orientation_error_deg=float(angle.mean()),
                      execution_window_orientation_p95_deg=float(np.quantile(angle, .95)))
        onset = np.zeros(len(states), dtype=bool) if motion_onset is None else np.asarray(motion_onset, dtype=bool)
        if onset.shape != (len(states),):
            raise ValueError("one motion-onset flag required per observation")
        selected = window & onset[:, None]
        result["onset_observations"] = int(onset.sum())
        if selected.any():
            result["onset_expert_lift_mm"] = float((truth_xyz[..., 2]-start_xyz[:, None, 2])[selected].mean()*1000)
            result["onset_predicted_lift_mm"] = float((pred_xyz[..., 2]-start_xyz[:, None, 2])[selected].mean()*1000)
    return result


EVALUATION_PHASES = ("initial_hold", "onset", "approach", "preclose", "transport", "release")
PRECONTACT_PHASES = ("onset", "approach", "preclose")


def evaluation_phase(frame, events, *, fps=25):
    """Exclusive observation phases from expert command events, not predictions."""
    onset, close, release = (events.get(key) for key in ("onset", "close", "release"))
    if release is not None and frame >= release-fps:
        return "release"
    # The close action is accepted AFTER its observation is prepared.
    if close is not None and frame > close:
        return "transport"
    if close is not None and frame >= close-fps:
        return "preclose"
    if onset is None or frame < onset:
        return "initial_hold"
    return "onset" if frame < onset+fps else "approach"


def overlapping_chunk_metrics(predictions, padding, observations):
    """Compare plans for the SAME future frames. Not executed jerk or stability."""
    keys = list(zip(observations["source_index"], observations["episode_index"], observations["frame_index"], strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate observation identities in overlap evaluation")
    phases = observations["phase"]
    errors, switches, episode_errors = [], [], {}
    ordered = sorted(range(len(keys)), key=lambda i: keys[i])
    pairs = 0
    for a, b in zip(ordered[:-1], ordered[1:], strict=True):
        if keys[a][:2] != keys[b][:2] or phases[a] != phases[b]:
            continue
        delta = int(keys[b][2]-keys[a][2])
        if not 0 < delta < predictions.shape[1]:
            continue
        valid = ~padding[a, delta:] & ~padding[b, :-delta]
        if not valid.any():
            continue
        left, right = predictions[a, delta:][valid], predictions[b, :-delta][valid]
        difference = np.abs(left[:, :6]-right[:, :6])
        errors.extend(difference.ravel())
        switches.extend((left[:, 6] >= 50) != (right[:, 6] >= 50))
        episode_errors.setdefault(keys[a][:2], []).append(float(difference.mean()))
        pairs += 1
    return {"pairs": pairs, "joint_mae_deg": float(np.mean(errors)) if errors else None,
            "joint_p95_deg": float(np.quantile(errors, .95)) if errors else None,
            "episode_macro_joint_mae_deg": float(np.mean([np.mean(v) for v in episode_errors.values()])) if episode_errors else None,
            "gripper_disagreement_rate": float(np.mean(switches)) if switches else None,
            "interpretation": "same-time plan disagreement on expert observations; not measured robot jerk"}


def grouped_chunk_metrics(states, targets, predictions, padding, stats, observations, *, geometry):
    """Equal weight per episode within a phase, then per pre-contact phase.

    Position ranks candidates; separate guards prevent gripper improvements
    from hiding movement regressions. This is an offline shortlist only.
    """
    states, targets, predictions, padding = map(np.asarray, (states, targets, predictions, padding))
    meta = {key: np.asarray(observations[key]) for key in ("source_index", "episode_index", "frame_index", "phase")}
    if any(values.shape != (len(states),) for values in meta.values()) or not np.isin(meta["phase"], EVALUATION_PHASES).all():
        raise ValueError("one valid phase and source/episode/frame identity required per observation")
    valid_observation = (~padding[:, 8:20]).any(axis=1)
    groups = sorted(set(zip(meta["source_index"], meta["episode_index"], meta["phase"], strict=True)))
    records = []
    for source, episode, phase in groups:
        selected = ((meta["source_index"] == source) & (meta["episode_index"] == episode)
                    & (meta["phase"] == phase) & valid_observation)
        if not selected.any():
            continue
        metrics = action_chunk_metrics(states[selected], targets[selected], predictions[selected], padding[selected], stats, geometry=geometry)
        row = {"source_index": int(source), "episode_index": int(episode), "phase": str(phase),
               "observations": int(selected.sum()), "joint_mae_deg": metrics["execution_window_joint_mae_deg"],
               "tcp_error_mm": metrics["execution_window_tcp_error_mm"],
               "tcp_p95_mm": metrics["execution_window_tcp_p95_mm"],
               "orientation_error_deg": metrics["execution_window_orientation_error_deg"]}
        for name in ("close", "release"):
            events = metrics[f"{name}_events"]
            row.update({f"{name}_failure_rate": events["failure_rate"],
                        f"{name}_extra_switches_per_window": events["extra_switches_per_window"],
                        f"{name}_event_errors_per_window": events["event_errors_per_window"],
                        f"{name}_timing_with_misses_steps": events["timing_error_with_misses_steps"]})
        records.append(row)
    fields = ("joint_mae_deg", "tcp_error_mm", "tcp_p95_mm", "orientation_error_deg",
              "close_failure_rate", "release_failure_rate", "close_extra_switches_per_window", "release_extra_switches_per_window",
              "close_event_errors_per_window", "release_event_errors_per_window",
              "close_timing_with_misses_steps", "release_timing_with_misses_steps")
    phases = {}
    for phase in EVALUATION_PHASES:
        rows = [r for r in records if r["phase"] == phase]
        if not rows:
            continue
        summary = {"episodes": len(rows), "observations": sum(r["observations"] for r in rows)}
        for field in fields:
            values = [r[field] for r in rows if r[field] is not None]
            summary[field] = float(np.mean(values)) if values else None
            summary[f"worst_episode_{field}"] = float(max(values)) if values else None
        phases[phase] = summary
    overlap = overlapping_chunk_metrics(predictions, padding, meta)
    precontact = [phases[p] for p in PRECONTACT_PHASES if p in phases]
    selection = None
    if geometry is not None and "preclose" in phases and precontact:
        def phase_value(phase, field):
            return phases.get(phase, {}).get(field)
        guards = {"precontact_joint_mae_deg": float(np.mean([p["joint_mae_deg"] for p in precontact])),
                  "preclose_orientation_error_deg": phase_value("preclose", "orientation_error_deg"),
                  "preclose_worst_episode_tcp_p95_mm": phase_value("preclose", "worst_episode_tcp_p95_mm"),
                  "transport_tcp_error_mm": phase_value("transport", "tcp_error_mm"),
                  # A policy that never switches has zero extra switches but misses
                  # every event. Count misses/false events/extra switches together,
                  # so its zero cannot veto learning to close and release.
                  "close_event_errors_per_window": phase_value("preclose", "close_event_errors_per_window"),
                  "close_timing_with_misses_steps": phase_value("preclose", "close_timing_with_misses_steps"),
                  "release_event_errors_per_window": phase_value("release", "release_event_errors_per_window"),
                  "release_timing_with_misses_steps": phase_value("release", "release_timing_with_misses_steps"),
                  "overlap_joint_mae_deg": overlap["episode_macro_joint_mae_deg"]}
        selection = {"primary": float(np.mean([p["tcp_error_mm"] for p in precontact])),
                     "primary_name": "episode_phase_macro_precontact_tcp_error_mm",
                     "guards": guards, "available_precontact_phases": [p for p in PRECONTACT_PHASES if p in phases],
                     "scope": "offline candidate; not target obedience or physical grasp success"}
    return {"episode_phase_metrics": records, "phase_metrics": phases, "overlapping_chunks": overlap,
            "phase_definition": "expert observation time; scored future rows 8..19 may cross an event",
            "offline_selection": selection}
