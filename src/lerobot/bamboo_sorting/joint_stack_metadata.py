"""Stack capture metadata and human outcomes, shared by capture and audit.

These fields describe operator observations, never robot IO proof. Legacy
captures without stack fields remain valid and are not silently relabeled.
"""

import json
from pathlib import Path

STACK_FIELDS = ("pick_step", "available_target_count")
MAX_STACK_TARGET_COUNT = 6
OUTCOME_FIELDS = ("target_matched", "held", "single_strip", "delivered", "neighbor_disturbed")
HUMAN_OUTCOMES = ("single_success", "empty", "multi_pick", "slip", "blocked", "uncertain")


def stack_metadata(metadata):
    if not any(key in metadata for key in STACK_FIELDS):
        return {}
    if not isinstance(metadata.get("target_id"), str) or not metadata["target_id"].strip():
        raise ValueError("stack capture requires target_id")
    for key in STACK_FIELDS:
        value = metadata.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        if key == "available_target_count" and value > MAX_STACK_TARGET_COUNT:
            raise ValueError(f"{key} must be an integer from 1 to {MAX_STACK_TARGET_COUNT}")
    return {key: metadata[key] for key in STACK_FIELDS}


def reviewed_stack_metadata(dataset_root, evidence_root, episode_index, metadata):
    """Read confirmed count overlays for offline consumers; never rewrite capture.

    Corrections already written to session/sidecars need no second overlay.
    Dataset and episode identify a correction; scene/target/step guard identity.
    """
    stack = stack_metadata(metadata)
    path = Path(evidence_root) / "recording_review/metadata_corrections.json"
    if not stack or not path.exists():
        return stack
    report = json.loads(path.read_text())
    if report.get("schema") != "aubo_joint_operator_confirmed_metadata_corrections":
        raise ValueError("unsupported stack metadata corrections")
    applied = []
    for correction in report["corrections"]:
        if correction["episode_index"] != episode_index:
            continue
        if correction.get("status") == "applied_to_saved_sidecar_and_session":
            continue
        if Path(correction["dataset_root"]).resolve() != Path(dataset_root).resolve():
            continue
        if (correction.get("status") != "operator_confirmed"
                or correction.get("field") != "available_target_count"
                or any(correction.get(key) != metadata.get(key)
                       for key in ("scene_id", "target_id", "pick_step"))
                or metadata["available_target_count"] != correction["original_value"]
                or applied):
            raise ValueError("stack metadata correction identity/value mismatch or duplicate")
        stack["available_target_count"] = correction["confirmed_value"]
        stack_metadata({**metadata, **stack})
        # Keep the declared source identity when evidence is staged remotely.
        # A relocation changes the reader's path, not the correction's origin.
        origin = (Path(correction["source_record"]).parent / "recording_review/metadata_corrections.json"
                  if correction.get("source_record") else path.resolve())
        applied.append({**correction, "correction_file": str(origin)})
    if applied:
        stack["metadata_corrections"] = applied
    return stack


def validate_stack_sequence(records):
    """Allow partial piles and failed attempts; prohibit duplicate saved steps.

pick_step numbers recorded attempts, not inferred successful removals. A retry
can use the same target_id at the next step; neither success nor remaining
object count is inferred from the step number. It has no object-count cap.
"""
    scenes = {}
    for record in records:
        metadata = record["metadata"]
        if not stack_metadata(metadata):
            continue
        scene = metadata["scene_id"].strip()
        prior = scenes.setdefault(scene, [])
        if prior and (metadata.get("split") != prior[0].get("split")
                      or metadata["pick_step"] <= prior[-1]["pick_step"]):
            raise ValueError("stack steps must increase within one scene and split")
        prior.append(metadata)


def validate_target_outcome(value):
    if not isinstance(value, dict) or any(key not in value for key in OUTCOME_FIELDS):
        raise ValueError("stack capture requires separate human target outcomes")
    if any(value[key] is not None and type(value[key]) is not bool for key in OUTCOME_FIELDS):
        raise ValueError("human target outcomes must be true, false, or unknown")
    return {key: value[key] for key in OUTCOME_FIELDS}


def successful_target_outcome(value):
    return isinstance(value, dict) and all(value.get(key) is True for key in OUTCOME_FIELDS[:4])


def confirmed_target_outcome(human_outcome):
    """Derive target fields from the operator's single on-site result.

Stack capture no longer asks each check: single_success means the operator saw
the specified strip held alone and released in the placement area. Failures
keep their category only in human_outcome; neighbor disturbance is not recorded.
"""
    confirmed = True if human_outcome == "single_success" else None
    return {**dict.fromkeys(OUTCOME_FIELDS[:4], confirmed), "neighbor_disturbed": None}


def stack_summary(episodes):
    rows = [e for e in episodes if stack_metadata(e)]
    return {"episodes": len(rows), "scenes": len({e["scene_id"] for e in rows}),
            "target_choice_eligible_episodes": [e["episode_index"] for e in rows if e["available_target_count"] >= 2],
            "grasp_only_episodes": [e["episode_index"] for e in rows if e["available_target_count"] == 1],
            # A known failure category is not an unknown outcome, even without target fields.
            "unknown_outcome_episodes": [e["episode_index"] for e in rows if e.get("human_outcome") == "uncertain"
                                         or (e.get("human_outcome") == "single_success" and any(
                                             e.get("target_outcome", {}).get(k) is None for k in OUTCOME_FIELDS[:4]))],
            "uniform_random_reference_only": True}
