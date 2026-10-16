"""Stack capture metadata and human outcomes, shared by capture and audit.

These fields describe operator observations, never robot IO proof. Legacy
captures without stack fields remain valid and are not silently relabeled.
"""

STACK_FIELDS = ("pick_step", "available_target_count")
MAX_STACK_PICK_STEP = 6
OUTCOME_FIELDS = ("target_matched", "held", "single_strip", "delivered", "neighbor_disturbed")
HUMAN_OUTCOMES = ("single_success", "empty", "multi_pick", "slip", "blocked", "uncertain")


def stack_metadata(metadata):
    if not any(key in metadata for key in STACK_FIELDS):
        return {}
    if not isinstance(metadata.get("target_id"), str) or not metadata["target_id"].strip():
        raise ValueError("stack capture requires target_id")
    for key in STACK_FIELDS:
        maximum = MAX_STACK_PICK_STEP if key == "pick_step" else 3
        if type(metadata.get(key)) is not int or not 1 <= metadata[key] <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}")
    return {key: metadata[key] for key in STACK_FIELDS}


def validate_stack_sequence(records):
    """Allow partial piles and failed attempts; prohibit duplicate saved steps.

pick_step numbers recorded attempts, not inferred successful removals. A retry
can use the same target_id at the next step; neither success nor remaining
object count is inferred from the step number.
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


def stack_summary(episodes):
    rows = [e for e in episodes if stack_metadata(e)]
    return {"episodes": len(rows), "scenes": len({e["scene_id"] for e in rows}),
            "target_choice_eligible_episodes": [e["episode_index"] for e in rows if e["available_target_count"] >= 2],
            "grasp_only_episodes": [e["episode_index"] for e in rows if e["available_target_count"] == 1],
            "unknown_outcome_episodes": [e["episode_index"] for e in rows
                                         if any(e.get("target_outcome", {}).get(k) is None for k in OUTCOME_FIELDS[:4])],
            "uniform_random_reference_only": True}
