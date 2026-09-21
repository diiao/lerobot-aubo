"""Read-only audit of finalized joint recordings against command sidecars."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .aubo_joint_contract import JOINT_FIELDS, JOINT_NAMES, JOINT_TASK, expand_joint_command, joint_contract_record, require_joint_dataset


def audit_joint_dataset(dataset_root, evidence_root):
    root, evidence = Path(dataset_root).resolve(), Path(evidence_root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    contract = json.loads((root / "meta/aubo_joint_contract.json").read_text())
    require_joint_dataset(info["features"], contract, info["fps"])
    if info["robot_type"] != "aubo_i10":
        raise ValueError("joint dataset must identify AUBO i10")
    session = json.loads((evidence / "session.json").read_text())
    if (session.get("dataset_root") != str(root) or session.get("dataset_finalized") is not True
            or session.get("contract") != joint_contract_record()):
        raise ValueError("matching finalized joint session is required")
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError("joint dataset has no parquet data")
    data = pq.read_table(files).to_pydict()
    states, actions = (np.asarray(data[k], dtype=np.float32) for k in ("observation.state", "action"))
    ep, frame = (np.asarray(data[k]) for k in ("episode_index", "frame_index"))
    if (states.shape != (len(ep), 6) or actions.shape != (len(ep), 6)
            or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError("joint state/action arrays must be finite [frames, 6]")
    if not np.isin(states[:, 5], [0, 100]).all() or not np.isin(actions[:, 5], [0, 100]).all():
        raise ValueError("invalid suction values")
    count = session["saved_episode_count"]
    if (count < 1 or count != info["total_episodes"] or len(ep) != info["total_frames"]
            or not np.array_equal(np.unique(ep), np.arange(count))):
        raise ValueError("joint dataset frame/episode counts mismatch")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    if set(tasks.get("__index_level_0__", tasks.get("task", []))) != {JOINT_TASK}:
        raise ValueError("joint dataset canonical task mismatch")
    task_ids = tasks.get("task_index", [])
    if not task_ids or not np.isin(data["task_index"], task_ids).all():
        raise ValueError("joint frame task index mismatch")
    saved = [a for a in session["attempts"] if a["status"] == "saved_pending_finalize"]
    if [a["episode_index"] for a in saved] != list(range(count)):
        raise ValueError("joint sidecar saved episode sequence mismatch")
    result = []
    for index, record in enumerate(saved):
        sidecar = json.loads((evidence / f"episode-{index:04d}.json").read_text())
        if sidecar != record or sidecar["dataset_root"] != str(root):
            raise ValueError("joint episode sidecar differs from finalized session")
        mask = ep == index
        if not np.array_equal(frame[mask], np.arange(mask.sum())):
            raise ValueError("joint episode frame indices are not contiguous")
        records = record["joint_frames"]
        expected = np.asarray([r["action"] for r in records], dtype=np.float32)
        if not np.array_equal(actions[mask], expected):
            raise ValueError("saved joint labels differ from accepted command evidence")
        for i, r in enumerate(records):
            trace = r["joint_command_trace"]
            full = expand_joint_command(dict(zip(JOINT_FIELDS, r["action"], strict=True)))
            joints = [full[k] for k in JOINT_NAMES]
            if (r["frame_index"] != i or trace["return_code"] != 0
                    or trace["sequence"] != r["sequence_before"] + 1
                    or trace["joint_target_deg"] != joints
                    or not np.allclose(trace["sdk_target_rad"], np.deg2rad(joints), rtol=0, atol=1e-12)):
                raise ValueError("accepted joint command trace mismatch")
        shared = record["sensor_and_suction_evidence"]
        timestamps = shared["timestamp_journal"]["frames"]
        cycles = shared["gripper_journal"]["cycles"]
        if len(timestamps) != len(records) or len(cycles) != len(records):
            raise ValueError("sensor/suction evidence count mismatch")
        for stream in ("robot_state", "global_rgb", "grasp_rgb"):
            values = [r["timestamps"][stream]["host_receive_monotonic_s"] for r in timestamps]
            if not np.isfinite(values).all() or (np.diff(values) <= 0).any():
                raise ValueError("sensor timestamps must be finite and increasing")
        if not np.array_equal(states[mask][1:, 5], actions[mask][:-1, 5]):
            raise ValueError("suction history does not equal preceding command")
        for cycle, a in zip(cycles, actions[mask], strict=True):
            if (cycle["dataset_action_gripper_pos"] != a[5] or cycle["action_consistency"] != "matched"
                    or not cycle["send_action_returned"]):
                raise ValueError("suction command evidence mismatch")
        result.append({"episode_index": index, "frames": int(mask.sum()),
                       **record["metadata"], "human_outcome": record["human_outcome"]})
    return {"schema_version": contract["schema_version"], "audit_passed": True,
            "episodes": result, "frames": len(ep), "capture_complete": session["capture_complete"],
            "video_decode_checked": False, "physical_success_verified": False,
            "training_authorized": False, "policy_execution_authorized": False}
