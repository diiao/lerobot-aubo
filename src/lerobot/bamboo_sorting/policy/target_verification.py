"""Interface acceptance for prepared target annotations.

The LeRobot dataset is read through the real training/runtime views; no policy
is loaded and no training starts.
"""

import json

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from ..file_io import read_json as read, write_json as write


def verify_dataset(dataset, evidence, folder, spec):
    """Check every real sample through the existing training and runtime views.

    No policy is loaded. One decoded sample is cached only to avoid decoding it
    twice for the raw/transformed comparison; the real LeRobot loader is used.
    """
    import torch

    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_FIELDS
    from lerobot.bamboo_sorting.policy.joint_target_policy import (
        TargetInputSession, prepare_target_source, read_visible_mask,
    )
    from lerobot.bamboo_sorting.policy.joint_training import load_prepared_dataset

    torch.set_num_threads(2)
    capture_check = read(evidence / "capture_check.json")
    if not capture_check["raw_contract_and_accepted_command_audit_passed"]:
        raise ValueError("existing raw capture audit must have passed")
    selected_episodes = spec["episodes"]
    stack_audit = None
    if any("pick_step" in ep for ep in selected_episodes):
        from lerobot.bamboo_sorting.capture.aubo_joint_audit import audit_joint_dataset
        stack_audit = audit_joint_dataset(dataset, evidence, source_dataset_root=spec["source_dataset_root"])
        selected_episodes = [ep for ep in selected_episodes
                             if ep["episode_index"] in stack_audit["supervised_candidate_episodes"]]
        if not selected_episodes:
            raise ValueError("no successful target demonstrations available for interface acceptance")
    record = {"root": str(dataset), "source_dataset_root": spec["source_dataset_root"],
              "evidence_root": str(evidence),
              "episodes": [e["episode_index"] for e in selected_episodes],
              "frames": sum(e["total_frames"] for e in selected_episodes)}
    prepared = prepare_target_source(record, folder / "annotations.json")
    view = load_prepared_dataset(prepared)

    class CurrentSample:
        def __init__(self, source):
            self.source, self.index, self.value = source, None, None

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            if self.index != index:
                self.value, self.index = self.source[index], index
            return self.value

    view.dataset = CurrentSample(view.dataset)
    captures = {e["episode_index"]: e for e in read(evidence / "session.json")["attempts"]
                if e["status"] == "saved_pending_finalize"}
    data = pq.read_table(sorted((dataset / "data").glob("chunk-*/*.parquet")),
                         columns=["episode_index", "frame_index", "observation.state", "action"]).to_pylist()
    by_ep = {ep: sorted([r for r in data if r["episode_index"] == ep], key=lambda r: r["frame_index"])
             for ep in record["episodes"]}
    sessions = {e["episode_index"]: TargetInputSession(e["target_id"]) for e in spec["episodes"]}
    counts = {ep: {"episode_index": ep, "frames": 0, "approach": 0, "transport": 0,
                   "padded_chunks": 0} for ep in record["episodes"]}
    labels = {(e["episode_index"], f["frame_index"]): f for e in spec["episodes"] for f in e["frames"]}
    for index in range(len(view)):
        raw = view.dataset[index]
        ep, fi = int(raw["episode_index"]), int(raw["frame_index"])
        frame = view.frames[(ep, fi)]
        output = view[index]
        for key, value in raw.items():
            if key == "observation.images.global_rgb":
                if key in output:
                    raise ValueError("unprocessed global image leaked into target view")
                continue
            if isinstance(value, torch.Tensor):
                if not torch.equal(value, output[key]):
                    raise ValueError(f"modified raw field: {ep}/{fi} {key}")
            elif value != output[key]:
                raise ValueError(f"modified raw metadata: {ep}/{fi} {key}")
        expected_state = torch.tensor(by_ep[ep][fi]["observation.state"], dtype=raw["observation.state"].dtype)
        expected_indices = np.minimum(fi + np.arange(50), len(by_ep[ep]) - 1)
        expected_actions = torch.tensor([by_ep[ep][int(i)]["action"] for i in expected_indices], dtype=raw["action"].dtype)
        expected_pad = torch.from_numpy(fi + np.arange(50) >= len(by_ep[ep]))
        if (not torch.equal(raw["observation.state"], expected_state)
                or not torch.equal(raw["action"], expected_actions)
                or not torch.equal(raw["action_is_pad"], expected_pad)):
            raise ValueError(f"source state/action/chunk padding mismatch: {ep}/{fi}")
        images = {name: raw[f"observation.images.{name}"].mul(255).round().to(torch.uint8)
                  .permute(1, 2, 0).numpy() for name in ("global_rgb", "grasp_rgb")}
        annotation = frame["annotation"]
        mask = read_visible_mask(annotation["mask_path"]) if annotation else None
        if annotation is not None:
            source = np.array(Image.open(folder / labels[(ep, fi)]["image"]).convert("RGB"))
            if not np.array_equal(source, images["global_rgb"]):
                raise ValueError(f"annotation image identity mismatch: {ep}/{fi}")
        clock = captures[ep]["sensor_and_suction_evidence"]["timestamp_journal"]["frames"][fi]
        runtime = sessions[ep].prepare(
            {"observation.state": raw["observation.state"].numpy(), "task": raw["task"],
             **{f"observation.images.{k}": v for k, v in images.items()}},
            frame_id=frame["frame_id"],
            global_timestamp=clock["timestamps"]["global_rgb"]["host_receive_monotonic_s"],
            visible_mask=mask, annotation=annotation)
        phase = "transport" if frame["transport"] else "approach"
        if runtime["target_input"]["phase"] != phase:
            raise ValueError("training/runtime phase disagreement")
        for key in ("target_global_rgb", "grasp_rgb"):
            actual = output[f"observation.images.{key}"].mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            if not np.array_equal(actual, runtime[f"observation.images.{key}"]):
                raise ValueError(f"training/runtime image mismatch: {ep}/{fi} {key}")
        action = captures[ep]["joint_frames"][fi]
        sessions[ep].accept_command(dict(zip(JOINT_FIELDS, action["action"], strict=True)),
                                   accepted_monotonic_s=action["joint_command_trace"]["accepted_monotonic_s"])
        counts[ep]["frames"] += 1
        counts[ep][phase] += 1
        counts[ep]["padded_chunks"] += int(expected_pad.any())
        if index % 100 == 0:
            print(f"checked {index + 1}/{len(view)}: episode {ep}, frame {fi}", flush=True)
    for e in capture_check["episodes"]:
        if e["episode_index"] not in counts:
            continue
        if counts[e["episode_index"]]["approach"] != e["approach_frames"]:
            raise ValueError("phase count differs from capture evidence")
    result = {"schema": "aubo_joint_target_dataset_acceptance", "passed": True,
              "episodes": list(counts.values()), "frames": len(view),
              "excluded_episode_indices": [ep["episode_index"] for ep in spec["episodes"]
                                           if ep["episode_index"] not in counts],
              "stack_summary": stack_audit["stack_summary"] if stack_audit else None,
              "source_dataset_root": str(dataset), "annotation_path": str(folder / "annotations.json"),
              "prepare_target_source_passed": True, "real_lerobot_dataset_backend": "pyav",
              "all_approach_images_match_annotation_source": True,
              "all_states_actions_and_50_step_padding_unchanged": True,
              "all_training_runtime_images_and_phases_equal": True,
              "user_review": spec.get("human_review"), "pixel_exact_ground_truth_claimed": False,
              "training_started": False, "model_loaded": False, "hardware_access": False,
              "independent_validation_available": False}
    write(folder / "dataset_check.json", result)
    spec["target_source_ready"] = True
    spec["target_training_ready"] = False
    spec["training_readiness_note"] = "source interface accepted; independent validation source and training authorization still required"
    write(folder / "annotations.json", spec)
    print(json.dumps(result, ensure_ascii=False, indent=2))
