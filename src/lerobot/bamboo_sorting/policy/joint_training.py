"""Offline preparation for full-joint AUBO SmolVLA. No hardware imports."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..capture.aubo_joint_audit import audit_joint_dataset
from ..aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, joint_contract_record
from ..joint_gripper_quality import require_joint_normalization_stats


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def fingerprint(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".parquet", ".mp4"}:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def prepare_training_data(train_root, train_evidence, validation_root, validation_evidence,
                          *, train_source_root=None, validation_source_root=None):
    roots = [Path(p).resolve() for p in (train_root, validation_root)]
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise ValueError("training and validation datasets must have independent roots")
    records = []
    for root, evidence, split, source in zip(roots, (train_evidence, validation_evidence), ("train", "validation"),
                                           (train_source_root, validation_source_root), strict=True):
        # This existing trainer constructs the original dual-640x480 adapter.
        # Native-resolution recordings need matched model preprocessing first.
        report = audit_joint_dataset(root, evidence, source_dataset_root=source)
        if any(tuple(shape) != (480, 640, 3) for shape in report.get("image_shapes", {}).values()):
            raise ValueError("this trainer uses dual 640x480 inputs; configure native-resolution model "
                             "preprocessing before training new camera recordings; keep originals unchanged")
        if not report["gripper_quality_passed"]:
            raise ValueError(f"{split} gripper quality failed: {report['gripper_quality_issues']}")
        selected = report["supervised_candidate_episodes"]
        episodes = [r for r in report["episodes"] if r["episode_index"] in selected]
        if not episodes or any(r.get("split") != split for r in episodes):
            raise ValueError(f"{split} requires successful episodes explicitly tagged with that split")
        # A failed earlier step can perturb the later successful sample in the
        # same pile. Group all stack records, including excluded outcomes.
        grouped = [r for r in report["episodes"] if "pick_step" in r]
        if any(r.get("split") != split for r in grouped):
            raise ValueError(f"{split} stack scenes must stay in one split")
        scenes = sorted({r["scene_id"].strip() for r in [*episodes, *grouped]})
        if not all(scenes):
            raise ValueError("nonempty scene identifiers are required")
        records.append({"root": str(root), "evidence_root": str(Path(evidence).resolve()),
                        "source_dataset_root": report["source_dataset_root"],
                        "episodes": selected, "scenes": scenes,
                        "stack_summary": report.get("stack_summary", {}),
                        "frames": sum(r["frames"] for r in episodes),
                        "capture_complete": report["capture_complete"]})
    if set(records[0]["scenes"]) & set(records[1]["scenes"]):
        raise ValueError("scene leakage between training and validation")
    # Do not reuse whole-dataset metadata stats after excluding failed episodes.
    data = pq.read_table(sorted((roots[0] / "data").glob("chunk-*/*.parquet")),
                         columns=["episode_index", "observation.state", "action"]).to_pydict()
    selected = np.isin(data["episode_index"], records[0]["episodes"])
    stats = {}
    for key in ("observation.state", "action"):
        values = np.asarray(data[key], dtype=np.float64)[selected]
        stats[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                      "min": values.min(0).tolist(), "max": values.max(0).tolist()}
    require_joint_normalization_stats(stats)
    for record in records:
        record["dataset_sha256"] = fingerprint(record["root"])
        record["evidence_sha256"] = fingerprint(record["evidence_root"])
    return {"contract": joint_contract_record(), "train": records[0], "validation": records[1],
            "train_only_stats": stats, "normalization_scope": "selected_successful_train_episodes_only",
            "physical_success_verified": False, "policy_execution_authorized": False}


def read_data_manifest(path):
    """Resolve an explicit list of original captures without rewriting evidence."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema_version") != "AuboJointTrainingSourcesV1":
        raise ValueError("unsupported training manifest")
    if not isinstance(value.get("train"), list) or not value["train"]:
        raise ValueError("manifest requires nonempty train sources")
    def resolve(item):
        if not isinstance(item, dict) or not {"root", "evidence_root"} <= item.keys():
            raise ValueError("manifest source requires root and evidence_root")
        result = {}
        for key in ("root", "evidence_root"):
            p = Path(item[key])
            result[key] = str((path.parent / p).resolve())
        source = item.get("source_dataset_root", result["root"])
        if not Path(source).is_absolute() or ".." in Path(source).parts:
            raise ValueError("original source root must be absolute and normalized")
        result["source_dataset_root"] = source
        if "target_annotations" in item:
            result["target_annotations"] = str((path.parent / item["target_annotations"]).resolve())
        return result
    train = [resolve(item) for item in value["train"]]
    validation_value = value.get("validation")
    if isinstance(validation_value, list):
        if not validation_value:
            raise ValueError("manifest requires nonempty validation sources")
        validation = [resolve(item) for item in validation_value]
    else:
        validation = resolve(validation_value)
    validations = validation if isinstance(validation, list) else [validation]
    for key in ("root", "evidence_root", "source_dataset_root"):
        paths = [Path(item[key]) for item in [*train, *validations]]
        for i, p in enumerate(paths):
            if any(p == q or p in q.parents or q in p.parents for q in paths[:i]):
                raise ValueError(f"duplicate or nested manifest {key}")
    return {"train": train, "validation": validation}


def prepare_training_manifest(path, *, target_conditioned=False):
    sources = read_data_manifest(path)
    validations = sources["validation"]
    validations = validations if isinstance(validations, list) else [validations]
    annotation_flags = ["target_annotations" in item for item in [*sources["train"], *validations]]
    if target_conditioned and not all(annotation_flags):
        raise ValueError("target training requires target_annotations for every source")
    if not target_conditioned and any(annotation_flags):
        raise ValueError("target annotations require explicit --target-conditioned")
    validation = validations[0]
    parts = [prepare_training_data(
        item["root"], item["evidence_root"], validation["root"], validation["evidence_root"],
        train_source_root=item["source_dataset_root"],
        validation_source_root=validation["source_dataset_root"],
    ) for item in sources["train"]]
    if any(part["validation"] != parts[0]["validation"] for part in parts):
        raise ValueError("validation changed during preparation")
    validation_records = [parts[0]["validation"]]
    for item in validations[1:]:
        extra = prepare_training_data(
            sources["train"][0]["root"], sources["train"][0]["evidence_root"],
            item["root"], item["evidence_root"],
            train_source_root=sources["train"][0]["source_dataset_root"],
            validation_source_root=item["source_dataset_root"],
        )
        validation_records.append(extra["validation"])
    train_scenes = {scene for part in parts for scene in part["train"]["scenes"]}
    if any(train_scenes.intersection(record["scenes"]) for record in validation_records):
        raise ValueError("scene leakage between training and validation")
    # Pool by selected frame counts, not by the number of source folders.
    weights = np.asarray([part["train"]["frames"] for part in parts], dtype=np.float64)
    weights /= weights.sum()
    stats = {}
    for key in ("observation.state", "action"):
        values = [part["train_only_stats"][key] for part in parts]
        means = np.asarray([v["mean"] for v in values])
        mean = np.sum(weights[:, None] * means, axis=0)
        variance = np.sum(weights[:, None] * (np.asarray([v["std"] for v in values]) ** 2
                                               + (means - mean) ** 2), axis=0)
        stats[key] = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(),
                      "min": np.min([v["min"] for v in values], axis=0).tolist(),
                      "max": np.max([v["max"] for v in values], axis=0).tolist()}
    require_joint_normalization_stats(stats)
    validation_record = validation_records[0] if len(validation_records) == 1 else {
        "sources": validation_records,
        "frames": sum(record["frames"] for record in validation_records),
        "scenes": sorted({scene for record in validation_records for scene in record["scenes"]}),
        "capture_complete": all(record["capture_complete"] for record in validation_records),
    }
    result = {**parts[0], "train": {"sources": [part["train"] for part in parts],
        "frames": sum(part["train"]["frames"] for part in parts),
        "scenes": sorted(train_scenes)}, "validation": validation_record,
        "train_only_stats": stats}
    if target_conditioned:
        from .joint_target_policy import prepare_target_source, target_policy_contract_record
        for record, source in zip(result["train"]["sources"], sources["train"], strict=True):
            record.update(prepare_target_source(record, source["target_annotations"]))
        records = result["validation"].get("sources", [result["validation"]])
        for record, source in zip(records, validations, strict=True):
            record.update(prepare_target_source(record, source["target_annotations"]))
        result["contract"] = target_policy_contract_record()
    return result


def read_prepared_training_data(path, manifest, *, target_conditioned=False):
    """Explicitly reuse audited provenance; reject source/order/contract mismatches.

    This is not a new full content audit. Callers must retain the original data;
    dataset lengths and hold/close labels are checked again by the loaders.
    """
    prepared = json.loads(Path(path).read_text())
    expected_contract = joint_contract_record()
    if target_conditioned:
        from .joint_target_policy import target_policy_contract_record
        expected_contract = target_policy_contract_record()
    if prepared["contract"] != expected_contract:
        raise ValueError("prepared data contract differs from requested training")
    sources = read_data_manifest(manifest)
    for split in ("train", "validation"):
        inputs = sources[split] if isinstance(sources[split], list) else [sources[split]]
        records = prepared[split].get("sources", [prepared[split]])
        if len(inputs) != len(records) or sum(r["frames"] for r in records) != prepared[split]["frames"]:
            raise ValueError("prepared data source count or frame count mismatch")
        for source, record in zip(inputs, records, strict=True):
            for key in ("root", "evidence_root", "source_dataset_root", "target_annotations"):
                if source.get(key) != record.get(key):
                    raise ValueError(f"prepared data source/order mismatch: {key}")
            if (not record["episodes"] or not Path(record["root"]).is_dir()
                    or not Path(record["evidence_root"]).is_dir()):
                raise ValueError("prepared data sources must still exist with selected episodes")
            if target_conditioned and (len(record["target_frames"]) != record["frames"]
                                       or not Path(record["target_annotations"]).is_file()):
                raise ValueError("prepared target frame coverage mismatch")
    require_joint_normalization_stats(prepared["train_only_stats"])
    return prepared


class PrecloseWeightedDataset:
    """Training-only weights tied to expert future frame indices, never predicted closure."""
    def __init__(self, dataset, record, *, weight, frames_before_close):
        self.dataset = dataset
        self.weight = float(weight)
        self.frames_before_close = frames_before_close
        self.source_name = Path(record["root"]).name
        annotations = json.loads(Path(record["target_annotations"]).read_text())
        self.close_frames = {ep["episode_index"]: ep["close_frame"] for ep in annotations["episodes"]
                             if ep["episode_index"] in record["episodes"]}
        if set(self.close_frames) != set(record["episodes"]):
            raise ValueError("preclose weights require every selected episode's expert close frame")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        raw = self.dataset[index]
        ep, frame = int(raw["episode_index"]), int(raw["frame_index"])
        future_frames = frame + torch.arange(raw["action"].shape[0])
        close = self.close_frames[ep]
        selected = (future_frames >= close-self.frames_before_close) & (future_frames <= close)
        weights = torch.where(selected, self.weight, 1.0).to(torch.float32)
        return {**raw, "action_loss_weights": weights,
                "training_sample_key": f"{self.source_name}/{ep}/{frame}"}


def expert_hold_close_labels(actions, states, close_frame, *, holding_tolerance_deg=.01, close_frames=3):
    """Label the contiguous expert hold containing the reviewed first close.

    Labels use raw degrees and accepted actions, never model predictions or
    observed gripper state as the action target. A later return to the same
    pose is not a hold continuation. The first release also ends this interval.
    """
    actions, states = np.asarray(actions), np.asarray(states)
    if (actions.ndim != 2 or actions.shape[1] != JOINT_DIM or states.shape != actions.shape
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        raise ValueError("hold/close labels require finite episode actions and states [frames, 7]")
    if not math.isfinite(holding_tolerance_deg) or holding_tolerance_deg <= 0:
        raise ValueError("holding tolerance must be finite and positive")
    if isinstance(close_frames, bool) or not isinstance(close_frames, int) or close_frames < 1:
        raise ValueError("close_frames must be a positive integer")
    closed = actions[:, GRIPPER_INDEX] >= 50
    closes = np.flatnonzero(closed)
    if not len(closes) or close_frame != int(closes[0]) or states[close_frame, GRIPPER_INDEX] >= 50:
        raise ValueError("reviewed close_frame must match first close action with an open input state")
    reopened = np.flatnonzero(~closed[close_frame:])
    release = close_frame + int(reopened[0]) if len(reopened) else len(actions)
    near = np.max(np.abs(actions[:, :GRIPPER_INDEX] - actions[close_frame, :GRIPPER_INDEX]), axis=1) <= holding_tolerance_deg
    start = end = close_frame
    while start > 0 and near[start - 1]:
        start -= 1
    while end + 1 < release and near[end + 1]:
        end += 1
    state_closed = np.flatnonzero(states[:, GRIPPER_INDEX] >= 50)
    return {"close_frame": int(close_frame), "hold_start": int(start), "hold_end": int(end),
            "close_window_end": int(min(close_frame + close_frames - 1, release - 1)),
            "first_release_frame": int(release) if release < len(actions) else None,
            "state_first_closed_frame": int(state_closed[0]) if len(state_closed) else None,
            "frames": len(actions)}


def prepare_hold_close_supervision(record, *, holding_tolerance_deg=.01, close_frames=3):
    """Read selected source-local numerical rows only; no videos, models or hashes."""
    annotations = json.loads(Path(record["target_annotations"]).read_text())
    selected = [ep for ep in annotations["episodes"] if ep["episode_index"] in record["episodes"]]
    if (len(selected) != len(record["episodes"])
            or {ep["episode_index"] for ep in selected} != set(record["episodes"])
            or any(ep.get("reviewed") is not True for ep in selected)):
        raise ValueError("hold/close supervision requires reviewed selected episodes")
    table = pq.read_table(sorted((Path(record["root"]) / "data").glob("chunk-*/*.parquet")),
                          columns=["episode_index", "frame_index", "action", "observation.state"]).to_pydict()
    episode_indices = np.asarray(table["episode_index"])
    frame_indices = np.asarray(table["frame_index"])
    actions, states = np.asarray(table["action"]), np.asarray(table["observation.state"])
    labels = {}
    for ep in selected:
        indices = np.flatnonzero(episode_indices == ep["episode_index"])
        indices = indices[np.argsort(frame_indices[indices])]
        if not len(indices) or not np.array_equal(frame_indices[indices], np.arange(len(indices))):
            raise ValueError("hold/close supervision requires contiguous unique episode frames")
        labels[ep["episode_index"]] = {**expert_hold_close_labels(
            actions[indices], states[indices], ep["close_frame"],
            holding_tolerance_deg=holding_tolerance_deg, close_frames=close_frames),
            "scene_id": ep["scene_id"]}
    if sum(label["frames"] for label in labels.values()) != record["frames"]:
        raise ValueError("hold/close supervision frame count changed")
    return labels


class HoldCloseDataset:
    """Training-only metadata over unchanged source-local observations/actions."""
    def __init__(self, dataset, record, *, holding, holding_tolerance_deg, gripper_weight, close_frames):
        self.dataset, self.holding, self.gripper_weight = dataset, holding, gripper_weight
        self.source_name = Path(record["root"]).name
        self.labels = prepare_hold_close_supervision(record, holding_tolerance_deg=holding_tolerance_deg,
                                                     close_frames=close_frames)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        raw = self.dataset[index]
        ep, frame = int(raw["episode_index"]), int(raw["frame_index"])
        label = self.labels[ep]
        future = frame + torch.arange(raw["action"].shape[0])
        valid = ~raw["action_is_pad"] & (future < label["frames"])
        extra = {"training_sample_key": f"{self.source_name}/{ep}/{frame}"}
        if self.holding:
            rows = (future >= label["hold_start"]) & (future <= label["hold_end"]) & valid
            extra["holding_pair_mask"] = rows[:-1] & rows[1:]
        if self.gripper_weight is not None:
            rows = (future >= label["close_frame"]) & (future <= label["close_window_end"]) & valid
            extra["gripper_loss_weights"] = torch.where(rows, self.gripper_weight, 1.0).to(torch.float32)
        return {**raw, **extra}


def training_supervision_record(dataset):
    if hasattr(dataset, "datasets"):
        return {"sources": [training_supervision_record(ds) for ds in dataset.datasets]}
    return {"source": dataset.source_name, "episodes": dataset.labels}


def training_observation_keys(dataset):
    """Source-local identities in actual loader order; never decode an image."""
    if hasattr(dataset, "datasets"):
        return [key for source in dataset.datasets for key in training_observation_keys(source)]
    native = dataset
    while hasattr(native, "dataset"):
        native = native.dataset
    source = Path(native.root).name
    return [f"{source}/{int(ep)}/{int(frame)}" for ep, frame in zip(
        native.hf_dataset["episode_index"], native.hf_dataset["frame_index"], strict=True)]


class FixedObservationDataset:
    """Diagnostic subset of complete native samples, not a production candidate."""
    def __init__(self, dataset, sample_keys):
        if (not isinstance(sample_keys, list) or not sample_keys
                or not all(isinstance(key, str) for key in sample_keys)
                or len(set(sample_keys)) != len(sample_keys)):
            raise ValueError("fit observations require a nonempty list of unique sample keys")
        keys = training_observation_keys(dataset)
        if len(keys) != len(dataset) or len(set(keys)) != len(keys):
            raise ValueError("loaded training observations must have unique, aligned identities")
        lookup = {key: index for index, key in enumerate(keys)}
        missing = set(sample_keys) - lookup.keys()
        if missing:
            raise ValueError(f"fit observations are absent from training data: {sorted(missing)}")
        self.dataset, self.sample_keys = dataset, list(sample_keys)
        self.indices = [lookup[key] for key in sample_keys]
        self.report = dict(diagnostic_only=True, source_frames=len(dataset), selected_frames=len(sample_keys),
                           sample_keys=self.sample_keys, source_indices=self.indices,
                           action_chunks_and_normalization_unchanged=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        raw = self.dataset[self.indices[index]]
        key = self.sample_keys[index]
        _, episode, frame = key.rsplit("/", 2)
        if int(raw["episode_index"]) != int(episode) or int(raw["frame_index"]) != int(frame):
            raise ValueError("loaded fit observation differs from its frozen identity")
        return {**raw, "training_sample_key": key}


def phase_sampling_inputs(dataset):
    """Read actual loaded row identities without decoding images or future actions."""
    if hasattr(dataset, "datasets"):
        keys, labels = [], {}
        for source in dataset.datasets:
            source_keys, source_labels = phase_sampling_inputs(source)
            if labels.keys() & source_labels.keys():
                raise ValueError("phase sampling requires unique source/episode identities")
            keys.extend(source_keys)
            labels.update(source_labels)
        return keys, labels
    native = dataset
    while hasattr(native, "dataset"):
        native = native.dataset
    keys = [f"{dataset.source_name}/{int(ep)}/{int(frame)}" for ep, frame in zip(
        native.hf_dataset["episode_index"], native.hf_dataset["frame_index"], strict=True)]
    return keys, {f"{dataset.source_name}/{ep}": label for ep, label in dataset.labels.items()}


class PhaseBalancedBatchSampler:
    """Frozen 4 whole-trajectory + 2 late/hold + 1 close + 1 release batches.

    Each event subgroup rotates demonstrations independently; coupling the
    three subgroups to one episode counter would omit two thirds of episodes
    from exact-event sampling when the episode count is divisible by three.
    Only input indices change. Labels never enter the model or alter targets.
    """
    def __init__(self, sample_keys, labels, *, steps, seed):
        from collections import Counter
        if steps < 1 or len(set(sample_keys)) != len(sample_keys):
            raise ValueError("phase sampling requires positive steps and unique input keys")
        self.sample_keys = list(sample_keys)
        by_demo = {demo: {} for demo in labels}
        for index, key in enumerate(sample_keys):
            demo, frame = key.rsplit("/", 1)
            by_demo[demo][int(frame)] = index
        pools = {name: {} for name in ("late_hold", "close_before", "close_event", "close_after",
                                       "release_before", "release_event", "release_after")}
        for demo, label in labels.items():
            frames, close, release = label["frames"], label["close_frame"], label["first_release_frame"]
            if (set(by_demo[demo]) != set(range(frames)) or release is None
                    or not 0 <= label["hold_start"] <= close <= label["hold_end"] < release < frames):
                raise ValueError("phase sampling requires complete episode indices and ordered hold/close/release")
            windows = {"late_hold": (label["hold_start"] - 25, label["hold_end"])}
            for name, event in (("close", close), ("release", release)):
                windows.update({name + "_before": (event - 10, event - 1),
                                name + "_event": (event, event), name + "_after": (event + 1, event + 10)})
            for name, (start, end) in windows.items():
                pools[name][demo] = [by_demo[demo][frame] for frame in range(max(0, start), min(frames-1, end)+1)]
                if not pools[name][demo]:
                    raise ValueError(f"phase sampling requires nonempty clipped {name} inputs: {demo}")
        rng = np.random.default_rng(seed)  # independent of model noise/dropout and global RNG
        def cycles(items):
            while True:
                yield from rng.permutation(items).tolist()
        uniform = cycles(list(range(len(sample_keys))))
        rotations = {name: cycles(list(labels)) for name in pools}
        self.batches, self.groups = [], []
        assigned = {demo: Counter() for demo in labels}
        events = {demo: Counter() for demo in labels}
        for step in range(steps):
            suffix = ("before", "event", "after")[step % 3]
            groups = ["uniform"]*4 + ["late_hold"]*2 + ["close_"+suffix, "release_"+suffix]
            batch = []
            for group in groups:
                if group == "uniform":
                    index = next(uniform)
                else:
                    demo = next(rotations[group])
                    pool = pools[group][demo]
                    index = pool[int(rng.integers(len(pool)))]
                demo, frame = sample_keys[index].rsplit("/", 1)
                assigned[demo][group] += 1
                frame, label = int(frame), labels[demo]
                for event, field in (("close", "close_frame"), ("release", "first_release_frame")):
                    if frame == label[field]:
                        events[demo][event] += 1
                batch.append(index)
            self.batches.append(batch)
            self.groups.append(groups)
        flat = [i for batch in self.batches for i in batch]
        self.report = dict(seed=seed, steps=steps, batch_size=8, input_frames=len(sample_keys),
            sample_presentations=len(flat), unique_inputs=len(set(flat)), repeated_presentations=len(flat)-len(set(flat)),
            within_batch_duplicate_presentations=sum(len(b)-len(set(b)) for b in self.batches),
            assigned_group_counts=dict(Counter(g for groups in self.groups for g in groups)),
            by_demonstration={demo: dict(assigned_groups=dict(assigned[demo]), current_events=dict(events[demo]))
                              for demo in labels},
            current_event_demonstrations={event: sum(events[d][event] > 0 for d in labels)
                                          for event in ("close", "release")},
            batches=[dict(indices=batch, groups=groups, sample_keys=[sample_keys[i] for i in batch])
                     for batch, groups in zip(self.batches, self.groups, strict=True)])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def load_prepared_dataset(record, *, video_backend="pyav", preclose_loss_weight=None, preclose_frames=10,
                          holding_loss_coefficient=0., holding_tolerance_deg=.01,
                          gripper_close_loss_weight=None, gripper_close_frames=3, phase_balanced_sampling=False):
    """Each source constructs action chunks before concatenation: no cross-source tails."""
    from torch.utils.data import ConcatDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if "sources" in record:
        ds = ConcatDataset([load_prepared_dataset(source, video_backend=video_backend,
                            preclose_loss_weight=preclose_loss_weight, preclose_frames=preclose_frames,
                            holding_loss_coefficient=holding_loss_coefficient, holding_tolerance_deg=holding_tolerance_deg,
                            gripper_close_loss_weight=gripper_close_loss_weight, gripper_close_frames=gripper_close_frames,
                            phase_balanced_sampling=phase_balanced_sampling)
                            for source in record["sources"]])
    else:
        ds = LeRobotDataset("local/aubo-joint", root=record["root"], episodes=record["episodes"],
            delta_timestamps={"action": [i / 25 for i in range(50)]}, video_backend=video_backend)
        if "target_frames" in record:
            from .joint_target_policy import JointTargetDataset
            ds = JointTargetDataset(ds, record["target_frames"])
        if preclose_loss_weight is not None:
            ds = PrecloseWeightedDataset(ds, record, weight=preclose_loss_weight, frames_before_close=preclose_frames)
        if holding_loss_coefficient > 0 or gripper_close_loss_weight is not None or phase_balanced_sampling:
            ds = HoldCloseDataset(ds, record, holding=holding_loss_coefficient > 0,
                                  holding_tolerance_deg=holding_tolerance_deg,
                                  gripper_weight=gripper_close_loss_weight, close_frames=gripper_close_frames)
    if len(ds) != record["frames"]:
        raise ValueError("selected dataset frame count changed")
    return ds


def prediction_metrics(states, targets, predictions):
    """First-action teacher-forced metrics, explicitly separated from success."""
    states, targets, predictions = map(np.asarray, (states, targets, predictions))
    if (targets.ndim != 2 or targets.shape[1] != JOINT_DIM or not len(targets)
            or states.shape != targets.shape or predictions.shape != targets.shape
            or not all(np.isfinite(a).all() for a in (states, targets, predictions))):
        raise ValueError("evaluation requires matching finite nonempty [frames, 7] arrays")
    current, truth, predicted = states[:, GRIPPER_INDEX] >= 50, targets[:, GRIPPER_INDEX] >= 50, predictions[:, GRIPPER_INDEX] >= 50
    activate, release = ~current & truth, current & ~truth
    hold = current == truth
    def fraction(values, mask):
        return float(values[mask].mean()) if mask.any() else None
    return {"frames": len(targets), "joint_mae_deg": np.abs(predictions[:, :GRIPPER_INDEX] - targets[:, :GRIPPER_INDEX]).mean(0).tolist(),
            "gripper_mae": float(np.abs(predictions[:, GRIPPER_INDEX] - targets[:, GRIPPER_INDEX]).mean()),
            "gripper_accuracy": float((predicted == truth).mean()),
            "copy_current_state_accuracy": float(hold.mean()),
            "activate_samples": int(activate.sum()), "release_samples": int(release.sum()),
            "activate_recall": fraction(predicted, activate), "release_recall": fraction(~predicted, release),
            "hold_false_switch_rate": fraction(predicted != current, hold),
            "physical_success_verified": False}
