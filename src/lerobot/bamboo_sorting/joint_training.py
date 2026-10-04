"""Offline preparation for full-joint AUBO SmolVLA. No hardware imports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .aubo_joint_audit import audit_joint_dataset
from .aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, joint_contract_record
from .joint_gripper_quality import require_joint_normalization_stats


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
        scenes = sorted({r["scene_id"].strip() for r in episodes})
        if not all(scenes):
            raise ValueError("nonempty scene identifiers are required")
        records.append({"root": str(root), "evidence_root": str(Path(evidence).resolve()),
                        "source_dataset_root": report["source_dataset_root"],
                        "episodes": selected, "scenes": scenes,
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


def prepare_training_manifest(path):
    sources = read_data_manifest(path)
    validations = sources["validation"]
    validations = validations if isinstance(validations, list) else [validations]
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
    return {**parts[0], "train": {"sources": [part["train"] for part in parts],
        "frames": sum(part["train"]["frames"] for part in parts),
        "scenes": sorted(train_scenes)}, "validation": validation_record,
        "train_only_stats": stats}


def load_prepared_dataset(record, *, video_backend="pyav"):
    """Each source constructs action chunks before concatenation: no cross-source tails."""
    from torch.utils.data import ConcatDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if "sources" in record:
        ds = ConcatDataset([load_prepared_dataset(source, video_backend=video_backend)
                            for source in record["sources"]])
    else:
        ds = LeRobotDataset("local/aubo-joint", root=record["root"], episodes=record["episodes"],
            delta_timestamps={"action": [i / 25 for i in range(50)]}, video_backend=video_backend)
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
