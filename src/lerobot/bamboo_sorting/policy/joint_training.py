"""Offline preparation for full-joint AUBO SmolVLA. No hardware imports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..capture.aubo_joint_audit import audit_joint_dataset
from ..aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, joint_contract_record
from ..joint_gripper_quality import require_joint_normalization_stats
from ..joint_demo_metadata import DEMO_SCOPES
from ..joint_stack_metadata import stack_summary


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
                          *, train_source_root=None, validation_source_root=None,
                          train_scenes=None, validation_scenes=None, _cache=None):
    roots = [Path(p).resolve() for p in (train_root, validation_root)]
    shared_selection = (roots[0] == roots[1] and train_scenes and validation_scenes
                        and not set(train_scenes).intersection(validation_scenes)
                        and Path(train_evidence).resolve() == Path(validation_evidence).resolve())
    if (roots[0] == roots[1] and not shared_selection
            or roots[0] in roots[1].parents or roots[1] in roots[0].parents):
        raise ValueError("training and validation datasets must have independent roots")
    cache = {} if _cache is None else _cache
    records = []
    for root, evidence, split, source, selection in zip(
            roots, (train_evidence, validation_evidence), ("train", "validation"),
            (train_source_root, validation_source_root), (train_scenes, validation_scenes), strict=True):
        # This existing trainer constructs the original dual-640x480 adapter.
        # Native-resolution recordings need matched model preprocessing first.
        key = ("audit", str(root), str(Path(evidence).resolve()), source)
        if key not in cache:
            cache[key] = audit_joint_dataset(root, evidence, source_dataset_root=source)
        report = cache[key]
        if any(tuple(shape) != (480, 640, 3) for shape in report.get("image_shapes", {}).values()):
            raise ValueError("this trainer uses dual 640x480 inputs; configure native-resolution model "
                             "preprocessing before training new camera recordings; keep originals unchanged")
        if not report["gripper_quality_passed"]:
            raise ValueError(f"{split} gripper quality failed: {report['gripper_quality_issues']}")
        candidates = report["episodes"]
        if selection is not None:
            if (not isinstance(selection, list) or not selection
                    or any(not isinstance(s, str) or not s.strip() or s != s.strip() for s in selection)
                    or len(set(selection)) != len(selection)):
                raise ValueError("selection_scenes requires distinct nonempty scene identifiers")
            if not set(selection) <= {r["scene_id"] for r in candidates}:
                raise ValueError("selection_scenes contains unknown scenes")
            candidates = [r for r in candidates if r["scene_id"] in selection]
            if any("pick_step" not in r for r in candidates):
                raise ValueError("selection_scenes requires whole stack scenes")
        selected = [r["episode_index"] for r in candidates
                    if r["episode_index"] in report["supervised_candidate_episodes"]]
        episodes = [r for r in candidates if r["episode_index"] in selected]
        if not episodes or (selection is None and any(r.get("split") != split for r in episodes)):
            raise ValueError(f"{split} requires successful episodes explicitly tagged with that split")
        # A failed earlier step can perturb the later successful sample in the
        # same pile. Group all stack records, including excluded outcomes.
        grouped = [r for r in candidates if "pick_step" in r]
        if selection is None and any(r.get("split") != split for r in grouped):
            raise ValueError(f"{split} stack scenes must stay in one split")
        scenes = sorted({r["scene_id"].strip() for r in [*episodes, *grouped]})
        if not all(scenes):
            raise ValueError("nonempty scene identifiers are required")
        scopes = {r.get("demo_scope", "full") for r in report["episodes"]}
        if len(scopes) != 1 or not scopes <= set(DEMO_SCOPES):
            raise ValueError("each capture source must have one demo_scope")
        records.append({"root": str(root), "evidence_root": str(Path(evidence).resolve()),
                        "source_dataset_root": report["source_dataset_root"],
                        "episodes": selected, "scenes": scenes,
                        "stack_summary": (stack_summary(candidates) if selection is not None
                                          else report.get("stack_summary", {})),
                        # Omit the default to retain legacy full-only provenance
                        # and its exact checkpoint evaluation comparison.
                        **({"demo_scope": "local_alignment"} if scopes == {"local_alignment"} else {}),
                        "frames": sum(r["frames"] for r in episodes),
                        "capture_complete": report["capture_complete"],
                        **({"selection_scenes": sorted(selection), "effective_split": split,
                            "capture_splits": sorted({r["split"] for r in candidates})}
                           if selection is not None else {})})
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
        for field, key in (("dataset_sha256", "root"), ("evidence_sha256", "evidence_root")):
            identity = ("fingerprint", record[key])
            if identity not in cache:
                cache[identity] = fingerprint(record[key])
            record[field] = cache[identity]
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
        result["demo_scope"] = item.get("demo_scope", "full")
        if result["demo_scope"] not in DEMO_SCOPES:
            raise ValueError("manifest source has unknown demo_scope")
        if "selection_scenes" in item:
            scenes = item["selection_scenes"]
            if (not isinstance(scenes, list) or not scenes
                    or any(not isinstance(s, str) or not s.strip() or s != s.strip() for s in scenes)
                    or len(set(scenes)) != len(scenes)):
                raise ValueError("selection_scenes requires distinct nonempty scene identifiers")
            result["selection_scenes"] = sorted(scenes)
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
        items = [*train, *validations]
        paths = [Path(item[key]) for item in items]
        for i, p in enumerate(paths):
            for j, q in enumerate(paths[:i]):
                if p == q:
                    a, b = items[i], items[j]
                    if (all(a[k] == b[k] for k in ("root", "evidence_root", "source_dataset_root"))
                            and a.get("target_annotations") == b.get("target_annotations")
                            and a["demo_scope"] == b["demo_scope"]
                            and a.get("selection_scenes") and b.get("selection_scenes")
                            and not set(a["selection_scenes"]).intersection(b["selection_scenes"])):
                        continue
                    raise ValueError(f"duplicate or overlapping manifest {key}")
                if p in q.parents or q in p.parents:
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
    cache = {}
    parts = [prepare_training_data(
        item["root"], item["evidence_root"], validation["root"], validation["evidence_root"],
        train_source_root=item["source_dataset_root"],
        validation_source_root=validation["source_dataset_root"],
        train_scenes=item.get("selection_scenes"), validation_scenes=validation.get("selection_scenes"),
        _cache=cache,
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
            train_scenes=sources["train"][0].get("selection_scenes"),
            validation_scenes=item.get("selection_scenes"), _cache=cache,
        )
        validation_records.append(extra["validation"])
    for record, source in zip([part["train"] for part in parts] + validation_records,
                              [*sources["train"], *validations], strict=True):
        if record.get("demo_scope", "full") != source["demo_scope"]:
            raise ValueError("manifest demo_scope differs from audited capture metadata")
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


def read_prepared_training_data(path, manifest, *, target_conditioned=False, target_outline_width_px=2):
    """Explicitly reuse audited provenance; reject source/order/contract mismatches.

    This is not a new full content audit. Callers must retain the original data;
    dataset lengths and hold/close labels are checked again by the loaders.
    """
    prepared = json.loads(Path(path).read_text())
    expected_contracts = [joint_contract_record()]
    if target_conditioned:
        from .joint_target_policy import target_policy_contract_record
        # Original audited 2px sources can create a 1px view without changing
        # raw data. A saved 1px view requires explicit 1px selection on reuse.
        expected_contracts = [target_policy_contract_record(),
                              target_policy_contract_record(outline_width_px=target_outline_width_px)]
    if prepared["contract"] not in expected_contracts:
        raise ValueError("prepared data contract differs from requested training")
    sources = read_data_manifest(manifest)
    for split in ("train", "validation"):
        inputs = sources[split] if isinstance(sources[split], list) else [sources[split]]
        records = prepared[split].get("sources", [prepared[split]])
        if len(inputs) != len(records) or sum(r["frames"] for r in records) != prepared[split]["frames"]:
            raise ValueError("prepared data source count or frame count mismatch")
        for source, record in zip(inputs, records, strict=True):
            if source.get("selection_scenes") != record.get("selection_scenes"):
                raise ValueError("prepared data scene selection mismatch; audit the new manifest first")
            if source.get("selection_scenes") and record.get("effective_split") != split:
                raise ValueError("prepared data effective split mismatch; audit the new manifest first")
            if source["demo_scope"] != record.get("demo_scope", "full"):
                raise ValueError("prepared data demo_scope mismatch; audit the new manifest first")
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


def load_prepared_dataset(record, *, video_backend="pyav", target_outline_width_px=2):
    """Each source builds its own original 50-row chunks before concatenation."""
    from torch.utils.data import ConcatDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if "sources" in record:
        ds = ConcatDataset([load_prepared_dataset(source, video_backend=video_backend,
                            target_outline_width_px=target_outline_width_px)
                            for source in record["sources"]])
    else:
        ds = LeRobotDataset("local/aubo-joint", root=record["root"], episodes=record["episodes"],
            delta_timestamps={"action": [i / 25 for i in range(50)]}, video_backend=video_backend)
        if "target_frames" in record:
            from .joint_target_policy import JointTargetDataset
            ds = JointTargetDataset(ds, record["target_frames"], outline_width_px=target_outline_width_px)
    if len(ds) != record["frames"]:
        raise ValueError("selected dataset frame count changed")
    return ds


def training_sampling_plan(record, local_demo_fraction=None):
    """Probability mass belongs to full/local groups, not to source folders."""
    sources = record.get("sources", [record])
    counts = dict.fromkeys(DEMO_SCOPES, 0)
    for source in sources:
        scope = source.get("demo_scope", "full")
        frames = source["frames"]
        if scope not in counts or type(frames) is not int or frames < 1:
            raise ValueError("sampling requires valid demo_scope and positive frame counts")
        counts[scope] += frames
    total = sum(counts.values())
    if total != record["frames"]:
        raise ValueError("sampling source frame count mismatch")
    fractions = {scope: frames / total for scope, frames in counts.items()}
    if local_demo_fraction is not None:
        if (isinstance(local_demo_fraction, bool) or not np.isfinite(local_demo_fraction)
                or not 0 < local_demo_fraction < 1):
            raise ValueError("local-demo-fraction must be finite and strictly between 0 and 1")
        if not all(counts.values()):
            raise ValueError("balanced sampling requires both full and local_alignment sources")
        fractions = {"full": 1 - local_demo_fraction, "local_alignment": local_demo_fraction}
    return {"mode": "uniform_frames" if local_demo_fraction is None else "balanced_demo_scopes",
            "group_frames": counts, "expected_group_fractions": fractions,
            "within_group": "uniform_frames", "replacement": local_demo_fraction is not None,
            "draws_per_pass": total}


def make_training_sampler(record, local_demo_fraction, *, seed):
    """Sample existing frame indices; never concatenate or splice trajectories."""
    if local_demo_fraction is None:
        return None
    from torch import Generator
    from torch.utils.data import WeightedRandomSampler
    plan = training_sampling_plan(record, local_demo_fraction)
    weights = np.concatenate([np.full(source["frames"],
        plan["expected_group_fractions"][source.get("demo_scope", "full")]
        / plan["group_frames"][source.get("demo_scope", "full")], dtype=np.float64)
        for source in record.get("sources", [record])])
    return WeightedRandomSampler(weights, plan["draws_per_pass"], replacement=True,
                                 generator=Generator().manual_seed(seed))


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
