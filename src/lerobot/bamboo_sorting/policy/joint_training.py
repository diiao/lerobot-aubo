"""Offline preparation for full-joint AUBO SmolVLA. No hardware imports."""

import copy
from fractions import Fraction
import hashlib
import json
import math
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
            if not target_conditioned and any(k in record for k in (
                    "target_annotations", "target_frames", "target_episode_metadata")):
                raise ValueError("raw training data must not contain target inputs")
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


def prepare_raw_training_data(audited_path, audited_manifest, raw_manifest):
    """Reuse audited captures/splits/stats while explicitly removing outline inputs.

    This is a new dataset view, not a fresh content hash audit. Source manifests
    must differ only by their annotation references; raw interfaces are checked
    by the dataset loaders before training. Original files are never rewritten.
    """
    prepared = read_prepared_training_data(audited_path, audited_manifest, target_conditioned=True)
    original, raw = map(read_data_manifest, (audited_manifest, raw_manifest))
    result = copy.deepcopy(prepared)
    result["contract"] = joint_contract_record()
    for split in ("train", "validation"):
        before = original[split] if isinstance(original[split], list) else [original[split]]
        after = raw[split] if isinstance(raw[split], list) else [raw[split]]
        expected = [{k: v for k, v in item.items() if k != "target_annotations"} for item in before]
        if after != expected:
            raise ValueError("raw manifest must preserve audited source order, scenes and demo scopes")
        for record in result[split].get("sources", [result[split]]):
            metadata = record.pop("target_episode_metadata")
            if {m["episode_index"] for m in metadata} != set(record["episodes"]):
                raise ValueError("audited episode metadata coverage differs")
            record["episode_metadata"] = [dict(episode_index=m["episode_index"], scene_id=m["scene_id"])
                                          for m in metadata]
            record.pop("target_annotations")
            record.pop("target_frames")
    if set(result["train"]["scenes"]) & set(result["validation"]["scenes"]):
        raise ValueError("scene leakage between training and validation")
    result["preparation_provenance"] = dict(mode="raw_rgb_view_of_audited_captures",
        audited_data=str(Path(audited_path).resolve()), audited_manifest=str(Path(audited_manifest).resolve()),
        source_selection_unchanged=True, train_only_stats_unchanged=True,
        content_hashes_reused=True, fresh_content_hash_audit=False, original_annotations_preserved=True)
    return result


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


def training_sampling_plan(record, local_demo_fraction=None, *, sampling_mode="weighted_random"):
    """Probability mass belongs to full/local groups, not to source folders."""
    if sampling_mode not in ("weighted_random", "full_coverage_local_oversample"):
        raise ValueError("unknown training sampling mode")
    if sampling_mode == "full_coverage_local_oversample" and local_demo_fraction is None:
        raise ValueError("full coverage oversampling requires local-demo-fraction")
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
    plan = {"mode": "uniform_frames" if local_demo_fraction is None else "balanced_demo_scopes",
            "group_frames": counts, "expected_group_fractions": fractions,
            "within_group": "uniform_frames", "replacement": local_demo_fraction is not None,
            "draws_per_pass": total}
    if sampling_mode == "full_coverage_local_oversample":
        # Exact decimal arithmetic avoids an extra draw at ratios such as 0.8.
        fraction = Fraction(str(local_demo_fraction))
        original_fraction = counts["local_alignment"] / total
        if local_demo_fraction < original_fraction:
            raise ValueError("local-demo-fraction cannot be below the original fraction when only adding local samples")
        local_draws = counts["local_alignment"] if local_demo_fraction == original_fraction else math.ceil(
            counts["full"] * fraction / (1 - fraction))
        draws = counts["full"] + local_draws
        plan.update(mode=sampling_mode, draws_per_pass=draws, base_draws_per_pass=total,
            extra_local_draws_per_pass=local_draws - counts["local_alignment"],
            group_draws_per_pass={"full": counts["full"], "local_alignment": local_draws},
            requested_local_demo_fraction=local_demo_fraction,
            expected_group_fractions={"full": counts["full"] / draws, "local_alignment": local_draws / draws},
            full_coverage_per_pass=True, replacement_scope="extra_local_draws_only",
            sample_unit="existing source-local observation and its episode-bounded future action chunk")
    return plan


class FullCoverageLocalSampler:
    """Yield existing observation indices, never build or join action sequences.

    Every complete pass includes each training index once, plus uniformly drawn
    local indices with replacement. The dataset still owns source/episode lookup
    and action padding. A private advancing generator isolates sampling RNG.
    """
    def __init__(self, total, local_indices, extra_local_draws, *, seed):
        import torch

        self.total = total
        self.local_indices = torch.tensor(local_indices, dtype=torch.long)
        self.extra_local_draws = extra_local_draws
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.total + self.extra_local_draws

    def __iter__(self):
        import torch

        extra = self.local_indices[torch.randint(len(self.local_indices), (self.extra_local_draws,),
                                                generator=self.generator)]
        indices = torch.cat((torch.arange(self.total), extra))
        yield from indices[torch.randperm(len(indices), generator=self.generator)].tolist()


def make_training_sampler(record, local_demo_fraction, *, seed, sampling_mode="weighted_random"):
    """Sample existing frame indices; never concatenate or splice trajectories."""
    plan = training_sampling_plan(record, local_demo_fraction, sampling_mode=sampling_mode)
    if local_demo_fraction is None:
        return None
    from torch import Generator
    from torch.utils.data import WeightedRandomSampler
    if sampling_mode == "full_coverage_local_oversample":
        local_indices, offset = [], 0
        for source in record.get("sources", [record]):
            if source.get("demo_scope", "full") == "local_alignment":
                local_indices.extend(range(offset, offset + source["frames"]))
            offset += source["frames"]
        return FullCoverageLocalSampler(record["frames"], local_indices,
                                        plan["extra_local_draws_per_pass"], seed=seed)
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
