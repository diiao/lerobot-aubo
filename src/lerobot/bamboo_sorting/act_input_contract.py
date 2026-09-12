"""Explicit input contracts for controlled AUBO ACT ablation experiments.

The recorded ``gripper_pos`` observation is a software latch, not a physical
sensor.  This module supports removing that one observation dimension without
rewriting the source LeRobot dataset or removing the gripper action target.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

STATE_KEY = "observation.state"
FULL_STATE_VARIANT = "full"
DROP_GRIPPER_STATE_VARIANT = "drop_gripper"
SUPPORTED_STATE_VARIANTS = frozenset({FULL_STATE_VARIANT, DROP_GRIPPER_STATE_VARIANT})


@dataclass(frozen=True)
class ActStateInputContract:
    """Describe how recorded robot state is presented to an ACT policy."""

    variant: str
    source_names: tuple[str, ...]
    model_names: tuple[str, ...]
    removed_name: str | None = None
    removed_index: int | None = None

    @property
    def source_width(self) -> int:
        return len(self.source_names)

    @property
    def model_width(self) -> int:
        return len(self.model_names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "AuboActStateInputContractV1",
            "variant": self.variant,
            "state_key": STATE_KEY,
            "source_names": list(self.source_names),
            "model_names": list(self.model_names),
            "removed_name": self.removed_name,
            "removed_index": self.removed_index,
        }


@dataclass(frozen=True)
class EpisodeSplit:
    """A verified train/validation episode split for one dataset snapshot."""

    path: Path
    train_episodes: tuple[int, ...]
    val_episodes: tuple[int, ...]
    dataset_info_sha256: str
    payload: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_state_input_contract(
    features: Mapping[str, Mapping[str, Any]],
    variant: str,
) -> ActStateInputContract:
    """Build a fail-closed state contract from dataset feature metadata."""

    if variant not in SUPPORTED_STATE_VARIANTS:
        raise ValueError(
            f"unsupported STATE_INPUT_VARIANT={variant!r}; expected one of "
            f"{sorted(SUPPORTED_STATE_VARIANTS)}"
        )
    state_feature = features.get(STATE_KEY)
    if state_feature is None:
        raise ValueError(f"dataset is missing {STATE_KEY}")
    names = state_feature.get("names")
    if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{STATE_KEY} must provide an ordered list of names")
    source_names = tuple(names)
    shape = tuple(state_feature.get("shape", ()))
    if shape != (len(source_names),):
        raise ValueError(f"{STATE_KEY} shape {shape} does not match names {source_names}")

    if variant == FULL_STATE_VARIANT:
        return ActStateInputContract(variant, source_names, source_names)

    matches = [index for index, name in enumerate(source_names) if name == "gripper_pos"]
    if len(matches) != 1:
        raise ValueError(
            f"{STATE_KEY} must contain exactly one gripper_pos for the ablation; found {len(matches)}"
        )
    removed_index = matches[0]
    model_names = source_names[:removed_index] + source_names[removed_index + 1 :]
    return ActStateInputContract(
        variant=variant,
        source_names=source_names,
        model_names=model_names,
        removed_name="gripper_pos",
        removed_index=removed_index,
    )


def _drop_last_axis(value: Any, *, index: int, expected_width: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim < 1 or value.shape[-1] != expected_width:
            raise ValueError(f"state tensor must end in width {expected_width}; got {tuple(value.shape)}")
        keep = torch.arange(expected_width, device=value.device) != index
        return value[..., keep]
    if isinstance(value, np.ndarray):
        if value.ndim < 1 or value.shape[-1] != expected_width:
            raise ValueError(f"state array must end in width {expected_width}; got {value.shape}")
        return np.delete(value, index, axis=-1)
    if isinstance(value, tuple):
        if len(value) != expected_width:
            raise ValueError(f"state tuple must have width {expected_width}; got {len(value)}")
        return value[:index] + value[index + 1 :]
    if isinstance(value, list):
        if len(value) != expected_width:
            raise ValueError(f"state list must have width {expected_width}; got {len(value)}")
        return value[:index] + value[index + 1 :]
    raise TypeError(f"unsupported state container: {type(value).__name__}")


def apply_state_input_contract(
    batch: Mapping[str, Any],
    contract: ActStateInputContract,
) -> dict[str, Any]:
    """Return a shallow batch copy with the declared state ablation applied."""

    if STATE_KEY not in batch:
        raise KeyError(f"batch is missing {STATE_KEY}")
    result = dict(batch)
    state = batch[STATE_KEY]
    if contract.removed_index is None:
        shape = getattr(state, "shape", None)
        width = shape[-1] if shape is not None and len(shape) else len(state)
        if width != contract.source_width:
            raise ValueError(
                f"state input width must be {contract.source_width} for full contract; got {width}"
            )
        return result
    result[STATE_KEY] = _drop_last_axis(
        state,
        index=contract.removed_index,
        expected_width=contract.source_width,
    )
    return result


def adapt_features_and_stats(
    features: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Mapping[str, Any]],
    contract: ActStateInputContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create policy metadata matching the state contract; source objects stay unchanged."""

    adapted_features = copy.deepcopy(dict(features))
    adapted_stats = copy.deepcopy(dict(stats))
    state_feature = adapted_features[STATE_KEY]
    state_feature["names"] = list(contract.model_names)
    state_feature["shape"] = [contract.model_width]

    if contract.removed_index is None:
        return adapted_features, adapted_stats

    for stat_name, value in list(adapted_stats[STATE_KEY].items()):
        shape = getattr(value, "shape", None)
        if shape is not None:
            if len(shape) > 0 and shape[-1] == contract.source_width:
                adapted_stats[STATE_KEY][stat_name] = _drop_last_axis(
                    value,
                    index=contract.removed_index,
                    expected_width=contract.source_width,
                )
        elif isinstance(value, (list, tuple)) and len(value) == contract.source_width:
            adapted_stats[STATE_KEY][stat_name] = _drop_last_axis(
                value,
                index=contract.removed_index,
                expected_width=contract.source_width,
            )
    return adapted_features, adapted_stats


def load_episode_split(
    path: Path,
    *,
    dataset_info_path: Path,
    total_episodes: int,
) -> EpisodeSplit:
    """Load a split and bind it to the exact dataset metadata hash."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "AuboActEpisodeSplitV1":
        raise ValueError("episode split schema_version must be AuboActEpisodeSplitV1")

    expected_hash = payload.get("dataset_info_sha256")
    actual_hash = sha256_file(dataset_info_path)
    if expected_hash != actual_hash:
        raise ValueError(
            "episode split dataset_info_sha256 does not match the selected dataset: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    def _episode_list(name: str) -> tuple[int, ...]:
        values = payload.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"episode split {name} must be a non-empty list")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise ValueError(f"episode split {name} must contain integer episode indices")
        result = tuple(values)
        if len(set(result)) != len(result):
            raise ValueError(f"episode split {name} contains duplicate indices")
        invalid = [value for value in result if value < 0 or value >= total_episodes]
        if invalid:
            raise ValueError(f"episode split {name} contains out-of-range indices: {invalid}")
        return result

    train = _episode_list("train_episodes")
    val = _episode_list("val_episodes")
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f"train and validation episodes overlap: {overlap}")
    return EpisodeSplit(resolved, train, val, actual_hash, payload)
