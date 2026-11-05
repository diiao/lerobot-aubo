"""Offline-only 7D legacy-teleop AUBO SmolVLA adapter; no robot access or execution."""

import json
from pathlib import Path

import numpy as np
import torch

from ..aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, JOINT_FIELDS, JOINT_IMAGE_KEYS, JOINT_TASK, joint_command, joint_contract_record
from ..joint_gripper_quality import require_joint_normalization_stats


def make_joint_smolvla_processors(config, dataset_stats, *, target_conditioned=False):
    """Build matched normalization/inverse normalization from AUBO statistics.

    Does not train or load policy weights. Tokenizer loading may need its cache.
    Use training-only statistics, and save both pipelines with the checkpoint.
    """
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    expected = make_joint_smolvla_config(device=config.device, target_conditioned=target_conditioned)
    if (config.input_features != expected.input_features or config.output_features != expected.output_features
            or config.normalization_mapping != expected.normalization_mapping
            or config.action_representation != "absolute" or config.adapt_to_pi_aloha
            or config.use_delta_joint_actions_aloha):
        raise ValueError("joint processors require the explicit full-joint AUBO configuration")
    require_joint_normalization_stats(dataset_stats)
    return make_smolvla_pre_post_processors(config, dataset_stats)


def make_joint_smolvla_config(*, device="cpu", target_conditioned=False):
    """Explicit AUBO features for a later base-checkpoint fine-tune.

    This constructs configuration only; it does not load weights or train.
    The seven coordinates mean AUBO J1/J2/J3/J4/J5/J6/suction.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    image_keys = ("target_global_rgb", "grasp_rgb") if target_conditioned else JOINT_IMAGE_KEYS
    return SmolVLAConfig(
        device=device,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(JOINT_DIM,)),
                        **{f"observation.images.{k}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
                           for k in image_keys}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(JOINT_DIM,))},
        action_representation="absolute", relative_action_stats=None, execution_loss_fraction=None,
        chunk_size=50, n_action_steps=1, num_steps=10,
        adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False,
        push_to_hub=False,
    )


def load_joint_smolvla_processors(path, config, *, contract):
    """Continue an AUBO checkpoint with its saved normalization and inverse.

    Only device and tokenizer cache location may change. New demonstration
    statistics must not silently redefine what the pretrained outputs mean.
    """
    from lerobot.processor import PolicyProcessorPipeline, NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action,
    )

    path = Path(path)
    if json.loads((path / "aubo_joint_contract.json").read_text()) != contract:
        raise ValueError("base checkpoint contract differs from requested continuation")
    pre_config = json.loads((path / "policy_preprocessor.json").read_text())
    overrides = {"device_processor": {"device": config.device}}
    for step in pre_config["steps"]:
        if step.get("registry_name") == "tokenizer_processor":
            overrides["tokenizer_processor"] = {"tokenizer_name": config.vlm_model_name}
    pre = PolicyProcessorPipeline.from_pretrained(path, "policy_preprocessor.json", local_files_only=True,
        overrides=overrides, to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(path, "policy_postprocessor.json", local_files_only=True,
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    normalizers = [s for s in pre.steps if isinstance(s, NormalizerProcessorStep)]
    unnormalizers = [s for s in post.steps if isinstance(s, UnnormalizerProcessorStep)]
    if len(normalizers) != 1 or len(unnormalizers) != 1:
        raise ValueError("base checkpoint requires one matched normalizer and unnormalizer")
    normalizer, unnormalizer = normalizers[0], unnormalizers[0]
    if (normalizer.features != {**config.input_features, **config.output_features}
            or unnormalizer.features != config.output_features
            or normalizer.norm_map != config.normalization_mapping
            or unnormalizer.norm_map != config.normalization_mapping
            or normalizer.eps != unnormalizer.eps
            or normalizer.normalize_observation_keys is not None):
        raise ValueError("base checkpoint processor features/normalization differ from AUBO configuration")
    # Read effective tensors, not a potentially rounded/stale JSON stats file.
    saved = normalizer.state_dict()
    inverse = unnormalizer.state_dict()
    stats = {key: {name: saved[f"{key}.{name}"].tolist() for name in ("mean", "std", "min", "max")}
             for key in ("observation.state", "action")}
    require_joint_normalization_stats(stats)
    if any(f"action.{name}" not in inverse or not torch.equal(saved[f"action.{name}"], inverse[f"action.{name}"])
           for name in ("mean", "std", "min", "max")):
        raise ValueError("base checkpoint action normalization and inverse disagree")
    return pre, post, stats


class SmolVLAJointOfflineAdapter:
    image_keys = JOINT_IMAGE_KEYS
    contract_record = staticmethod(joint_contract_record)

    def __init__(self, policy, preprocessor, postprocessor, *, contract):
        if contract != self.contract_record():
            raise ValueError(f"explicit {self.contract_record()['schema_version']} contract is required")
        config = policy.config
        expected = {"observation.state", *(f"observation.images.{k}" for k in self.image_keys)}
        if set(config.input_features) != expected or set(config.output_features) != {"action"}:
            raise ValueError("joint policy feature keys mismatch")
        if config.input_features["observation.state"].shape != (JOINT_DIM,) or config.output_features["action"].shape != (JOINT_DIM,):
            raise ValueError("joint policy must have 7D state and 7D action")
        if getattr(config, "action_representation", "absolute") != "absolute":
            raise ValueError("joint policy must use absolute actions, not relative TCP")
        if getattr(config, "adapt_to_pi_aloha", False) or getattr(config, "use_delta_joint_actions_aloha", False):
            raise ValueError("AUBO joint policy cannot use Aloha transforms")
        for key in self.image_keys:
            if config.input_features[f"observation.images.{key}"].shape != (3, 480, 640):
                raise ValueError("joint policy requires the frozen RGB image shapes")
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.raw_actions = None

    def __call__(self, frame):
        self.raw_actions = None
        state = frame.get("observation.state")
        if not isinstance(state, np.ndarray) or state.dtype != np.float32 or state.shape != (JOINT_DIM,):
            raise ValueError("joint state must be float32 [7]")
        joint_command(dict(zip(JOINT_FIELDS, state, strict=True)))
        if frame.get("task") != JOINT_TASK:
            raise ValueError("joint task text mismatch")
        inputs = {"observation.state": torch.from_numpy(state.copy()), "task": JOINT_TASK}
        for name in self.image_keys:
            key = f"observation.images.{name}"
            image = frame[key]
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError("joint RGB input must be HWC uint8")
            inputs[key] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        # Deliberately omit dataset action and any teleoperation-only TCP fields.
        with torch.inference_mode():
            output = self.postprocessor(self.policy.predict_action_chunk(self.preprocessor(inputs)))
        array = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
        if (array.ndim != 3 or array.shape[0] != 1 or array.shape[1] < 1 or array.shape[2] != JOINT_DIM
                or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all()):
            raise ValueError("joint prediction must be finite floating [1, horizon, 7]")
        self.raw_actions = array[0].copy()
        decoded = self.raw_actions.copy()
        decoded[:, GRIPPER_INDEX] = np.where(decoded[:, GRIPPER_INDEX] >= 50.0, 100.0, 0.0)
        return decoded  # Numerical prediction only; no safety/execution claim.
