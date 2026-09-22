"""Offline-only 7D legacy-teleop AUBO SmolVLA adapter; no robot access or execution."""

import numpy as np
import torch

from .aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, JOINT_FIELDS, JOINT_IMAGE_KEYS, JOINT_TASK, joint_command, joint_contract_record
from .joint_gripper_quality import require_joint_normalization_stats


def make_joint_smolvla_processors(config, dataset_stats):
    """Build matched normalization/inverse normalization from AUBO statistics.

    Does not train or load policy weights. Tokenizer loading may need its cache.
    Use training-only statistics, and save both pipelines with the checkpoint.
    """
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    expected = make_joint_smolvla_config(device=config.device)
    if (config.input_features != expected.input_features or config.output_features != expected.output_features
            or config.normalization_mapping != expected.normalization_mapping
            or config.action_representation != "absolute" or config.adapt_to_pi_aloha
            or config.use_delta_joint_actions_aloha):
        raise ValueError("joint processors require the explicit full-joint AUBO configuration")
    require_joint_normalization_stats(dataset_stats)
    return make_smolvla_pre_post_processors(config, dataset_stats)


def make_joint_smolvla_config(*, device="cpu"):
    """Explicit AUBO features for a later base-checkpoint fine-tune.

    This constructs configuration only; it does not load weights or train.
    The seven coordinates mean AUBO J1/J2/J3/J4/J5/J6/suction.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    return SmolVLAConfig(
        device=device,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(JOINT_DIM,)),
                        **{f"observation.images.{k}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
                           for k in JOINT_IMAGE_KEYS}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(JOINT_DIM,))},
        action_representation="absolute", relative_action_stats=None, execution_loss_fraction=None,
        chunk_size=50, n_action_steps=1, num_steps=10,
        adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False,
        push_to_hub=False,
    )


class SmolVLAJointOfflineAdapter:
    def __init__(self, policy, preprocessor, postprocessor, *, contract):
        if contract != joint_contract_record():
            raise ValueError("explicit AuboI10JointLegacyTeleopV2 contract is required")
        config = policy.config
        expected = {"observation.state", *(f"observation.images.{k}" for k in JOINT_IMAGE_KEYS)}
        if set(config.input_features) != expected or set(config.output_features) != {"action"}:
            raise ValueError("joint policy feature keys mismatch")
        if config.input_features["observation.state"].shape != (JOINT_DIM,) or config.output_features["action"].shape != (JOINT_DIM,):
            raise ValueError("joint policy must have 7D state and 7D action")
        if getattr(config, "action_representation", "absolute") != "absolute":
            raise ValueError("joint policy must use absolute actions, not relative TCP")
        if getattr(config, "adapt_to_pi_aloha", False) or getattr(config, "use_delta_joint_actions_aloha", False):
            raise ValueError("AUBO joint policy cannot use Aloha transforms")
        for key in JOINT_IMAGE_KEYS:
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
        for name in JOINT_IMAGE_KEYS:
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
