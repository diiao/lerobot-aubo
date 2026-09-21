"""Offline-only 6D fixed-J5 AUBO SmolVLA adapter; no robot access or execution."""

import numpy as np
import torch

from .aubo_joint_contract import JOINT_FIELDS, JOINT_IMAGE_KEYS, JOINT_TASK, joint_command, joint_contract_record


def make_joint_smolvla_config(*, device="cpu"):
    """Explicit AUBO features for a later base-checkpoint fine-tune.

    This constructs configuration only; it does not load weights or train.
    The six coordinates mean AUBO J1/J2/J3/J4/J6/suction, NOT SO100 joints.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    return SmolVLAConfig(
        device=device,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                        **{f"observation.images.{k}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
                           for k in JOINT_IMAGE_KEYS}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        action_representation="absolute", relative_action_stats=None, execution_loss_fraction=None,
        chunk_size=50, n_action_steps=1, num_steps=10,
        adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False,
        push_to_hub=False,
    )


class SmolVLAJointOfflineAdapter:
    def __init__(self, policy, preprocessor, postprocessor, *, contract):
        if contract != joint_contract_record():
            raise ValueError("explicit AuboI10JointFixedJ5V1 contract is required")
        config = policy.config
        expected = {"observation.state", *(f"observation.images.{k}" for k in JOINT_IMAGE_KEYS)}
        if set(config.input_features) != expected or set(config.output_features) != {"action"}:
            raise ValueError("joint policy feature keys mismatch")
        if config.input_features["observation.state"].shape != (6,) or config.output_features["action"].shape != (6,):
            raise ValueError("joint policy must have 6D state and 6D action")
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
        if not isinstance(state, np.ndarray) or state.dtype != np.float32 or state.shape != (6,):
            raise ValueError("joint state must be float32 [6]")
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
        if (array.ndim != 3 or array.shape[0] != 1 or array.shape[1] < 1 or array.shape[2] != 6
                or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all()):
            raise ValueError("joint prediction must be finite floating [1, horizon, 6]")
        self.raw_actions = array[0].copy()
        decoded = self.raw_actions.copy()
        decoded[:, 5] = np.where(decoded[:, 5] >= 50.0, 100.0, 0.0)
        return decoded  # Numerical prediction only; no safety/execution claim.
