"""Offline AUBO absolute-joint PI05 preparation; no robot or service access."""

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

from ..aubo_joint_contract import JOINT_DIM, JOINT_TASK
from .joint_target_policy import SmolVLAJointTargetOfflineAdapter

PRECISION_FILE = "aubo_joint_precision.json"
FP32_EXPERT_PRECISION = {
    "schema": "AuboPI05Precision",
    "trainable_storage_dtype": "float32",
    "compute_dtype": "bfloat16",
    "frozen_storage": "preserve checkpoint",
}


def configure_joint_pi05_precision(policy):
    """Keep the frozen backbone unchanged; accumulate expert updates in FP32.

    The sidecar is restored BEFORE loading weights, so strict reload cannot
    silently round the saved expert back into the constructor's BF16 tensors.
    """
    backbone = policy.model.paligemma_with_expert.paligemma
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("FP32 expert precision requires a frozen visual-language backbone")
    backbone_parameters = {id(p) for p in backbone.parameters()}
    for parameter in policy.parameters():
        # Inference callers may have disabled every gradient before loading.
        # Storage precision must still be restored for the entire expert.
        if id(parameter) not in backbone_parameters:
            parameter.data = parameter.data.float()
    policy.aubo_joint_precision = dict(FP32_EXPERT_PRECISION)


def joint_pi05_compute_context(policy):
    # The validation override puts BOTH controls under identical autocast.
    enabled = (getattr(policy, "aubo_joint_precision", None) is not None
               or getattr(policy, "aubo_validation_bf16_compute", False))
    if enabled and next(policy.parameters()).device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def predict_joint_pi05_chunk(policy, batch):
    with joint_pi05_compute_context(policy):
        return policy.predict_action_chunk(batch)


class PI05JointTargetOfflineAdapter(SmolVLAJointTargetOfflineAdapter):
    """Reuse the model-independent RGB/state/target boundary with PI05 processors.

    The parent only validates seven absolute coordinates, converts RGB to CHW,
    calls the supplied processors/policy, and decodes the gripper. It does not
    construct SmolVLA processors or alter the predicted joint coordinates.
    """

    def __init__(self, policy, preprocessor, postprocessor, *, contract):
        config = policy.config
        expected = make_joint_pi05_config(device=config.device)
        if (config.type != 'pi05' or config.dtype != 'bfloat16'
                or config.chunk_size != 50 or config.max_state_dim != 32 or config.max_action_dim != 32
                or config.normalization_mapping != expected.normalization_mapping):
            raise ValueError('explicit AUBO PI05 inference configuration required')
        super().__init__(policy, preprocessor, postprocessor, contract=contract)

    def __call__(self, frame):
        with joint_pi05_compute_context(self.policy):
            return super().__call__(frame)


def make_joint_pi05_config(*, device="cpu"):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    return PI05Config(
        device=device, dtype="bfloat16", chunk_size=50, n_action_steps=1,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(JOINT_DIM,)),
            **{f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
               for name in ("target_global_rgb", "grasp_rgb")},
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(JOINT_DIM,))},
        freeze_vision_encoder=True, train_expert_only=True, gradient_checkpointing=True,
        compile_model=False, push_to_hub=False,
    )


def train_quantile_stats(prepared):
    """Read selected training rows only; keep original metadata and held-out data intact."""
    import pyarrow.parquet as pq

    train, validation = prepared["train"], prepared["validation"]
    if set(train["scenes"]) & set(validation["scenes"]):
        raise ValueError("scene leakage between training and validation")
    arrays = {key: [] for key in ("observation.state", "action")}
    for record in train.get("sources", [train]):
        files = sorted((Path(record["root"]) / "data").glob("chunk-*/*.parquet"))
        if not files:
            raise ValueError("training source has no Parquet data")
        data = pq.read_table(files, columns=["episode_index", *arrays]).to_pydict()
        selected = np.isin(data["episode_index"], record["episodes"])
        if int(selected.sum()) != record["frames"]:
            raise ValueError("selected training frame count changed")
        for key in arrays:
            values = np.asarray(data[key], dtype=np.float64)[selected]
            if values.shape != (record["frames"], JOINT_DIM) or not np.isfinite(values).all():
                raise ValueError("finite seven-dimensional training values required")
            arrays[key].append(values)
    stats = {}
    for key, parts in arrays.items():
        values = np.concatenate(parts)
        if len(values) != train["frames"]:
            raise ValueError("training total frame count changed")
        q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
        if np.any(q99 <= q01):
            raise ValueError(f"degenerate training quantiles: {key}")
        stats[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                      "min": values.min(0).tolist(), "max": values.max(0).tolist(),
                      "q01": q01.tolist(), "q99": q99.tolist()}
        for field in ("mean", "std", "min", "max"):
            if not np.allclose(stats[key][field], prepared["train_only_stats"][key][field],
                               rtol=1e-5, atol=1e-6):
                raise ValueError(f"training statistics changed: {key}/{field}")
    return stats


def make_public_paligemma_tokenizer(directory):
    """Convert the official anonymous SentencePiece model, checking OpenPI token IDs."""
    import sentencepiece
    from transformers import AutoTokenizer, GemmaTokenizer

    directory = Path(directory).resolve()
    source = directory / "tokenizer.model"
    reference = sentencepiece.SentencePieceProcessor(model_file=str(source))
    tokenizer = GemmaTokenizer(vocab_file=str(source), add_bos_token=True, add_eos_token=False)
    # Check the text actually supplied to PI05, plus whitespace and Unicode cases.
    prompts = [f"Task: {task}, State: {' '.join(map(str, state))};\nAction: "
               for task in (JOINT_TASK, "pick A", "竹条")
               for state in ([128] * 32, list(range(32)), [-1, 0, 255, 256] * 8)]
    prompts.extend([" hello\nworld ", "A_B", ""])
    for text in prompts:
        if tokenizer.encode(text) != reference.encode(text, add_bos=True):
            raise ValueError("Gemma/OpenPI SentencePiece encoding mismatch")
    config_file = directory / "tokenizer_config.json"
    if not config_file.exists():
        tokenizer.save_pretrained(directory)
    # Production TokenizerProcessorStep uses AutoTokenizer's default fast implementation.
    reloaded = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    for text in prompts:
        if reloaded.encode(text) != reference.encode(text, add_bos=True):
            raise ValueError("saved AutoTokenizer/OpenPI encoding mismatch")
    if (reloaded.pad_token_id, reloaded.bos_token_id, reloaded.eos_token_id) != (0, 2, 1):
        raise ValueError("unexpected PaliGemma special token IDs")
    return {"reference": "OpenPI PaligemmaTokenizer / big_vision/paligemma_tokenizer.model",
            "comparison_cases": len(prompts), "encoding_equal": True,
            "vocab_size": reloaded.vocab_size, "pad_id": 0, "bos_id": 2, "eos_id": 1}


def make_joint_pi05_processors(config, stats, tokenizer_path):
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import (AddBatchDimensionProcessorStep, DeviceProcessorStep,
        NormalizerProcessorStep, PolicyProcessorPipeline, RenameObservationsProcessorStep,
        TokenizerProcessorStep, UnnormalizerProcessorStep)
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    expected = make_joint_pi05_config(device=config.device)
    if (config.input_features != expected.input_features or config.output_features != expected.output_features
            or config.normalization_mapping != expected.normalization_mapping
            or config.chunk_size != 50 or config.max_state_dim != 32 or config.max_action_dim != 32):
        raise ValueError("explicit absolute-joint AUBO PI05 configuration required")
    for key in ("observation.state", "action"):
        low, high = (np.asarray(stats[key][field]) for field in ("q01", "q99"))
        if low.shape != (JOINT_DIM,) or high.shape != low.shape or not np.isfinite([low, high]).all() or np.any(high <= low):
            raise ValueError("finite nondegenerate seven-dimensional quantiles required")
    tokenizer_path = Path(tokenizer_path).resolve()
    if not (tokenizer_path / "tokenizer_config.json").is_file():
        raise ValueError("a prepared LOCAL tokenizer is required")
    pre = PolicyProcessorPipeline(steps=[
        RenameObservationsProcessorStep(rename_map={}), AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(features={**config.input_features, **config.output_features},
                                norm_map=config.normalization_mapping, stats=stats),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(tokenizer_name=str(tokenizer_path), max_length=config.tokenizer_max_length,
                               padding_side="right", padding="max_length"),
        DeviceProcessorStep(device=config.device),
    ], name="policy_preprocessor")
    post = PolicyProcessorPipeline(steps=[
        UnnormalizerProcessorStep(features=config.output_features, norm_map=config.normalization_mapping, stats=stats),
        DeviceProcessorStep(device="cpu"),
    ], name="policy_postprocessor", to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return pre, post


def masked_joint_loss(losses, action_is_pad):
    """Mean over real timesteps and seven AUBO coordinates; exclude model padding."""
    if losses.ndim != 3 or losses.shape[-1] < JOINT_DIM:
        raise ValueError("per-coordinate losses must have shape [batch, time, >=7]")
    if action_is_pad.dtype != torch.bool or action_is_pad.shape != losses.shape[:2]:
        raise ValueError("boolean action_is_pad must match batch and time")
    valid = ~action_is_pad
    if not valid.any():
        raise ValueError("batch contains no valid action steps")
    return losses[..., :JOINT_DIM][valid].mean()


def joint_pi05_loss(policy, batch):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    with joint_pi05_compute_context(policy):
        images, image_masks = policy._preprocess_images(batch)
        losses = policy.model(images, image_masks, batch[OBS_LANGUAGE_TOKENS],
                              batch[OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_action(batch))
    return masked_joint_loss(losses, batch["action_is_pad"])


def load_joint_pi05_weights(policy, path):
    """Strict loading, including safetensors tied weights; never allow random fallback."""
    from safetensors import safe_open
    from safetensors.torch import load_model

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Official pi05_base saves the inner model; our checkpoints save the policy.
    # load_model also verifies keys omitted by safetensors for tied parameters.
    with safe_open(path, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
    wrapped = [key.startswith("model.") for key in keys]
    if not keys or (any(wrapped) and not all(wrapped)):
        raise ValueError("unsupported mixed checkpoint key layout")
    precision_path = path.parent / PRECISION_FILE
    if precision_path.exists():
        if json.loads(precision_path.read_text()) != FP32_EXPERT_PRECISION:
            raise ValueError("unsupported AUBO PI05 precision contract")
        configure_joint_pi05_precision(policy)
    elif all(wrapped):
        # Old checkpoints have BF16 expert matrices. A new FP32 checkpoint
        # without its contract must fail instead of silently discarding updates.
        with safe_open(path, framework="pt", device="cpu") as stream:
            for key in keys:
                if ".gemma_expert.model.layers." in key and ".self_attn.q_proj.weight" in key:
                    if stream.get_slice(key).get_dtype() == "F32":
                        raise ValueError("FP32 expert checkpoint is missing its precision contract")
    load_model(policy if all(wrapped) else policy.model, str(path), strict=True, device="cpu")
