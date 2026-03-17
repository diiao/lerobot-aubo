#!/usr/bin/env python3
"""
诊断脚本：验证相机到策略的数据流
用法：
    conda run -n lerobot python diagnose_camera_policy.py \
        --robot.type=aubo_i10 \
        --robot.id=aubo_i10 \
        --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': '/dev/video0', 'width': 640, 'height': 480, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': '/dev/video2', 'width': 640, 'height': 480, 'fps': 30}}" \
        --policy_path="/home/ninz/program/lerobot-aubo/src/lerobot/scripts/outputs/train/act1_so101_test/checkpoints/last/pretrained_model"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="诊断相机-策略数据流")
    parser.add_argument("--policy_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    args, unknown = parser.parse_known_args()

    policy_path = Path(args.policy_path)
    device = args.device

    print("=" * 70)
    print("相机到策略数据流诊断")
    print("=" * 70)

    # ======================== Step 1: 检查模型配置 ========================
    print("\n[Step 1] 检查模型配置...")
    config_path = policy_path / "config.json"
    if not config_path.exists():
        print(f"  ❌ 模型配置不存在: {config_path}")
        return
    with open(config_path) as f:
        config = json.load(f)

    input_features = config.get("input_features", {})
    print(f"  模型期望的输入特征:")
    visual_keys = []
    state_keys = []
    for key, feat in input_features.items():
        feat_type = feat.get("type", "UNKNOWN")
        feat_shape = feat.get("shape", [])
        print(f"    {key}: type={feat_type}, shape={feat_shape}")
        if feat_type == "VISUAL":
            visual_keys.append(key)
        elif feat_type == "STATE":
            state_keys.append(key)

    if not visual_keys:
        print("  ⚠️ 模型没有视觉输入特征！模型可能不需要相机图像。")
    else:
        print(f"  ✅ 模型需要 {len(visual_keys)} 个相机输入: {visual_keys}")

    # ======================== Step 2: 检查预处理器 ========================
    print("\n[Step 2] 检查预处理器配置...")
    preprocessor_path = policy_path / "policy_preprocessor.json"
    if preprocessor_path.exists():
        with open(preprocessor_path) as f:
            preprocessor_config = json.load(f)
        for step in preprocessor_config.get("steps", []):
            step_name = step.get("registry_name", "unknown")
            step_config = step.get("config", {})
            print(f"    预处理步骤: {step_name}")
            if step_name == "rename_observations_processor":
                rename_map = step_config.get("rename_map", {})
                print(f"      重命名映射: {rename_map if rename_map else '(空，不重命名)'}")
            if step_name == "normalizer_processor":
                norm_features = step_config.get("features", {})
                norm_map = step_config.get("norm_map", {})
                print(f"      归一化模式: {norm_map}")
                print(f"      归一化特征:")
                for k, v in norm_features.items():
                    print(f"        {k}: {v}")
    else:
        print(f"  ⚠️ 预处理器配置不存在: {preprocessor_path}")

    # ======================== Step 3: 检查归一化统计量 ========================
    print("\n[Step 3] 检查归一化统计量...")
    stats_path = policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if stats_path.exists():
        try:
            import safetensors.torch as st
            tensors = st.load_file(str(stats_path))
            stats_keys = set()
            for k in tensors:
                feature_key, stat_name = k.rsplit(".", 1)
                stats_keys.add(feature_key)

            print(f"  统计量包含的特征:")
            for sk in sorted(stats_keys):
                mean_key = f"{sk}.mean"
                std_key = f"{sk}.std"
                if mean_key in tensors and std_key in tensors:
                    mean_val = tensors[mean_key]
                    std_val = tensors[std_key]
                    print(f"    {sk}:")
                    print(f"      mean: shape={mean_val.shape}, range=[{mean_val.min():.4f}, {mean_val.max():.4f}]")
                    print(f"      std:  shape={std_val.shape}, range=[{std_val.min():.4f}, {std_val.max():.4f}]")

            # 验证模型的视觉特征 key 在统计量里存在
            for vk in visual_keys:
                if vk in stats_keys:
                    print(f"  ✅ 视觉特征 '{vk}' 在归一化统计量中存在")
                else:
                    print(f"  ❌ 视觉特征 '{vk}' 在归一化统计量中不存在！这会导致图像不被归一化！")
        except ImportError:
            print("  ⚠️ 无法导入 safetensors，跳过统计量检查")
    else:
        print(f"  ⚠️ 归一化统计量文件不存在: {stats_path}")

    # ======================== Step 4: 连接相机并测试 ========================
    print("\n[Step 4] 测试相机连接和图像采集...")
    try:
        # 尝试直接连接相机
        import cv2
        camera_tests = [
            ("/dev/video0", "handeye"),
            ("/dev/video2", "fixed"),
        ]

        camera_images = {}
        for dev_path, name in camera_tests:
            print(f"\n  测试相机 {name} ({dev_path}):")
            cap = cv2.VideoCapture(dev_path)
            if not cap.isOpened():
                print(f"    ❌ 无法打开相机 {dev_path}")
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # 读取几帧让相机稳定
            for _ in range(10):
                cap.read()
            time.sleep(0.5)

            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"    ❌ 相机 {name} 无法读取帧")
                cap.release()
                continue

            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"    ✅ 读取成功: shape={frame_rgb.shape}, dtype={frame_rgb.dtype}")
            print(f"    像素范围: min={frame_rgb.min()}, max={frame_rgb.max()}, mean={frame_rgb.mean():.2f}")

            # 检查是否是黑屏
            if frame_rgb.mean() < 5:
                print(f"    ⚠️ 图像几乎全黑! (mean={frame_rgb.mean():.2f}) — 请检查相机是否正常工作")
            elif frame_rgb.std() < 5:
                print(f"    ⚠️ 图像几乎没有变化! (std={frame_rgb.std():.2f}) — 可能是相机被遮挡")
            else:
                print(f"    ✅ 图像质量正常 (mean={frame_rgb.mean():.2f}, std={frame_rgb.std():.2f})")

            camera_images[name] = frame_rgb
            cap.release()

            # 保存诊断图像
            diag_path = f"/tmp/diag_{name}.jpg"
            cv2.imwrite(diag_path, frame)
            print(f"    已保存诊断图像: {diag_path}")

    except Exception as e:
        print(f"  ❌ 相机测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ======================== Step 5: 模拟推理数据流 ========================
    if camera_images:
        print("\n[Step 5] 模拟推理数据流...")
        print("  模拟 observation 构建...")

        # 模拟 robot.get_observation() 的输出
        obs_dict = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
        }

        for name, img in camera_images.items():
            obs_dict[f"observation.image.{name}"] = img

        print(f"  observation 字典 keys: {list(obs_dict.keys())}")

        # 模拟 build_dataset_frame
        print("\n  模拟 build_dataset_frame 转换...")
        observation_frame = {}
        # State
        state_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]
        observation_frame["observation.state"] = np.array(
            [obs_dict[n] for n in state_names], dtype=np.float32
        )
        # Images
        for name in camera_images:
            dataset_key = f"observation.images.observation.image.{name}"
            obs_key = f"observation.image.{name}"
            observation_frame[dataset_key] = obs_dict[obs_key]

        print(f"  observation_frame keys: {list(observation_frame.keys())}")
        for k, v in observation_frame.items():
            if isinstance(v, np.ndarray):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")

        # 验证 key 匹配
        print("\n  验证 key 匹配:")
        for vk in visual_keys:
            if vk in observation_frame:
                print(f"    ✅ '{vk}' 在 observation_frame 中存在")
            else:
                print(f"    ❌ '{vk}' 在 observation_frame 中不存在!")
                print(f"       可用的 keys: {list(observation_frame.keys())}")

        for sk in state_keys:
            if sk in observation_frame:
                print(f"    ✅ '{sk}' 在 observation_frame 中存在")
            else:
                print(f"    ❌ '{sk}' 在 observation_frame 中不存在!")

        # ======================== Step 6: 尝试加载模型并做推理 ========================
        print("\n[Step 6] 尝试加载模型并执行推理...")
        try:
            from lerobot.policies.utils import prepare_observation_for_inference

            # 模拟 prepare_observation_for_inference
            obs_for_inference = {}
            for name, value in observation_frame.items():
                tensor = torch.from_numpy(value)
                if "image" in name:
                    tensor = tensor.type(torch.float32) / 255
                    tensor = tensor.permute(2, 0, 1).contiguous()
                tensor = tensor.unsqueeze(0)
                tensor = tensor.to(device)
                obs_for_inference[name] = tensor

            print(f"  推理输入 tensor:")
            for k, v in obs_for_inference.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, range=[{v.min():.4f}, {v.max():.4f}]")

            # 尝试加载完整的策略并做推理
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            print(f"\n  加载策略...")
            policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
            policy_cfg.pretrained_path = str(policy_path)

            from lerobot.policies.act.modeling_act import ACTPolicy
            policy = ACTPolicy.from_pretrained(
                str(policy_path),
                config=policy_cfg,
            )
            print(f"  ✅ 策略加载成功: {type(policy).__name__}")
            print(f"  策略 image_features: {list(policy.config.image_features.keys())}")

            # 准备 batch 并调用 select_action
            # 需要把 OBS_IMAGES 注入到 batch 中
            from lerobot.utils.constants import OBS_IMAGES
            batch = dict(obs_for_inference)
            batch["task"] = ""
            batch["robot_type"] = "aubo_i10"

            # 模拟 policy 的 predict_action_chunk
            if policy.config.image_features:
                image_feature_keys = list(policy.config.image_features.keys())
                print(f"\n  策略期望从 batch 中获取的图像 keys: {image_feature_keys}")
                for ik in image_feature_keys:
                    if ik in batch:
                        print(f"    ✅ batch['{ik}']: shape={batch[ik].shape}")
                    else:
                        print(f"    ❌ batch['{ik}'] 不存在! 模型无法获取此相机图像!")
                        print(f"       batch 可用 keys: {[k for k in batch if isinstance(batch.get(k), torch.Tensor)]}")

                # 注入 OBS_IMAGES 列表
                batch[OBS_IMAGES] = [batch[key] for key in image_feature_keys if key in batch]
                print(f"  batch['{OBS_IMAGES}']: list of {len(batch[OBS_IMAGES])} images")

            # 执行推理
            print(f"\n  执行模型推理...")
            with torch.inference_mode(), torch.no_grad():
                action = policy.select_action(batch)

            print(f"  ✅ 推理成功!")
            print(f"  输出 action: shape={action.shape}, values={action.cpu().numpy()}")

            # 做第二次推理用不同图像（全黑），看 action 是否变化
            print(f"\n  对比测试: 用全黑图像进行推理...")
            policy.reset()
            batch_black = dict(batch)
            for ik in image_feature_keys:
                if ik in batch_black:
                    batch_black[ik] = torch.zeros_like(batch_black[ik])
            if OBS_IMAGES in batch_black:
                batch_black[OBS_IMAGES] = [batch_black[key] for key in image_feature_keys if key in batch_black]

            with torch.inference_mode(), torch.no_grad():
                action_black = policy.select_action(batch_black)

            print(f"  全黑图像 action: values={action_black.cpu().numpy()}")
            diff = (action.cpu() - action_black.cpu()).abs().mean().item()
            print(f"  两次推理的 action 差异: {diff:.6f}")
            if diff < 0.01:
                print(f"  ⚠️ 图像变化但 action 几乎不变! 模型可能没有有效利用图像特征!")
                print(f"     可能原因: 1) 训练不充分 2) 图像归一化不匹配 3) 模型过拟合到固定动作")
            else:
                print(f"  ✅ 模型对不同图像产生不同 action，图像特征被正常使用")

        except Exception as e:
            print(f"  ❌ 模型推理测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
