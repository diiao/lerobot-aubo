#!/usr/bin/env python
"""聚合新视角的多个录制批次，用于纯 ACT 训练。

用法:
    cd examples/phone_to_auboi10
    python aggregate.py

默认自动发现 bamboo_newview_s01、s02...，也可通过 DATASET_SOURCES
传入逗号分隔的 repo_id。每次增加新批次后重新运行；脚本会先删除旧聚合输出再重建
（LeRobotDataset.create 用 exist_ok=False，输出目录已存在会报错）。

注意: SRC 里的每个批次必须用相同的相机配置、处理器链、FPS、TASK_DESCRIPTION
录制，否则 aggregate 的 validate_all_metadata 会报 features 不一致。
"""

import json
import os
import re
import shutil

import numpy as np
import pyarrow.parquet as pq

# 纯本地数据集，禁用 HuggingFace Hub 联网查询（否则会去 Hub 查 refs，本地无凭证时 401）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

DST = os.environ.get("OUTPUT_DATASET_PATH", "./datasets/bamboo_newview_full")


def discover_sources() -> list[str]:
    configured = os.environ.get("DATASET_SOURCES", "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]

    dataset_root = HF_LEROBOT_HOME / "datasets"
    return [
        f"./datasets/{path.name}"
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and re.fullmatch(r"bamboo_newview_s\d+", path.name)
    ]


def validate_image_stats(dataset_root) -> None:
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    for key in ("observation.images.handeye", "observation.images.fixed"):
        std = stats[key]["std"]
        min_std = min(float(channel[0][0]) for channel in std)
        if min_std < 0.02:
            raise RuntimeError(
                f"{key} std={std} 异常偏小；请确认 uint8 图像统计修复已生效后再训练"
            )


def validate_gripper_labels(dataset_root) -> None:
    """Require persistent suction state labels, never the old neutral command 50."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    action_names = info["features"]["action"]["names"]
    try:
        gripper_index = action_names.index("ee.gripper_pos")
    except ValueError as exc:
        raise RuntimeError("action features 中缺少 ee.gripper_pos") from exc

    invalid_values: set[float] = set()
    seen_values: set[float] = set()

    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=["action"], batch_size=65_536):
            actions = batch.column(0)
            action_width = actions.type.list_size
            values = actions.values.to_numpy(zero_copy_only=False).reshape(-1, action_width)
            gripper = values[:, gripper_index]
            seen_values.update(float(value) for value in np.unique(gripper))
            invalid = gripper[~(np.isclose(gripper, 0.0) | np.isclose(gripper, 100.0))]
            invalid_values.update(float(value) for value in np.unique(invalid))

    if not seen_values:
        raise RuntimeError("聚合数据集中没有找到 action 标签")
    if invalid_values:
        preview = sorted(invalid_values)[:10]
        raise RuntimeError(
            f"夹爪标签包含非 0/100 值 {preview}；不能混入旧的 50 中立命令数据"
        )
    if not ({0.0, 100.0} <= seen_values):
        raise RuntimeError(
            f"夹爪标签只包含 {sorted(seen_values)}；每批数据应同时包含吸取和释放状态"
        )


def main():
    sources = discover_sources()
    if not sources:
        raise RuntimeError(
            "没有发现 bamboo_newview_sXX 批次；请先录制，或设置 DATASET_SOURCES"
        )
    source_roots = [(HF_LEROBOT_HOME / source).resolve() for source in sources]
    dst_root = (HF_LEROBOT_HOME / DST).resolve()
    if dst_root in source_roots:
        raise ValueError("聚合输出不能同时出现在源数据集列表中")

    # 先验证每个源批次，避免无效批次导致已有聚合输出被白白删除。
    for source, source_root in zip(sources, source_roots, strict=True):
        if not source_root.is_dir():
            raise FileNotFoundError(f"源数据集不存在: {source_root}")
        print(f"检查源批次: {source}")
        validate_image_stats(source_root)
        validate_gripper_labels(source_root)

    # merge_datasets 内部用 create(exist_ok=False)，输出目录已存在会 FileExistsError，先删
    if dst_root.exists():
        print(f"删除旧的聚合输出: {dst_root}")
        shutil.rmtree(dst_root)

    print(f"合并: {sources} -> {DST}")
    datasets = [LeRobotDataset(repo_id=name) for name in sources]
    merge_datasets(datasets, output_repo_id=DST)

    m = LeRobotDataset(repo_id=DST)
    validate_image_stats(dst_root)
    validate_gripper_labels(dst_root)
    print(f"合并完成: {m.meta.total_episodes} episodes, {m.meta.total_frames} frames")
    print(f"输出位置: {dst_root}")


if __name__ == "__main__":
    main()
