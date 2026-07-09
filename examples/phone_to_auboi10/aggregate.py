#!/usr/bin/env python
"""聚合多个录制批次的数据集为一个总集，用于训练。

用法:
    cd examples/phone_to_auboi10
    python aggregate.py

把 SRC 列表里所有数据集合并成 DST（phone_auboi10_full）。
每次加了新批次后，重新跑这个脚本即可——会先删掉旧的 phone_auboi10_full 再重建
（LeRobotDataset.create 用 exist_ok=False，输出目录已存在会报错）。

注意: SRC 里的每个批次必须用相同的相机配置、处理器链、FPS、TASK_DESCRIPTION
录制，否则 aggregate 的 validate_all_metadata 会报 features 不一致。
"""

import os
import shutil

# 纯本地数据集，禁用 HuggingFace Hub 联网查询（否则会去 Hub 查 refs，本地无凭证时 401）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME

# 要合并的所有批次（录了多少就列多少，顺序无所谓）。
# repo_id 必须带 "./datasets/" 前缀，和 record.py / train.py 的 LOCAL_DATASET_PATH 一致，
# 否则路径对不上（HF_LEROBOT_HOME 是 ~/.cache/huggingface/lerobot，数据在其 datasets/ 子目录下）。
SRC = [
    "./datasets/phone_auboi10",
    "./datasets/phone_auboi10_s2",
    "./datasets/phone_auboi10_s3",
]
DST = "./datasets/phone_auboi10_full"


def main():
    # merge_datasets 内部用 create(exist_ok=False)，输出目录已存在会 FileExistsError，先删
    dst_root = HF_LEROBOT_HOME / DST
    if dst_root.exists():
        print(f"删除旧的聚合输出: {dst_root}")
        shutil.rmtree(dst_root)

    print(f"合并: {SRC} -> {DST}")
    datasets = [LeRobotDataset(repo_id=name) for name in SRC]
    merge_datasets(datasets, output_repo_id=DST)

    m = LeRobotDataset(repo_id=DST)
    print(f"合并完成: {m.meta.total_episodes} episodes, {m.meta.total_frames} frames")
    print(f"输出位置: {dst_root}")


if __name__ == "__main__":
    main()
