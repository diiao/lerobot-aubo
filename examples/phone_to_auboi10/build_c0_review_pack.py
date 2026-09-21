#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the offline C0 human-review evidence pack; never overwrite output."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.bamboo_sorting.c0_post_capture_review import build_c0_review_pack

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "c0_formal_single_strip_20260921_full",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "c0_final_review" / "c0_formal_single_strip_20260921_full",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_c0_review_pack(
        repo_root=REPO_ROOT,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    print(f"Created review pack: {args.output_root}")
    for episode in manifest.episodes:
        print(
            f"episode {episode.aggregate_episode_index:02d}: "
            f"frames={episode.frame_count}, ON={episode.gripper_on_frame}, "
            f"OFF={episode.gripper_off_frame}, split={episode.split_name}, "
            f"source_outcome={episode.source_human_outcome}, review=pending"
        )
    print("Human outcomes pending; VLM shadow not run; training/inference/execution unauthorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
