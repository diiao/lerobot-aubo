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

"""Create the offline C0 digest and pre-training readiness pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.bamboo_sorting.c0_pretraining_gate import build_c0_pretraining_pack

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "c0_formal_single_strip_20260921_full",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "c0_final_review" / "c0_formal_single_strip_20260921_full",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "c0_pretraining" / "c0_formal_single_strip_20260921_full",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_c0_pretraining_pack(
        dataset_root=args.dataset_root,
        review_root=args.review_root,
        output_root=args.output_root,
    )
    print(f"Created pre-training pack: {args.output_root}")
    print(f"C0 review: {'PASS' if report['c0_capture_and_human_review_passed'] else 'FAIL'}")
    print(f"C1 collection: {report['collection_decision']}")
    print(f"Training: {report['training_decision']}")
    for blocker in report["training_blockers"]:
        print(f"  - {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
