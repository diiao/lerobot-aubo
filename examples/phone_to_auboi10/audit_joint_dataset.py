#!/usr/bin/env python
"""Read-only joint dataset/command audit. No models, network or hardware."""
import argparse
import json

from lerobot.bamboo_sorting.aubo_joint_audit import audit_joint_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--source-dataset-root", help="Explicit original absolute dataset root after relocation")
    parser.add_argument("--require-gripper-quality", action="store_true",
                        help="Exit nonzero if suction statistics or successful pick/release labels are incomplete")
    args = parser.parse_args(argv)
    report = audit_joint_dataset(args.dataset_root, args.evidence_root, source_dataset_root=args.source_dataset_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.require_gripper_quality and not report["gripper_quality_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
