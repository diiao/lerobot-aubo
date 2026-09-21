#!/usr/bin/env python
"""Read-only joint dataset/command audit. No models, network or hardware."""
import argparse
import json

from lerobot.bamboo_sorting.aubo_joint_audit import audit_joint_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit_joint_dataset(args.dataset_root, args.evidence_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
