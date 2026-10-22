#!/usr/bin/env python
"""Measure saved wrist RGB in explicit ROIs; no model, hardware or raw edits.

Business logic lives in lerobot.bamboo_sorting.diagnostics.joint_wrist_angle."""
import argparse
import json
from pathlib import Path
from lerobot.bamboo_sorting.diagnostics.joint_wrist_angle import *


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'measure', 'regress'), default='measure')
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-decisions', type=Path)
    args = parser.parse_args(argv)
    if args.stage == 'prepare':
        if args.sources is None:
            parser.error('--sources required')
        result = prepare_manifest(args.sources.resolve(), args.output.resolve())
    elif args.stage == 'measure':
        if args.manifest is None:
            parser.error('--manifest required')
        result = measure_manifest(args.manifest.resolve(), args.output.resolve())
    else:
        if args.review_decisions is None:
            parser.error('--review-decisions required')
        result = regress_reviewed(args.output.resolve(), args.review_decisions.resolve())
    print(json.dumps({'output': str(args.output), 'samples': len(result.get('samples', [])),
                      'causal_diagnosis': None}, ensure_ascii=False))


if __name__ == '__main__':
    main()
