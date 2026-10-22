#!/usr/bin/env python
"""Offline wrist distributions and independent two-point measurement review.

Business logic lives in lerobot.bamboo_sorting.diagnostics.joint_wrist_distribution."""
import argparse
import json
from pathlib import Path
from lerobot.bamboo_sorting.diagnostics.joint_wrist_distribution import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','plots','errors'),default='prepare')
    parser.add_argument('--measurements',type=Path);parser.add_argument('--annotations',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.stage=='prepare':
        if args.measurements is None:parser.error('--measurements required')
        prepare(args.measurements.resolve(),args.output.resolve())
    elif args.stage=='errors':
        if args.annotations is None:parser.error('--annotations required')
        import_manual_axes(args.output.resolve(),args.annotations.resolve())
    else:plots(args.output.resolve())
    print(json.dumps(dict(output=str(args.output),stage=args.stage,regression_run=False)))


if __name__=='__main__':
    main()
