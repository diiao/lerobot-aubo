#!/usr/bin/env python
"""Measure saved wrist RGB in explicit ROIs; no model, hardware or raw edits.

Manifest: {"schema":"joint_wrist_angle_samples", "samples":[{"id":"...",
"image":"relative.png", "gap_line":[[x,y],[x,y]], "roi_polygon":[[x,y],...],
"group":"expert|model", "episode_id":"...", "horizon_s":0.5,
"delta_j6_deg":1.2}]}.

Use --stage prepare --sources sources.json to extract time-aligned samples.
Sources: {"defaults":{"gap_line":[[x,y],[x,y]]},
"demonstrations":[{"root":"dataset", "evidence_root":"capture_evidence"}],
"runs":[{"root":"saved_trial", "action_fps":25}],
"horizons_s":[0.5,1.0], "before_close_s":[3,0], "sample_interval_s":0.2}.
Paths are relative to sources.json. Optional training_sources points to the
existing train/validation source manifest (uses its annotated episode subset).
Each source can override defaults and before_close_s; demonstrations can supply
episode_windows mapping episode index strings to [start,end] seconds before close.
Preclose windows are candidates, not automatically identified descent segments.
Measure first, review the overlays, then regress only explicitly accepted rows.
Angles are image projections, offsets are pixels, never J6 corrections.

Business logic lives in lerobot.bamboo_sorting.diagnostics.joint_wrist_angle.
"""
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
