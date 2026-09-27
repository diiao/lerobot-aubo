#!/usr/bin/env python
"""Compare static top-layer VLM predictions with human labels; no model or hardware access."""

import argparse
import json

from lerobot.bamboo_sorting.layer_order_eval import evaluate, validate_truth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Human-labeled scene JSONL")
    parser.add_argument("--predictions", help="VLM prediction JSONL")
    parser.add_argument("--validate-truth-only", action="store_true", help="Check human labels and image hashes")
    parser.add_argument("--split", choices=("development", "test"), default="test")
    args = parser.parse_args(argv)
    if args.validate_truth_only:
        if args.predictions:
            parser.error("--predictions cannot be used with --validate-truth-only")
        report = validate_truth(args.truth)
    else:
        if not args.predictions:
            parser.error("--predictions is required unless --validate-truth-only is set")
        report = evaluate(args.truth, args.predictions, split=args.split)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
