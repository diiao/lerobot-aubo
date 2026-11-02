#!/usr/bin/env python3
"""AUBO PI05: read-only plan, zero-update GPU probe, or explicitly selected training."""

import argparse
import json
import math
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "probe", "validate", "train"), default="plan",
                        help="validate: paired <=256-step precision comparison on training observations; saves one isolated candidate")
    parser.add_argument("--preflight", type=Path, default=Path("artifacts/pi05_preparation/preflight"))
    parser.add_argument("--smoke-report", type=Path, default=Path("outputs/pi05_joint_smoke/result.json"))
    parser.add_argument("--base-path", type=Path, default=Path("models/pi05_base"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pi05_joint_three_four"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--precision", choices=("fp32-expert", "legacy"), default="fp32-expert")
    parser.add_argument("--evaluation-geometry", type=Path,
                        help="frozen offline geometry JSON for TCP metrics; never used for control")
    parser.add_argument("--batch-size", type=int,
                        help="default 16; paired precision diagnosis retains batch 8")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-frames-per-episode", type=int, default=32)
    parser.add_argument("--early-stopping", action="store_true",
                        help="opt in to stopping for persistent validation regression (default: warn only)")
    parser.add_argument("--probe-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 8 if args.stage == "validate" else 16
    if args.steps is None:
        args.steps = 128 if args.stage == "validate" else 30000
    if args.stage == "validate" and (args.steps > 256 or args.precision != "fp32-expert"):
        parser.error("validate requires <=256 steps and fp32-expert precision")
    if min(args.steps, args.batch_size, args.probe_batches, args.eval_interval) < 1:
        parser.error("steps, batch-size, probe-batches and eval-interval must be positive")
    if args.eval_frames_per_episode < 2:
        parser.error("eval-frames-per-episode must be at least 2")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("learning-rate must be finite and positive")
    if len(set(args.checkpoint_steps)) != len(args.checkpoint_steps) or any(s <= 0 or s >= args.steps for s in args.checkpoint_steps):
        parser.error("checkpoint-steps must be unique positive updates before final step")
    for key in ("preflight", "smoke_report", "base_path", "output"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    # Offline behavior applies only to this explicit subprocess, not shell/system configuration.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    from lerobot.bamboo_sorting.policy.pi05_joint_training import build_plan, run
    plan, prepared, provenance, config = build_plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.stage != "plan":
        run(args, plan, prepared, provenance, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
