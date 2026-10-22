#!/usr/bin/env python
"""Manual 25-Hz joint servo with asynchronous policy inference; no auto start."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys

from lerobot.bamboo_sorting.model_profiles import (
    BOTH_ORDERS_APPROACH_SCRIPT, BOTH_ORDERS_CHECKPOINT, BOTH_ORDERS_INFERENCE_SCRIPT,
    BOTH_ORDERS_MODEL_SHA, BOTH_ORDERS_REMOTE_ROOT, LEGACY_MODEL_SHA as MODEL_SHA,
    MIXED_CHECKPOINT, MIXED_INFERENCE_SCRIPT, MIXED_MODEL_SHA, MIXED_REMOTE_ROOT,
)
from lerobot.bamboo_sorting.policy.joint_target_policy import TargetInputSession
from lerobot.bamboo_sorting.runtime.continuous import MAX_TARGET_REOBSERVATIONS, execute_continuous
from lerobot.bamboo_sorting.runtime.joint_smooth import (
    DT, MAX_ACCEL, MAX_CAMERA_SKEW, MAX_PREDICTION_AGE, MAX_SPEED, MAX_TCP_SPEED,
    MAX_TRACKING_ERROR_DEG, MAX_WITHOUT_PREDICTION,
)
from lerobot.bamboo_sorting.runtime.station import CAMERA_SET, RPC_TIMEOUT_MS

__all__ = ['BOTH_ORDERS_MODEL_SHA', 'CAMERA_SET', 'DT', 'MIXED_MODEL_SHA', 'MODEL_SHA', 'main']


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--model',choices=('legacy-single','mixed-double','mixed-both-orders','target-pair','target-stack'),default='legacy-single')
    p.add_argument('--target-snapshot',type=Path,help='Reviewed seed snapshot; only its target selection is reused')
    p.add_argument('--target-selections',type=Path,help='Human-reviewed visible target selection JSON bound to the snapshot')
    p.add_argument('--target-id',help='Explicit target identity, e.g. s-001')
    p.add_argument('--target-sequence',nargs='+',help='target-stack: ordered remaining targets; automatically home between picks')
    p.add_argument('--target-preview',action='store_true',help='Read-only localhost view of same-request RGB and predicted target outline')
    p.add_argument('--track-target-motion',action='store_true',help='Experimental target-stack RGB motion references with current-foreground checks')
    p.add_argument('--paired-target-separation',action='store_true',
                   help='Explicit target-pair trial: separate current foreground using both reviewed target references')
    p.add_argument('--multi-target-separation',action='store_true',
                   help='Use current instances and all reviewed visible seeds for one stack pick')
    p.add_argument('--transport-age-aligned',action='store_true',
                   help='Target model trial: age-align transport actions without skipping the first release')
    p.add_argument('--cycles',type=int,help='Stop after this many close/open command cycles; defaults to the model setting')
    p.add_argument('--inference-timeout-s',type=float,
                   help='Wait for a model response; does not relax observation age or servo watchdog limits')
    p.add_argument('--max-seconds',type=float,help='Maximum run duration, or per-target duration with --target-sequence; completion stops earlier')
    p.add_argument('--latency-tolerant',action='store_true',
                   help='Target models: 1.5 s prediction age/watchdog; target-stack also: 400 ms camera age, 300 ms skew, 700 ms control gap, 250 ms state age; mixed-both-orders: 750 ms / 1 s and 400 ms / 300 ms; all: 5 s response wait')
    p.add_argument('--approach-age-aligned',action='store_true',
                   help='Experimental mixed-both-orders approach: age-align open-gripper targets; retain full pick/place cycles')
    args=p.parse_args(argv)
    target_mode=args.model in ('target-pair','target-stack')
    if args.target_preview and not target_mode:p.error('--target-preview requires a target model')
    if args.track_target_motion and args.model!='target-stack':p.error('--track-target-motion requires target-stack')
    sequence=args.target_sequence
    if sequence:
        if args.model!='target-stack' or args.target_id:
            p.error('--target-sequence requires target-stack and replaces --target-id')
        if len(set(sequence))!=len(sequence):p.error('target sequence must contain distinct identities')
        if args.cycles not in (None,len(sequence)):p.error('--cycles must match the target sequence length')
        args.target_id=sequence[0]
    if args.model=='target-stack':
        if args.paired_target_separation:p.error('target-stack uses multi-target separation')
        args.multi_target_separation=True
    if args.multi_target_separation and (not target_mode or args.paired_target_separation):
        p.error('--multi-target-separation requires a target model and excludes --paired-target-separation')
    if args.paired_target_separation and not target_mode:
        p.error('--paired-target-separation requires --model target-pair')
    if args.transport_age_aligned and not target_mode:
        p.error('--transport-age-aligned requires a target model')
    if args.cycles is not None and args.cycles<1:p.error('--cycles must be a positive integer')
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds<=0):
        p.error('--max-seconds must be finite and positive')
    inference_timeout_s=args.inference_timeout_s if args.inference_timeout_s is not None else (5. if args.latency_tolerant else 2.)
    if not math.isfinite(inference_timeout_s) or inference_timeout_s<=0:
        p.error('--inference-timeout-s must be finite and positive')
    if args.latency_tolerant and not (args.model=='mixed-both-orders' or target_mode):
        p.error('--latency-tolerant requires mixed-both-orders or a target model')
    approach=args.approach_age_aligned or target_mode
    if approach and not (args.model=='mixed-both-orders' or target_mode):p.error('--approach-age-aligned requires mixed-both-orders or a target model')
    mixed=args.model in ('mixed-double','mixed-both-orders') or target_mode
    return args,sequence,target_mode,approach,mixed,inference_timeout_s,p


def resolve_targets(args, sequence, target_mode, parser):
    """Load reviewed seeds and model options; no hardware or model access."""
    target_session=None;target_seed=None;target_options=None;target_setups=None
    if target_mode:
        if not all((args.target_snapshot,args.target_selections,args.target_id)):
            parser.error('target models require --target-snapshot, --target-selections and --target-id')
        if not sequence and args.cycles not in (None,1):parser.error('target models run exactly one selected-object cycle')
        from lerobot.bamboo_sorting.runtime.live_setup import prepare_live_setup
        setup_options={'paired_separation':True} if args.paired_target_separation else {}
        if args.multi_target_separation:
            setup_options={'multi_separation':True}
        if args.model=='target-stack':
            setup_options['model']='target-stack'
        if args.track_target_motion:
            setup_options['motion_tracking']=True
        if sequence:
            target_setups=[prepare_live_setup(args.target_snapshot,args.target_selections,target,
                remaining_target_ids=sequence[index:],**setup_options) for index,target in enumerate(sequence)]
            target_seed,target_options=target_setups[0]
        else:
            target_seed,target_options=prepare_live_setup(args.target_snapshot,args.target_selections,args.target_id,**setup_options)
        target_session=TargetInputSession(args.target_id)
    elif any((args.target_snapshot,args.target_selections,args.target_id)):
        parser.error('target selection requires a target model')
    return target_session,target_seed,target_options,target_setups


def resolve_config(args, sequence, target_mode, approach, mixed, inference_timeout_s, target_options):
    """Derive timing/limit settings and remote worker options from the mode."""
    expected_cycles=len(sequence) if sequence else (args.cycles if args.cycles is not None else (2 if mixed and not target_mode else 1))
    max_seconds=args.max_seconds if args.max_seconds is not None else (360 if mixed and not target_mode else 120)
    servo_command_time=.08 if mixed else DT
    max_camera_age_ms=250 if mixed else 100
    max_prediction_age=.5 if args.model=='mixed-both-orders' or target_mode else MAX_PREDICTION_AGE
    max_without_prediction=MAX_WITHOUT_PREDICTION
    max_camera_skew=MAX_CAMERA_SKEW
    max_control_gap=.35
    max_state_age=.1
    if args.latency_tolerant:
        max_prediction_age=1.5 if target_mode else .75
        max_without_prediction=1.5 if target_mode else 1.
        if not target_mode or args.model=='target-stack':
            max_camera_age_ms=400;max_camera_skew=.3
        if args.model=='target-stack':
            max_control_gap=.7;max_state_age=.25
    model_sha=MIXED_MODEL_SHA if mixed else MODEL_SHA
    pipe_options=({'remote_root':MIXED_REMOTE_ROOT,'checkpoint':MIXED_CHECKPOINT,
                   'expected_sha256':MIXED_MODEL_SHA,'inference_script':MIXED_INFERENCE_SCRIPT}
                  if mixed else {})
    if args.model=='mixed-both-orders':
        model_sha=BOTH_ORDERS_MODEL_SHA
        pipe_options={'remote_root':BOTH_ORDERS_REMOTE_ROOT,'checkpoint':BOTH_ORDERS_CHECKPOINT,
                      'expected_sha256':model_sha,'inference_script':BOTH_ORDERS_INFERENCE_SCRIPT}
        if approach:pipe_options.update(inference_script=BOTH_ORDERS_APPROACH_SCRIPT,return_action_chunk=True)
    if target_mode:
        pipe_options=target_options
        model_sha=target_options['expected_sha256']
    pipe_options['request_timeout_s']=inference_timeout_s
    return dict(expected_cycles=expected_cycles,max_seconds=max_seconds,servo_command_time=servo_command_time,
        max_camera_age_ms=max_camera_age_ms,max_prediction_age=max_prediction_age,
        max_without_prediction=max_without_prediction,max_camera_skew=max_camera_skew,
        max_control_gap=max_control_gap,max_state_age=max_state_age,mixed=mixed,
        model_sha=model_sha,pipe_options=pipe_options,inference_timeout_s=inference_timeout_s,
        approach=approach)


def build_plan(args, sequence, target_mode, target_seed, target_options, config):
    """Describe the requested run without connecting hardware or a model."""
    expected_cycles=config['expected_cycles'];max_seconds=config['max_seconds']
    plan={'mode':'asynchronous_inference_continuous_servo','model':args.model,
          'action_selection':('observation_age_open_and_transport' if args.transport_age_aligned else
                              'observation_age_while_open_first_step_while_closed' if config['approach'] else 'first_step'),
          'transport_age_aligned':args.transport_age_aligned,
          'pending_suction_policy':'continue_inference_while_open' if config['approach'] else 'accept_fresh_inflight_before_io',
          'checkpoint_sha256':config['model_sha'],'requested_suction_cycles':expected_cycles,
          'camera_set_sha256':hashlib.sha256(CAMERA_SET.read_bytes()).hexdigest(),'control_hz':1/DT,
          'joint_speed_deg_s':MAX_SPEED,'joint_acceleration_deg_s2':MAX_ACCEL,'tcp_speed_m_s':MAX_TCP_SPEED,
          'max_prediction_delta_deg':None,'max_prediction_delta_m':None,
          'max_servo_tracking_error_deg':MAX_TRACKING_ERROR_DEG,
          'latency_tolerant':args.latency_tolerant,
          'max_prediction_age_s':config['max_prediction_age'],'watchdog_s':config['max_without_prediction'],
          'max_camera_skew_s':config['max_camera_skew'],
          'max_camera_age_ms':config['max_camera_age_ms'],
          'rpc_timeout_ms':RPC_TIMEOUT_MS,'inference_timeout_s':config['inference_timeout_s'],
          'max_control_gap_s':config['max_control_gap'],
          'max_state_age_s':config['max_state_age'],'stale_state_refreshes_per_step':1 if config['mixed'] else 0,
          'servo_command_time_s':config['servo_command_time'],
          'suction_writes':True,'max_seconds':max_seconds,'automatic_home':False,'automatic_release':False,
          'startup_release_if_unset':'operator_confirmed',
          'servo_disable_confirmation_timeout_s':1.0,'video_recording':{'enabled':True,'fps':25,'directory':'videos'}}
    if target_mode:
        plan.update(target_id=args.target_id,seed_snapshot_id=target_seed['snapshot_id'],
            checkpoint=target_options.get('checkpoint'),inference_script=target_options.get('inference_script'),
            target_selection_mode=target_seed.get('target_selection_mode', 'highest_score'),
            target_mask_source=('current instances with visible seeds; merged instances flagged; stop if target has no pixels'
                                if args.multi_target_separation else
                                'fresh foreground separated by both reviewed targets; stop if selected target has no pixels'
                                if args.paired_target_separation else 'fresh segmentation plus reviewed-seed association; stop on failure'),
            target_phase_source='first accepted close command; never prediction',
            segmentation_checkpoint=target_options['segmentation_checkpoint'])
    if sequence:
        plan.update(target_sequence=sequence,automatic_home=len(sequence)>1,
                    homing_between_targets_only=True,max_seconds_scope='per_target',
                    inference_session_per_target=True,
                    remaining_targets_by_pick=[sequence[i:] for i in range(len(sequence))])
    if args.model=='target-stack':
        plan.update(empty_target_reobservations_per_target=MAX_TARGET_REOBSERVATIONS,
            empty_target_policy='stop servo, clear actions, preserve gripper; fresh observations only')
    plan.update(target_preview=args.target_preview,track_target_motion=args.track_target_motion)
    return plan


def main(argv=None):
    args,sequence,target_mode,approach,mixed,inference_timeout_s,parser=parse_args(argv)
    target_session,target_seed,target_options,target_setups=resolve_targets(args,sequence,target_mode,parser)
    config=resolve_config(args,sequence,target_mode,approach,mixed,inference_timeout_s,target_options)
    plan=build_plan(args,sequence,target_mode,target_seed,target_options,config)
    if not args.execute:print(json.dumps(plan,indent=2));return 0
    if not sys.stdin.isatty():parser.error('interactive terminal required')
    return execute_run(args,sequence,target_mode,target_seed,target_options,target_setups,target_session,plan,config)


def execute_run(args,sequence,target_mode,target_seed,target_options,target_setups,target_session,plan,config):
    """Delegate the live session to the runtime execution module."""
    return execute_continuous(args=args,sequence=sequence,target_mode=target_mode,
        target_seed=target_seed,target_options=target_options,target_setups=target_setups,
        target_session=target_session,plan=plan,config=config)


if __name__=='__main__':
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    raise SystemExit(main())
