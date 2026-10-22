#!/usr/bin/env python
"""Offline diagnostics from saved expert observations; never a robot controller.

Events consume existing predictions. Prepare decodes only selected validation
frames. Infer loads the frozen action checkpoint with no optimizer. Alternative
masks are diagnostic drafts, never additions to accepted training annotations.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def event_windows(data, radius=25):
    """Never count prediction changes after observed IO changes as autonomous recovery."""
    events = []
    identities = sorted(set(zip(data['source_index'].tolist(), data['episode_index'].tolist())))
    for source, episode in identities:
        ids = np.flatnonzero((data['source_index'] == source) & (data['episode_index'] == episode))
        ids = ids[np.argsort(data['frame_index'][ids])]
        frames = data['frame_index'][ids]
        if not np.array_equal(frames, np.arange(len(ids))):
            raise ValueError('complete contiguous episode frames required')
        state, target, pred = (data[k][ids, 6] for k in ('states', 'targets', 'predictions'))
        current, truth, predicted = state >= 50, target >= 50, pred >= 50
        for j in np.flatnonzero(current != truth):
            direction = bool(truth[j])
            window = np.arange(max(0, j-radius), min(len(ids), j+radius+1))
            hits = window[predicted[window] == direction]
            # Predictions from later teacher-forced frames see the expert's new
            # gripper state; they cannot establish closed-loop autonomous timing.
            prechange_hits = hits[current[hits] == current[j]]
            changes = window[(predicted[window] == direction) &
                             (predicted[np.maximum(window-1, 0)] != direction) & (window > 0)]
            nearest = int(changes[np.argmin(abs(changes-j))]-j) if len(changes) else None
            events.append(dict(source_index=int(source), episode_index=int(episode),
                frame_index=int(j), event='close' if direction else 'release',
                event_frame_hit=bool(predicted[j] == direction),
                nearest_prediction_transition_offset_frames=nearest,
                prechange_target_prediction_offsets=(prechange_hits-j).tolist(),
                input_changes_next_frame=bool(j+1 < len(ids) and current[j+1] == direction),
                prediction_matches_new_input_next_frame=bool(j+1 < len(ids) and predicted[j+1] == current[j+1]),
                window=[dict(offset=int(k-j), frame_index=int(k), state=float(state[k]),
                             target=float(target[k]), prediction=float(pred[k])) for k in window]))
    return dict(schema='joint_target_event_windows', nominal_fps=25,
        timing_scope='frame offsets only; not measured wall-clock delays',
        closed_loop_success_verified=False, events=events,
        summary={kind:dict(events=sum(e['event']==kind for e in events),
            event_frame_hits=sum(e['event']==kind and e['event_frame_hit'] for e in events),
            next_frame_input_changed_and_prediction_matches=sum(e['event']==kind and e['input_changes_next_frame']
                and e['prediction_matches_new_input_next_frame'] for e in events)) for kind in ('close','release')})


def region_mask(instance_map, region, alternative):
    """Intersect same-frame foreground with the existing geometric selection rule."""
    y, x = np.mgrid[:480, :640]
    pixels = np.stack((x, y), axis=-1)
    distances = []
    for segment in region['strip_center_segments_xy']:
        a, b = np.asarray(segment, dtype=float)
        v = b-a
        if np.dot(v,v) <= 0:
            raise ValueError('degenerate region line')
        t = np.clip(np.sum((pixels-a)*v, axis=-1)/np.dot(v,v), 0, 1)
        distances.append(np.linalg.norm(pixels-(a+t[...,None]*v), axis=-1))
    if len(distances) != 2:
        raise ValueError('diagnostic requires exactly two explicit strip regions')
    target = region['target_segment_index']
    if alternative:
        target = 1-target
    return (instance_map != 0) & (distances[target] <= region['max_distance_px']) & (distances[target] < distances[1-target])


def outline(rgb, mask):
    import cv2
    if mask.dtype != np.bool_ or mask.shape != (480,640) or not mask.any():
        raise ValueError('nonempty same-frame diagnostic mask required')
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5,5),np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    result = rgb.copy()
    result[mask & (eroded == 0)] = (255,0,255)
    return result


def prepare(args):
    import cv2
    from PIL import Image, ImageDraw
    from lerobot.bamboo_sorting.policy.joint_training import load_prepared_dataset
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = np.load(args.predictions, allow_pickle=False)
    events = event_windows(predictions)
    write(args.output/'event_windows.json', events)
    preflight = read(args.preflight)
    records = preflight['validation'].get('sources', [preflight['validation']])
    samples, arrays = [], {}
    for si, record in enumerate(records):
        dataset = load_prepared_dataset(record)
        original = dataset.dataset
        annotation_path = Path(record['target_annotations'])
        annotations = read(annotation_path)
        regions = {r['episode_index']:r for r in read(annotation_path.parent/'target_region_selection.json')['regions']}
        source_name = Path(record['root']).name.removeprefix('joint_target_pair_')
        pred_path = annotation_path.parent/'predictions'/f'{source_name}_predictions.json'
        segmentation = {(r['episode_index'],r['frame_index']):r for r in read(pred_path)['samples']}
        positions = {(f['episode_index'],f['frame_index']):i for i,f in enumerate(record['target_frames'])}
        for ep in annotations['episodes']:
            ei = ep['episode_index']
            requests = [(k,'target_swap',None) for k in (0,100)]
            for event in events['events']:
                if event['source_index']==si and event['episode_index']==ei:
                    requests.extend((event['frame_index']+offset, 'gripper_event',
                                     dict(event=event['event'],event_frame=event['frame_index'],offset=offset))
                                    for offset in (-10,0,1))
            for fi,kind,event in requests:
                pos = positions[(ei,fi)]
                raw = original[pos]
                sample = dataset[pos]
                rgb = raw['observation.images.global_rgb'].mul(255).round().byte().permute(1,2,0).numpy()
                wrist = raw['observation.images.grasp_rgb'].mul(255).round().byte().permute(1,2,0).numpy()
                marked = sample['observation.images.target_global_rgb'].mul(255).round().byte().permute(1,2,0).numpy()
                key = f's{si}_e{ei}_f{fi}_{kind}'
                variants = ['original']
                arrays[key+'_original'] = marked
                arrays[key+'_wrist'] = wrist
                arrays[key+'_state'] = raw['observation.state'].numpy()
                arrays[key+'_expert_actions'] = raw['action'].numpy()
                arrays[key+'_action_is_pad'] = raw['action_is_pad'].numpy()
                item = dict(id=key,kind=kind,source_index=si,episode_index=ei,frame_index=fi,
                            scene_id=ep['scene_id'],target_id=ep['target_id'],variants=variants,event=event,
                            transport=record['target_frames'][pos]['transport'])
                if kind=='target_swap':
                    seg = segmentation[(ei,fi)]
                    instance = cv2.imread(str(pred_path.parent/seg['instance_map']),cv2.IMREAD_UNCHANGED)
                    alternate = region_mask(instance, regions[ei], True)
                    arrays[key+'_alternate'] = outline(rgb, alternate)
                    arrays[key+'_unmarked'] = rgb
                    variants.extend(['alternate','unmarked','repeat'])
                    item.update(alternative_mask_pixels=int(alternate.sum()),alternative_mask_human_reviewed=False,
                                alternative_mask_source='same-frame frozen foreground intersect existing other-strip region')
                    panel = Image.new('RGB',(960,260),'white');draw=ImageDraw.Draw(panel)
                    for col,(label,img) in enumerate([('original',marked),('alternate draft',arrays[key+'_alternate']),('raw',rgb)]):
                        panel.paste(Image.fromarray(img).resize((320,240)),(col*320,20));draw.text((col*320+5,3),label,fill='black')
                    panel.save(args.output/f'{key}.jpg')
                elif event['offset'] in (0,1):
                    # At the event flip old -> new; after it clamp new -> old.
                    variants.append('state_intervention')
                samples.append(item)
    np.savez_compressed(args.output/'inputs.npz', **arrays)
    manifest=dict(schema='joint_target_frozen_diagnostic_inputs',samples=samples,seeds=[42,43,44],
        source_preflight=str(args.preflight.resolve()),source_predictions=str(args.predictions.resolve()),
        action_horizon=50,optimizer_steps=0,hardware_access=False,
        alternative_masks_human_reviewed=False,source_data_modified=False,
        intervention_scope='same decoded RGB/wrist/joints and paired seed; replace only target outline or gripper state',
        limits=['Expert-state observations are not closed-loop rollouts.',
                'Alternative regions are assistant-inspected diagnostic drafts, not new accepted annotations.',
                'Action differences establish input sensitivity, not correct target selection or physical success.'])
    write(args.output/'inputs.json',manifest)
    print(json.dumps({'samples':len(samples),'requests':sum(len(s['variants'])*3 for s in samples),'output':str(args.output)}))


def infer(args):
    import subprocess
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:
        raise RuntimeError(f'GPU occupied: {busy}')
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition,transition_to_batch,policy_action_to_transition,transition_to_policy_action
    from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK
    if read(args.checkpoint/'aubo_joint_contract.json')!=target_policy_contract_record():
        raise ValueError('target checkpoint contract mismatch')
    spec=read(args.inputs/'inputs.json'); inputs=np.load(args.inputs/'inputs.npz',allow_pickle=False)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg=PreTrainedConfig.from_pretrained(args.checkpoint,local_files_only=True);cfg.device='cuda'
    model=SmolVLAPolicy.from_pretrained(args.checkpoint,config=cfg,local_files_only=True,strict=True)
    model.eval();model.requires_grad_(False)
    pre=PolicyProcessorPipeline.from_pretrained(args.checkpoint,'policy_preprocessor.json',
        overrides={'device_processor':{'device':'cuda'}},to_transition=batch_to_transition,to_output=transition_to_batch)
    post=PolicyProcessorPipeline.from_pretrained(args.checkpoint,'policy_postprocessor.json',
        to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
    results=[];chunks=[]
    with torch.inference_mode():
        for sample in spec['samples']:
            key=sample['id']
            for seed in spec['seeds']:
                for variant in sample['variants']:
                    state=inputs[key+'_state'].copy()
                    if variant=='state_intervention':
                        state[6]=100-state[6]
                    image_variant=variant if variant in ('alternate','unmarked') else 'original'
                    frame={'observation.state':torch.from_numpy(state).unsqueeze(0),'task':[JOINT_TASK]}
                    for image_name,array_key in [('target_global_rgb',key+'_'+image_variant),('grasp_rgb',key+'_wrist')]:
                        frame['observation.images.'+image_name]=torch.from_numpy(inputs[array_key].copy()).permute(2,0,1).float().div(255).unsqueeze(0)
                    model.reset();torch.manual_seed(seed)
                    chunk=post(model.predict_action_chunk(pre(frame))).detach().cpu().numpy()[0]
                    if chunk.shape!=(50,7) or not np.isfinite(chunk).all():
                        raise ValueError('invalid full action chunk')
                    results.append(dict(sample_id=key,seed=seed,variant=variant,array_index=len(chunks)))
                    chunks.append(chunk)
            print('finished',key,flush=True)
    np.savez_compressed(args.output/'chunks.npz',chunks=np.stack(chunks))
    write(args.output/'predictions.json',dict(checkpoint=str(args.checkpoint),results=results,optimizer_steps=0,
        strict_load=True,completed=True,physical_success_verified=False))


def summarize(args):
    spec=read(args.inputs/'inputs.json')
    info=read(args.inference/'predictions.json')
    if not info['completed'] or info['optimizer_steps'] != 0:
        raise ValueError('completed frozen inference required')
    arrays=np.load(args.inference/'chunks.npz',allow_pickle=False)['chunks']
    inputs=np.load(args.inputs/'inputs.npz',allow_pickle=False)
    if not np.isfinite(arrays).all():
        raise ValueError('nonfinite predictions')
    lookup={(r['sample_id'],r['seed'],r['variant']):arrays[r['array_index']] for r in info['results']}
    expected={(s['id'],seed,v) for s in spec['samples'] for seed in spec['seeds'] for v in s['variants']}
    if len(lookup)!=len(info['results']) or set(lookup)!=expected:
        raise ValueError('comparison coverage mismatch')
    swaps=[];events=[]
    def mean_joint_change(a,b):
        return float(np.abs(a[:,:6]-b[:,:6]).mean())
    for sample in spec['samples']:
        key=sample['id']
        current=inputs[key+'_state'][6]>=50
        if sample['kind']=='target_swap':
            noise=[mean_joint_change(lookup[key,a,'original'],lookup[key,b,'original'])
                   for i,a in enumerate(spec['seeds']) for b in spec['seeds'][i+1:]]
            paired=[mean_joint_change(lookup[key,seed,'original'],lookup[key,seed,'alternate']) for seed in spec['seeds']]
            row=dict(sample_id=key,scene_id=sample['scene_id'],frame_index=sample['frame_index'],
                target_swap_mean_absolute_joint_change_deg=float(np.mean(paired)),
                different_seed_same_input_joint_change_deg=float(np.mean(noise)),
                paired_seed_joint_changes_deg=paired,
                same_seed_repeat_max_abs_difference=float(max(np.abs(lookup[key,seed,'original']-lookup[key,seed,'repeat']).max() for seed in spec['seeds'])),
                no_outline_mean_absolute_joint_change_deg=float(np.mean([mean_joint_change(lookup[key,seed,'original'],lookup[key,seed,'unmarked']) for seed in spec['seeds']])),
                first_action_target_swap_max_abs_joint_change_deg=float(max(np.abs(lookup[key,seed,'original'][0,:6]-lookup[key,seed,'alternate'][0,:6]).max() for seed in spec['seeds'])))
            swaps.append(row)
        else:
            for seed in spec['seeds']:
                for variant in sample['variants']:
                    chunk=lookup[key,seed,variant]
                    desired=sample['event']['event']=='close'
                    switches=np.flatnonzero((chunk[:,6]>=50)!=current)
                    events.append(dict(sample_id=key,seed=seed,variant=variant,**sample['event'],
                        original_observed_gripper=int(current)*100,
                        intervention_gripper=int(not current)*100 if variant=='state_intervention' else None,
                        first_action_gripper=float(chunk[0,6]),
                        first_action_matches_event=bool((chunk[0,6]>=50)==desired),
                        first_action_matches_supplied_state=bool((chunk[0,6]>=50)==(not current if variant=='state_intervention' else current)),
                        first_horizon_index_opposite_original_state=int(switches[0]) if len(switches) else None))
    grouped=[]
    for kind,offset,variant in sorted({(r['event'],r['offset'],r['variant']) for r in events}):
        selected=[r for r in events if (r['event'],r['offset'],r['variant'])==(kind,offset,variant)]
        indices=[r['first_horizon_index_opposite_original_state'] for r in selected if r['first_horizon_index_opposite_original_state'] is not None]
        grouped.append(dict(event=kind,offset=offset,variant=variant,predictions=len(selected),
            first_action_matches_event=sum(r['first_action_matches_event'] for r in selected),
            first_action_matches_supplied_state=sum(r['first_action_matches_supplied_state'] for r in selected),
            chunks_with_opposite_original_state=len(indices),
            median_first_opposite_index=float(np.median(indices)) if indices else None))
    report=dict(schema='joint_target_frozen_diagnostics',checkpoint=info['checkpoint'],
        samples=len(spec['samples']),predictions=len(lookup),seeds=spec['seeds'],optimizer_steps=0,
        target_swap_summary=dict(observations=len(swaps),
            mean_joint_change_deg=float(np.mean([r['target_swap_mean_absolute_joint_change_deg'] for r in swaps])),
            mean_different_seed_change_deg=float(np.mean([r['different_seed_same_input_joint_change_deg'] for r in swaps])),
            repeat_max_abs_difference=max(r['same_seed_repeat_max_abs_difference'] for r in swaps),
            observations_swap_exceeds_seed_noise=sum(r['target_swap_mean_absolute_joint_change_deg']>r['different_seed_same_input_joint_change_deg'] for r in swaps)),
        target_swaps=swaps,gripper_summary=grouped,gripper_predictions=events,
        target_direction_correctness_verified=False,physical_success_verified=False,
        limits=spec['limits']+['State-only interventions retain the saved expert image phase; they are sensitivity tests, not simulated closed-loop trajectories.'])
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('samples','predictions','target_swap_summary','gripper_summary')},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('events','prepare','infer','summarize'),required=True)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--inference',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('a new output directory is required')
    if args.stage=='events':
        if not args.predictions:parser.error('--predictions required')
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'event_windows.json',event_windows(np.load(args.predictions,allow_pickle=False)))
    elif args.stage=='prepare':
        if not args.predictions or not args.preflight:parser.error('--predictions and --preflight required')
        prepare(args)
    elif args.stage=='infer':
        if not args.inputs or not args.checkpoint:parser.error('--inputs and --checkpoint required')
        infer(args)
    else:
        if not args.inputs or not args.inference:parser.error('--inputs and --inference required')
        summarize(args)


if __name__=='__main__':
    main()
