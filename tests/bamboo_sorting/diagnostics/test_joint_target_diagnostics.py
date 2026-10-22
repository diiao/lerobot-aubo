import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.bamboo_sorting.paths import REPO_ROOT


spec = importlib.util.spec_from_file_location(
    'target_diagnostics', REPO_ROOT / 'examples/phone_to_auboi10/diagnose_joint_target_policy.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_teacher_forced_next_frame_is_not_autonomous_recovery():
    states = np.zeros((6, 7), dtype=np.float32)
    states[3:, 6] = 100
    targets = states.copy()
    targets[2, 6] = 100
    data = dict(states=states, targets=targets, predictions=states.copy(),
                source_index=np.zeros(6, dtype=int), episode_index=np.zeros(6, dtype=int),
                frame_index=np.arange(6))
    event = diagnostics.event_windows(data)['events'][0]
    assert not event['event_frame_hit']
    assert event['nearest_prediction_transition_offset_frames'] == 1
    assert event['input_changes_next_frame']
    assert event['prediction_matches_new_input_next_frame']
    assert event['prechange_target_prediction_offsets'] == []
    data['frame_index'][-1] = 7
    with pytest.raises(ValueError, match='contiguous'):
        diagnostics.event_windows(data)


def test_alternative_region_never_fills_background_or_borrows_other_frame():
    instances = np.zeros((480, 640), dtype=np.uint16)
    instances[50:150, 98:103] = 1
    instances[50:150, 198:203] = 1  # Same predicted ID may span both objects.
    region = dict(strip_center_segments_xy=[[[100,50],[100,150]], [[200,50],[200,150]]],
                  target_segment_index=0, max_distance_px=18)
    first = diagnostics.region_mask(instances, region, False)
    other = diagnostics.region_mask(instances, region, True)
    assert first.any() and other.any() and not (first & other).any()
    assert np.all((first | other) <= (instances != 0))
    assert not diagnostics.region_mask(np.zeros_like(instances), region, True).any()


def test_outline_preserves_raw_image_and_uses_two_pixel_inner_boundary():
    raw = np.full((480,640,3), 17, dtype=np.uint8)
    mask = np.zeros((480,640), dtype=bool)
    mask[100:120,100:120] = True
    marked = diagnostics.outline(raw, mask)
    assert np.all(raw == 17)
    assert np.array_equal(marked[~mask], raw[~mask])
    assert np.all(marked[102:118,102:118] == 17)
    assert np.array_equal(marked[100,100], [255,0,255])


def test_summary_separates_seed_noise_and_state_intervention(tmp_path):
    inputs=tmp_path/'inputs'; inputs.mkdir()
    inference=tmp_path/'inference'; inference.mkdir()
    samples=[dict(id='swap',kind='target_swap',scene_id='s',frame_index=0,
                  variants=['original','alternate','unmarked','repeat']),
             dict(id='event',kind='gripper_event',event=dict(event='close',event_frame=5,offset=0),
                  variants=['original','state_intervention'])]
    (inputs/'inputs.json').write_text(json.dumps(dict(samples=samples,seeds=[42,43,44],limits=[])))
    np.savez(inputs/'inputs.npz',swap_state=np.zeros(7),event_state=np.zeros(7))
    chunks=[]; rows=[]
    for sample in samples:
        for seed in (42,43,44):
            for variant in sample['variants']:
                chunk=np.zeros((50,7),dtype=np.float32)
                chunk[:,:6]=seed-42+(3 if variant=='alternate' else 0)
                if variant=='state_intervention': chunk[:,6]=100
                rows.append(dict(sample_id=sample['id'],seed=seed,variant=variant,array_index=len(chunks)))
                chunks.append(chunk)
    (inference/'predictions.json').write_text(json.dumps(dict(results=rows,completed=True,optimizer_steps=0,checkpoint='test')))
    np.savez(inference/'chunks.npz',chunks=chunks)
    diagnostics.summarize(SimpleNamespace(inputs=inputs,inference=inference,output=tmp_path/'summary'))
    report=json.loads((tmp_path/'summary/report.json').read_text())
    assert report['target_swap_summary']['repeat_max_abs_difference']==0
    assert report['target_swap_summary']['mean_joint_change_deg']==3
    assert report['target_swap_summary']['observations_swap_exceeds_seed_noise']==1
    assert all(g['first_action_matches_supplied_state']==3 for g in report['gripper_summary'])
