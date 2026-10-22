import base64
import cv2
import numpy as np
import pytest

from lerobot.bamboo_sorting.perception.joint_target_motion import TargetMotionReferences
from lerobot.bamboo_sorting.runtime.joint_target_runtime import FreshTargetInputs, encode_mask, decode_mask
from lerobot.bamboo_sorting.runtime.joint_target_preview import render_packet


def textured():
    rng=np.random.default_rng(23)
    rgb=np.zeros((480,640,3),np.uint8)+40
    mask=np.zeros((480,640),bool);mask[120:340,280:305]=True
    rgb[mask]=rng.integers(60,240,(mask.sum(),3),dtype=np.uint8)
    return rgb,mask


@pytest.mark.parametrize('angle,dx,dy',[(0,20,9),(8,14,0),(-8,-16,4)])
def test_motion_exceeds_old_seed_radius_but_preserves_identity(angle,dx,dy):
    rgb,mask=textured();t=TargetMotionReferences(rgb,{'B':mask})
    transform=cv2.getRotationMatrix2D((292,230),angle,1);transform[:,2]+=[dx,dy]
    new=cv2.warpAffine(rgb,transform,(640,480),borderValue=(40,40,40))
    expected=cv2.warpAffine(mask.astype(np.uint8),transform,(640,480),flags=cv2.INTER_NEAREST)>0
    packet=t.update(new,['B'],frame_id='new');actual=decode_mask(packet['masks_png']['B'])
    assert (actual&expected).sum()/(actual|expected).sum()>.93
    assert np.array_equal(t.original['B'],mask)
    assert packet['human_reviewed'] is False


def test_occluded_texture_does_not_move_reference_or_pick_neighbor():
    rgb,mask=textured();t=TargetMotionReferences(rgb,{'B':mask})
    new=np.zeros_like(rgb)+40
    packet=t.update(new,['B'],frame_id='lost',required=False)
    assert 'B' in packet['failures']
    assert np.array_equal(decode_mask(packet['masks_png']['B']),mask)
    with pytest.raises(ValueError,match='tracking lost'):t.update(new,['B'],frame_id='lost2')


def session(mask,current):
    seed=dict(schema='joint_target_live_seed',target_id='B',snapshot_id='scene',human_reviewed=True,
        mask_png=encode_mask(mask),excluded_mask_png=encode_mask(np.zeros_like(mask),allow_empty=True),
        target_selection_mode='multi_foreground',visible_targets=[dict(target_id='B',mask_png=encode_mask(mask))],
        motion_tracking='rgb_optical_flow')
    return FreshTargetInputs(seed,lambda rgb:(current.astype(np.uint16),[dict(mask_id=1,class_id=0)]))


def frame(rgb):
    return {'observation.state':np.zeros(7),'task':'test',
        'observation.images.global_rgb':rgb,'observation.images.grasp_rgb':rgb.copy()}


def test_moved_reference_only_selects_current_foreground_and_is_bound_to_frame():
    rgb,mask=textured();moved=np.roll(mask,20,axis=1)
    packet=dict(schema='joint_target_motion_reference',frame_id='new',source='rgb_optical_flow',
        human_reviewed=False,masks_png={'B':encode_mask(moved)},diagnostics={},failures={})
    meta=dict(frame_id='new',target_id='B',phase='approach')
    s=session(mask,moved);_,e=s.prepare(frame(rgb),meta,motion_reference=packet)
    assert np.array_equal(decode_mask(e['mask_png']),moved)
    assert e['motion_reference']['current_foreground_support']['B']==1
    assert np.array_equal(decode_mask(s.seed['mask_png']),mask)
    with pytest.raises(ValueError,match='same-frame'):
        session(mask,moved).prepare(frame(rgb),meta,motion_reference={**packet,'frame_id':'old'})


def test_flow_failure_needs_current_support_and_tiny_fragment_is_rejected():
    rgb,mask=textured();packet=dict(schema='joint_target_motion_reference',frame_id='new',
        source='rgb_optical_flow',human_reviewed=False,masks_png={'B':encode_mask(mask)},
        diagnostics={},failures={'B':'occlusion'})
    meta=dict(frame_id='new',target_id='B',phase='approach')
    _,e=session(mask,mask).prepare(frame(rgb),meta,motion_reference=packet)
    assert e['motion_reference']['current_foreground_support']['B']==1
    fragment=mask.copy();fragment[140:]=False
    with pytest.raises(ValueError,match='lacks current foreground support'):
        session(mask,fragment).prepare(frame(rgb),meta,motion_reference=packet)


def test_preview_same_frame_does_not_mutate_model_inputs_or_retain_transport_box():
    rgb,mask=textured();images={'global_rgb':rgb,'grasp_rgb':np.zeros_like(rgb)}
    evidence=dict(phase='approach',target_id='B',mask_png=encode_mask(mask))
    before=rgb.copy();packet=render_packet(images,evidence,index=7,sensor_times=[10,11,12])
    assert packet['pixels']==mask.sum() and packet['source_time']==10
    assert np.array_equal(rgb,before)
    transport=render_packet(images,dict(phase='transport',target_id='B',mask_png=None),index=8,sensor_times=[12,13,14])
    assert transport['pixels']==0
    for p in [packet,transport]:
        image=cv2.imdecode(np.frombuffer(base64.b64decode(p['jpeg']),np.uint8),1)
        assert image.shape==(480,1280,3)
