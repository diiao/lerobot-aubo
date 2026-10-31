import base64
import json
import cv2
import numpy as np
import pytest

from lerobot.bamboo_sorting.perception.joint_target_motion import TargetMotionReferences, _InsufficientMotionPoints
from lerobot.bamboo_sorting.runtime.joint_target_runtime import FreshTargetInputs, encode_mask, decode_mask
from lerobot.bamboo_sorting.runtime.joint_target_preview import perception_notice, render_packet


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


def weak_texture_with_glare():
    rng = np.random.default_rng(23)
    mask = np.zeros((480, 640), bool)
    mask[120:340, 280:305] = True
    rgb = np.full((480, 640, 3), 100, np.uint8)
    rgb[mask] = rng.integers(100, 116, (mask.sum(), 3), dtype=np.uint8)
    glare = rgb.copy()
    glare[220:230, 285:300] = 255
    return rgb, glare, mask


def test_carried_points_avoid_low_texture_redetection_dead_end():
    rgb, glare, mask = weak_texture_with_glare()
    tracker = TargetMotionReferences(rgb, {'A': mask})
    first = tracker.update(glare, ['A'], frame_id='glare')
    assert not first['failures']  # Old points still match, but new detection has only four.
    for offset in (6, 12):
        current = np.roll(rgb, offset, axis=0)
        with pytest.raises(ValueError, match='insufficient visible texture'):
            tracker._move(tracker._gray(glare), tracker._gray(current), mask)
        packet = tracker.update(current, ['A'], frame_id=f'moved{offset}')
        expected = np.roll(mask, offset, axis=0)
        actual = decode_mask(packet['masks_png']['A'])
        assert (actual & expected).sum() / (actual | expected).sum() > .96
        assert not packet['failures']
        if offset == 6:
            assert packet['diagnostics']['A'].get('recovered', False) is False
            assert packet['diagnostics']['A']['reference_frame_id'] == 'glare'
        else:
            assert packet['diagnostics']['A']['reference_frame_id'] == 'moved6'
    np.testing.assert_array_equal(tracker.original['A'], mask)


def test_moving_occluder_does_not_become_stationary_strip_motion():
    rgb, _, mask = weak_texture_with_glare()
    tracker = TargetMotionReferences(rgb, {'A': mask})
    for y in (220, 222, 224, 226, 228):
        current = rgb.copy()
        current[y:y+10, 285:300] = 255
        packet = tracker.update(current, ['A'], frame_id=f'occluder{y}')
        assert not packet['failures']
        assert packet['diagnostics']['A']['translation_px'] < .1
        np.testing.assert_array_equal(decode_mask(packet['masks_png']['A']), mask)


def test_backup_cannot_recover_from_full_occlusion_or_replace_it_with_neighbor():
    rgb, glare, mask = weak_texture_with_glare()
    tracker = TargetMotionReferences(rgb, {'A': mask})
    tracker.update(glare, ['A'], frame_id='glare')
    reference = tracker.references['A']
    backup = tracker.recovery_references['A']
    for current in (np.full_like(rgb, 100), np.roll(rgb, 100, axis=1)):
        packet = tracker.update(current, ['A'], frame_id='hidden', required=False)
        assert 'A' in packet['failures']
        assert tracker.references['A'] is reference
        assert tracker.recovery_references['A'] is backup
        np.testing.assert_array_equal(decode_mask(packet['masks_png']['A']), mask)


def test_point_tracks_follow_crossing_top_strip_without_taking_lower_fragments():
    rgb, _, top = weak_texture_with_glare()
    lower = np.zeros_like(top)
    lower[215:240, 190:400] = True
    lower &= ~top
    rng = np.random.default_rng(31)
    rgb[lower] = rng.integers(100, 116, (lower.sum(), 3), dtype=np.uint8)
    glare = rgb.copy()
    glare[150:160, 285:300] = 255
    tracker = TargetMotionReferences(rgb, {'A': top, 'B': lower})
    tracker.update(glare, ['A', 'B'], frame_id='glare')
    for offset in (2, 4, 6, 8, 10):
        expected = np.roll(top, offset, axis=0)
        current = rgb.copy()
        current[top] = 100
        current[expected] = np.roll(rgb, offset, axis=0)[expected]
        packet = tracker.update(current, ['A', 'B'], frame_id=f'moved{offset}')
        assert not packet['failures']
        actual = decode_mask(packet['masks_png']['A'])
        other = decode_mask(packet['masks_png']['B'])
        assert (actual & expected).sum() / (actual | expected).sum() > .96
        assert (other & lower).sum() / (other | lower).sum() > .96
        assert not np.any(actual & other)


@pytest.mark.parametrize('offset,accepted', [(26, True), (-20, False)])
def test_backup_matches_old_texture_and_respects_last_accepted_step(monkeypatch, offset, accepted):
    rgb, mask = textured()
    tracker = TargetMotionReferences(rgb, {'A': mask})
    tracker.update(np.roll(rgb, 20, axis=0), ['A'], frame_id='plus20')
    move = tracker._move
    last = tracker.references['A'][0]

    def lose_latest_points(previous, current, region, **kwargs):
        if previous is last:
            raise _InsufficientMotionPoints('insufficient visible texture')
        return move(previous, current, region, **kwargs)

    monkeypatch.setattr(tracker, '_move', lose_latest_points)
    # Both fit the initial image; +20 -> -20 must still fail the 35 px limit.
    packet = tracker.update(np.roll(rgb, offset, axis=0), ['A'], frame_id='retry', required=False)
    if accepted:
        assert not packet['failures']
        assert packet['diagnostics']['A']['recovered'] is True
        assert packet['diagnostics']['A']['reference_frame_id'] == 'initial_snapshot'
        expected = np.roll(mask, offset, axis=0)
        actual = decode_mask(packet['masks_png']['A'])
        assert (actual & expected).sum() / (actual | expected).sum() > .96
        json.dumps(packet)  # Internal point arrays must not enter the wire packet.
    else:
        assert 'recovery exceeds last accepted step' in packet['failures']['A']
        assert tracker.reference_frames['A'] == 'plus20'


def test_conflicting_recovery_does_not_commit_either_reference(monkeypatch):
    rgb, mask = textured()
    neighbor = np.roll(mask, 30, axis=1)
    tracker = TargetMotionReferences(rgb, {'A': mask, 'B': neighbor})
    tracker.recovery_references['A'] = (tracker._gray(rgb), mask.copy(), 'backup')
    tracker.recovery_points['A'] = None
    saved = dict(tracker.references)
    backup = tracker.recovery_references['A']
    calls = 0

    def move(previous, current, region, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _InsufficientMotionPoints('insufficient visible texture')
        return neighbor.copy(), {}

    monkeypatch.setattr(tracker, '_move', move)
    packet = tracker.update(rgb, ['A', 'B'], frame_id='conflict', required=False)
    assert packet['failures'] == dict(A='overlapping motion identities', B='overlapping motion identities')
    assert all(tracker.references[t] is saved[t] for t in saved)
    assert tracker.recovery_references['A'] is backup
    assert 'B' not in tracker.recovery_references


@pytest.mark.parametrize('reason', ['ambiguous motion fit', 'motion exceeds texture/rigid-step bounds'])
def test_recovery_does_not_retry_ambiguous_or_out_of_bounds_fit(monkeypatch, reason):
    rgb, mask = textured()
    tracker = TargetMotionReferences(rgb, {'A': mask})
    tracker.update(rgb, ['A'], frame_id='accepted')
    calls = 0

    def reject(previous, current, region, **kwargs):
        nonlocal calls
        calls += 1
        raise ValueError(reason)

    monkeypatch.setattr(tracker, '_move', reject)
    packet = tracker.update(rgb, ['A'], frame_id='bad_fit', required=False)
    assert calls == 1
    assert packet['failures']['A'] == reason
    assert tracker.reference_frames['A'] == 'accepted'


def session(mask,current):
    seed=dict(schema='joint_target_live_seed',target_id='B',snapshot_id='scene',human_reviewed=True,
        mask_png=encode_mask(mask),excluded_mask_png=encode_mask(np.zeros_like(mask),allow_empty=True),
        target_selection_mode='multi_foreground',visible_targets=[dict(target_id='B',mask_png=encode_mask(mask))],
        motion_tracking='rgb_optical_flow')
    return FreshTargetInputs(seed,lambda rgb:(current.astype(np.uint16),
        [dict(mask_id=1,class_id=0)] if current.any() else []))


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


@pytest.mark.parametrize('flow_failed',[False,True])
@pytest.mark.parametrize('visible_rows',[20,54,55,109,110])
def test_selected_partial_foreground_survives_old_support_thresholds(flow_failed,visible_rows):
    rgb,mask=textured();packet=dict(schema='joint_target_motion_reference',frame_id='new',
        source='rgb_optical_flow',human_reviewed=False,masks_png={'B':encode_mask(mask)},
        diagnostics={},failures={'B':'occlusion'} if flow_failed else {})
    meta=dict(frame_id='new',target_id='B',phase='approach')
    fragment=mask.copy();fragment[120+visible_rows:]=False
    _,e=session(mask,fragment).prepare(frame(rgb),meta,motion_reference=packet)
    np.testing.assert_array_equal(decode_mask(e['mask_png']),fragment)
    assert e['motion_reference']['support_policy']=='diagnostic_only'
    assert e['motion_reference']['current_foreground_support']['B']==pytest.approx(visible_rows/220)


def test_live_flow_failure_keeps_reference_but_never_fills_empty_foreground():
    from lerobot.bamboo_sorting.runtime.joint_target_runtime import TargetForegroundMissing
    rgb,mask=textured();tracker=TargetMotionReferences(rgb,{'B':mask})
    blank=np.zeros_like(rgb)+40
    packet=tracker.update(blank,['B'],frame_id='hidden',required=False)
    assert 'B' in packet['failures']
    np.testing.assert_array_equal(decode_mask(packet['masks_png']['B']),mask)
    with pytest.raises(TargetForegroundMissing):
        session(mask,np.zeros_like(mask)).prepare(frame(blank),
            dict(frame_id='hidden',target_id='B',phase='approach'),motion_reference=packet)
    fragment=mask.copy();fragment[140:]=False
    packet=tracker.update(rgb,['B'],frame_id='visible_again',required=False)
    _,e=session(mask,fragment).prepare(frame(rgb),
        dict(frame_id='visible_again',target_id='B',phase='approach'),motion_reference=packet)
    np.testing.assert_array_equal(decode_mask(e['mask_png']),fragment)


def test_live_search_tolerates_reference_drift_without_absorbing_distant_foreground():
    rgb,mask=textured();reference=np.roll(mask,12,axis=1)
    distant=np.roll(mask,-180,axis=1)
    packet=dict(schema='joint_target_motion_reference',frame_id='drift',source='rgb_optical_flow',
        human_reviewed=False,masks_png={'B':encode_mask(reference)},diagnostics={},
        failures={'B':'ambiguous motion fit'})
    _,e=session(mask,mask|distant).prepare(frame(rgb),
        dict(frame_id='drift',target_id='B',phase='approach'),motion_reference=packet)
    np.testing.assert_array_equal(decode_mask(e['mask_png']),mask)
    assert e['separation']['motion_radius_px']==16
    assert e['separation']['unassigned_foreground_pixels']==distant.sum()


@pytest.mark.parametrize('flow_failed',[False,True])
@pytest.mark.parametrize('visible_rows',[0,20])
def test_neighbor_and_selected_partial_masks_use_only_current_foreground(flow_failed,visible_rows):
    rgb,active=textured()
    neighbor=np.roll(active,100,axis=1)
    fragment=neighbor.copy();fragment[120+visible_rows:]=False
    instances=active.astype(np.uint16)+2*fragment.astype(np.uint16)
    segments=[dict(mask_id=1,class_id=0)]
    if fragment.any():segments.append(dict(mask_id=2,class_id=0))
    seeds={'A':active,'C':neighbor}
    packet=dict(schema='joint_target_motion_reference',frame_id='new',source='rgb_optical_flow',
        human_reviewed=False,masks_png={t:encode_mask(m) for t,m in seeds.items()},
        diagnostics={},failures={'C':'ambiguous motion fit'} if flow_failed else {})

    def prepare(target):
        seed=dict(schema='joint_target_live_seed',target_id=target,snapshot_id='scene',human_reviewed=True,
            mask_png=encode_mask(seeds[target]),excluded_mask_png=encode_mask(seeds['C' if target=='A' else 'A']),
            target_selection_mode='multi_foreground',
            visible_targets=[dict(target_id=t,mask_png=encode_mask(m)) for t,m in seeds.items()],
            motion_tracking='rgb_optical_flow')
        return FreshTargetInputs(seed,lambda rgb:(instances,segments)).prepare(frame(rgb),
            dict(frame_id='new',target_id=target,phase='approach'),motion_reference=packet)

    _,evidence=prepare('A')
    np.testing.assert_array_equal(decode_mask(evidence['mask_png']),active)
    motion=evidence['motion_reference']
    assert motion['support_policy']=='diagnostic_only'
    assert motion['insufficient_support_targets']==['C']
    assert motion['current_foreground_support']['C']==pytest.approx(visible_rows/220)
    # Selecting C uses only its visible fragment, never its full reference.
    if fragment.any():
        _,selected=prepare('C')
        np.testing.assert_array_equal(decode_mask(selected['mask_png']),fragment)
    else:
        from lerobot.bamboo_sorting.runtime.joint_target_runtime import TargetForegroundMissing
        with pytest.raises(TargetForegroundMissing,match='no current foreground: C'):prepare('C')


@pytest.mark.parametrize('outline_width_px', [1,2])
def test_preview_same_frame_does_not_mutate_model_inputs_or_retain_transport_box(monkeypatch,outline_width_px):
    rgb,mask=textured();images={'global_rgb':rgb,'grasp_rgb':np.zeros_like(rgb)}
    evidence=dict(phase='approach',target_id='B',mask_png=encode_mask(mask),target_outline_width_px=outline_width_px)
    encoded=[];imencode=cv2.imencode
    def capture(extension,image,*args):
        encoded.append(image.copy())
        return imencode(extension,image,*args)
    monkeypatch.setattr(cv2,'imencode',capture)
    before=rgb.copy();packet=render_packet(images,evidence,index=7,sensor_times=[10,11,12])
    assert packet['pixels']==mask.sum() and packet['source_time']==10
    assert np.array_equal(rgb,before)
    # One pixel inside the display box is unmarked for 1px, magenta for 2px.
    expected=before[200,281] if outline_width_px==1 else np.array([255,0,255],np.uint8)
    np.testing.assert_array_equal(encoded[0][200,281,::-1],expected)
    transport=render_packet(images,dict(phase='transport',target_id='B',mask_png=None),index=8,sensor_times=[12,13,14])
    assert transport['pixels']==0
    for p in [packet,transport]:
        image=cv2.imdecode(np.frombuffer(base64.b64decode(p['jpeg']),np.uint8),1)
        assert image.shape==(480,1280,3)


def test_preview_exposes_selected_target_perception_degradation_and_clears_on_transport():
    evidence=dict(phase='approach',target_id='A',motion_reference=dict(
        failures={'A':'insufficient consistent motion points'},
        insufficient_support_targets=['A'],current_foreground_support={'A':.38753}),
        separation=dict(merged_instances=[dict(targets=['A','B'])]))
    notice,attention=perception_notice(evidence)
    assert attention
    assert '跟踪失败' in notice and '沿用旧参考' in notice
    assert '前景支持不足' in notice and '拆分' in notice and '38.8%' in notice
    assert '非识别准确率' in notice
    assert perception_notice({**evidence,'phase':'transport'})==('',False)


def test_preview_does_not_attribute_other_targets_degradation_to_selected_target():
    notice,attention=perception_notice(dict(phase='approach',target_id='A',
        motion_reference=dict(failures={'C':'no points'},insufficient_support_targets=['C'],
                              current_foreground_support={'A':.9}),
        separation=dict(merged_instances=[dict(targets=['B','C'])])))
    assert not attention
    assert '跟踪失败' not in notice and '前景支持不足' not in notice and '拆分' not in notice
    assert '尚未逐像素确认' in notice and '90.0%' in notice
