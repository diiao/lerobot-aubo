import numpy as np
import pytest

from lerobot.bamboo_sorting.joint_target_runtime import FreshTargetInputs, encode_mask, decode_mask
from lerobot.bamboo_sorting.joint_target import prepare_target_images


def inputs():
    mask = np.zeros((480,640),bool);mask[200:300,400:420] = True
    other = np.zeros_like(mask);other[200:300,500:520] = True
    seed = dict(schema='joint_target_live_seed',target_id='A',snapshot_id='snapshot',human_reviewed=True,
                mask_png=encode_mask(mask),excluded_mask_png=encode_mask(other))
    frame = {'observation.state':np.zeros(7),'task':'test',
             **{'observation.images.'+key:np.zeros((480,640,3),np.uint8) for key in ('global_rgb','grasp_rgb')}}
    return mask,seed,frame


def test_fresh_mask_render_matches_training_without_forging_review():
    mask,seed,frame=inputs()
    fresh=np.roll(mask,2,axis=1)
    session=FreshTargetInputs(seed,lambda rgb:(fresh.astype(np.uint16),[dict(mask_id=1,class_id=1)]))
    value,evidence=session.prepare(frame,dict(frame_id='new',target_id='A',phase='approach'))
    expected=prepare_target_images({key:frame['observation.images.'+key] for key in ('global_rgb','grasp_rgb')},
        fresh,dict(schema='aubo_joint_target_frame',frame_id='new',target_id='A',status='visible',reviewed=True),
        frame_id='new',target_id='A')
    assert np.array_equal(value['observation.images.target_global_rgb'],expected['images']['target_global_rgb'])
    assert evidence['human_reviewed'] is False and evidence['association']['reviewed'] is False
    assert evidence['source']=='same_frame_segmentation'


def test_lost_target_never_reuses_seed_or_recovers_implicitly():
    mask,seed,frame=inputs();maps=[np.zeros((480,640),np.uint16),mask.astype(np.uint16)]
    session=FreshTargetInputs(seed,lambda rgb:(maps.pop(0),[dict(mask_id=1,class_id=0)]))
    for index in range(2):
        with pytest.raises(ValueError,match='association failed'):
            session.prepare(frame,dict(frame_id=str(index),target_id='A',phase='approach'))


@pytest.mark.parametrize('width',[2,5])
def test_partial_visibility_and_full_area_restoration_use_current_pixels(width):
    from lerobot.bamboo_sorting.joint_target_association import TargetAssociation
    mask,seed,frame=inputs()
    partial=np.zeros_like(mask);partial[225:275,409:409+width]=True
    # 5% and 12.5% of the initial area, followed by 20x/8x recovery.
    maps=[mask,partial,mask]
    pending=list(maps)
    session=FreshTargetInputs(seed,lambda rgb:(pending.pop(0).astype(np.uint16),[dict(mask_id=1,class_id=0)]))
    for index,expected in enumerate(maps):
        _,evidence=session.prepare(frame,dict(frame_id=str(index),target_id='A',phase='approach'))
        assert evidence['association']['status']=='matched'
        assert np.array_equal(decode_mask(evidence['mask_png']),expected)
        assert evidence['association_geometry_limits_enforced'] is False
        assert 'min_seed_area_ratio' not in evidence['association_parameters']
    assert TargetAssociation.PARAMETERS['min_area_ratio']==.25
    assert TargetAssociation.PARAMETERS['max_area_ratio']==4.
    assert TargetAssociation.PARAMETERS['min_seed_area_ratio']==.2


@pytest.mark.parametrize('case',['low_overlap','low_score','ambiguous'])
def test_live_selects_best_candidate_without_overlap_score_or_ambiguity_rejection(case):
    from lerobot.bamboo_sorting.joint_target_association import TargetAssociation
    mask,seed,frame=inputs();instances=np.zeros((480,640),np.uint16)
    if case=='low_overlap':instances[230:270,409:411]=1  # 4% IoU, below 0.05.
    elif case=='low_score':instances[225:275,444:446]=1  # Aligned IoU=0.05, low location score.
    else:
        instances[200:300,400:409]=1
        instances[200:300,411:420]=2
    segments=[dict(mask_id=1,class_id=0)]
    if case=='ambiguous':segments.append(dict(mask_id=2,class_id=0))
    offline=TargetAssociation('A',alignment='rigid',enforce_geometry_limits=False)
    offline.select(0,mask.astype(np.uint16),[segments[0]],1)
    assert offline.step(1,instances,segments)[0] is None
    session=FreshTargetInputs(seed,lambda rgb:(instances,segments))
    _,evidence=session.prepare(frame,dict(frame_id='1',target_id='A',phase='approach'))
    assert evidence['association_match_thresholds_enforced'] is False
    assert not {'min_overlap','min_score','ambiguity_margin'} & evidence['association_parameters'].keys()
    association=evidence['association']
    best=max(association['candidates'],key=lambda candidate:candidate['score'])
    assert association['mask_id']==best['mask_id']
    assert np.array_equal(decode_mask(evidence['mask_png']),instances==best['mask_id'])


@pytest.mark.parametrize('case',['distant','rotated','large_area'])
def test_live_drops_geometry_gates_while_offline_keeps_them(case):
    from lerobot.bamboo_sorting.joint_target_association import TargetAssociation
    mask,seed,frame=inputs();instances=np.zeros((480,640),np.uint16)
    if case=='distant':instances[200:300,500:520]=1
    elif case=='rotated':instances[240:260,360:460]=1
    else:instances[100:400,390:430]=1  # Six times the seed area.
    segments=[dict(mask_id=1,class_id=0)]
    offline=TargetAssociation('A',alignment='rigid')
    offline.select(0,mask.astype(np.uint16),segments,1)
    assert offline.step(1,instances,segments)[0] is None
    session=FreshTargetInputs(seed,lambda rgb:(instances,segments))
    _,evidence=session.prepare(frame,dict(frame_id='1',target_id='A',phase='approach'))
    assert evidence['association']['status']=='matched'
    assert np.array_equal(decode_mask(evidence['mask_png']),instances==1)


@pytest.mark.parametrize('overlap',[4,85,437])
def test_other_reviewed_overlap_is_recorded_without_stopping_or_cropping(overlap):
    mask,seed,frame=inputs()
    merged=mask.copy()
    # Keep the synthetic long-axis change within the unchanged association gate.
    other=np.roll(decode_mask(seed['excluded_mask_png']),-60,axis=1)
    seed['excluded_mask_png']=encode_mask(other)
    y,x=np.nonzero(other);merged[y[:overlap],x[:overlap]]=True
    session=FreshTargetInputs(seed,lambda rgb:(merged.astype(np.uint16),[dict(mask_id=1,class_id=0)]))
    _,evidence=session.prepare(frame,dict(frame_id='1',target_id='A',phase='approach'))
    assert evidence['other_reviewed_target_overlap_px']==overlap
    assert np.array_equal(decode_mask(evidence['mask_png']),merged)
    assert evidence['association']['status']=='matched'
    assert evidence['human_reviewed'] is False


def test_transport_needs_no_segmentation_and_cannot_revert():
    mask,seed,frame=inputs()
    def forbidden(rgb):raise AssertionError('segmentation in transport')
    session=FreshTargetInputs(seed,forbidden)
    value,evidence=session.prepare(frame,dict(frame_id='1',target_id='A',phase='transport'))
    assert np.array_equal(value['observation.images.target_global_rgb'],frame['observation.images.global_rgb'])
    assert evidence['mask_png'] is None
    with pytest.raises(ValueError,match='cannot revert'):
        session.prepare(frame,dict(frame_id='2',target_id='A',phase='approach'))


def paired_inputs():
    mask,seed,frame=inputs()
    other=decode_mask(seed['excluded_mask_png'])
    seed.update(target_selection_mode='paired_foreground',
                paired_targets=[dict(target_id='A',mask_png=seed['mask_png']),
                                dict(target_id='B',mask_png=seed['excluded_mask_png'])])
    return mask,other,seed,frame


def test_explicit_paired_input_separates_merged_candidate_and_saves_raw_evidence():
    import base64
    import cv2
    mask,other,seed,frame=paired_inputs()
    merged=(mask|other).astype(np.uint16)
    session=FreshTargetInputs(seed,lambda rgb:(merged,[dict(mask_id=1,class_id=0)]))
    value,evidence=session.prepare(frame,dict(frame_id='new',target_id='A',phase='approach'))
    assert evidence['target_selection_mode']=='paired_foreground'
    assert evidence['association']['status']=='separated'
    np.testing.assert_array_equal(decode_mask(evidence['mask_png']),mask)
    np.testing.assert_array_equal(decode_mask(evidence['paired_masks_png']['B']),other)
    decoded=cv2.imdecode(np.frombuffer(base64.b64decode(evidence['segmentation_instances_png']),np.uint8),-1)
    np.testing.assert_array_equal(decoded,merged)
    assert evidence['raw_foreground_other_seed_overlap_px']==int(other.sum())
    assert evidence['other_reviewed_target_overlap_px']==0
    assert evidence['human_reviewed'] is False
    assert 'observation.images.target_global_rgb' in value


def test_paired_mode_does_not_rename_only_competitor_as_selected_target():
    mask,other,seed,frame=paired_inputs()
    session=FreshTargetInputs(seed,lambda rgb:(other.astype(np.uint16),[dict(mask_id=1,class_id=0)]))
    with pytest.raises(ValueError,match='no current foreground'):
        session.prepare(frame,dict(frame_id='new',target_id='A',phase='approach'))


def test_multi_empty_evidence_and_next_fresh_frame_preserve_indices_and_seeds():
    from lerobot.bamboo_sorting.joint_target_runtime import TargetForegroundMissing
    mask,other,seed,frame=paired_inputs()
    seed['target_selection_mode']='multi_foreground'
    seed['visible_targets']=seed.pop('paired_targets')
    maps=[mask|other,other,mask|other]
    session=FreshTargetInputs(seed,lambda rgb:(maps.pop(0).astype(np.uint16),[dict(mask_id=1,class_id=0)]))
    session.prepare(frame,dict(frame_id='first',target_id='A',phase='approach'))
    with pytest.raises(TargetForegroundMissing) as caught:
        session.prepare(frame,dict(frame_id='lost',target_id='A',phase='approach'))
    evidence=caught.value.evidence
    assert evidence['association']['status']=='missing'
    assert evidence['segmentation_segments']==[dict(mask_id=1,class_id=0)]
    assert evidence['visible_masks_png']['A'] is None
    np.testing.assert_array_equal(decode_mask(evidence['visible_masks_png']['B']),other)
    assert not decode_mask(evidence['mask_png'],allow_empty=True).any()
    assert session.last_frame=='lost' and session.index==2 and session.separator.last_frame==1
    with pytest.raises(ValueError,match='fresh frame'):
        session.prepare(frame,dict(frame_id='lost',target_id='A',phase='approach'))
    _,recovered=session.prepare(frame,dict(frame_id='new',target_id='A',phase='approach'))
    assert session.index==3 and recovered['separation']['frame']==2
    np.testing.assert_array_equal(decode_mask(recovered['mask_png']),mask)
    assert session.transport_started is False
    np.testing.assert_array_equal(session.separator.references['A'],mask)


def test_paired_transport_uses_raw_rgb_without_segmentation():
    _,_,seed,frame=paired_inputs()
    def forbidden(rgb):raise AssertionError('transport segmentation')
    value,evidence=FreshTargetInputs(seed,forbidden).prepare(frame,dict(frame_id='new',target_id='A',phase='transport'))
    assert evidence['target_selection_mode']=='paired_foreground'
    assert evidence['mask_png'] is None
    np.testing.assert_array_equal(value['observation.images.target_global_rgb'],frame['observation.images.global_rgb'])


@pytest.mark.parametrize('bad',['mode','duplicate_id','mismatched_seed'])
def test_paired_seed_cannot_silently_change_reviewed_identity(bad):
    _,_,seed,_=paired_inputs()
    if bad=='mode':seed['target_selection_mode']='unknown'
    elif bad=='duplicate_id':seed['paired_targets'][1]['target_id']='A'
    else:seed['paired_targets'][0]['mask_png']=seed['excluded_mask_png']
    with pytest.raises(ValueError):FreshTargetInputs(seed,None)


@pytest.mark.parametrize('bad',['identity','repeat','unreviewed','phase'])
def test_invalid_target_provenance_is_rejected(bad):
    mask,seed,frame=inputs()
    if bad=='unreviewed':
        seed['human_reviewed']=False
        with pytest.raises(ValueError,match='reviewed'):FreshTargetInputs(seed,None)
        return
    session=FreshTargetInputs(seed,lambda rgb:(mask.astype(np.uint16),[dict(mask_id=1,class_id=0)]))
    session.prepare(frame,dict(frame_id='1',target_id='A',phase='approach'))
    with pytest.raises(ValueError,match='identity'):
        session.prepare(frame,dict(frame_id='1' if bad=='repeat' else '2',
            target_id='B' if bad=='identity' else 'A',phase='bad' if bad=='phase' else 'approach'))
