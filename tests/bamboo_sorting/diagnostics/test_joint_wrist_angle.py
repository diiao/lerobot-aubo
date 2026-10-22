import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
from tests.bamboo_sorting.paths import REPO_ROOT


path = REPO_ROOT / 'examples/phone_to_auboi10/diagnose_joint_wrist_angle.py'
spec = importlib.util.spec_from_file_location('wrist_angle', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample(angle=12, offset=8):
    rgb = np.full((480, 640, 3), 255, np.uint8)
    center = np.array([320-offset, 240.])
    direction = np.array([-np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    ends = [tuple(np.rint(center + t*direction).astype(int)) for t in (-130, 130)]
    cv2.line(rgb, ends[0], ends[1], (180, 130, 70), 15)
    return rgb


def test_signed_projected_angle_and_pixel_offset():
    gap = [[320, 80], [320, 400]]
    roi = [[180, 70], [450, 70], [450, 420], [180, 420]]
    result, _ = m.measure_alignment(sample(), gap, roi)
    assert result['valid']
    assert result['angle_deg'] == pytest.approx(12, abs=.5)
    assert result['lateral_offset_px'] == pytest.approx(8, abs=.5)
    result, _ = m.measure_alignment(np.full((480, 640, 3), 255, np.uint8), gap, roi)
    assert not result['valid'] and result['reason'] == 'missing_or_occluded'


def test_measurement_requires_review_and_keeps_raw_input(tmp_path):
    rows = []
    for i, angle in enumerate((-15, 0, 15)):
        image = tmp_path/f'{i}.png'
        Image.fromarray(sample(angle)).save(image)
        rows.append(dict(id=str(i), image=image.name, gap_line=[[320, 80], [320, 400]],
                         roi_polygon=[[180, 70], [450, 70], [450, 420], [180, 420]],
                         group='expert', episode_id=str(i), horizon_s=.5, delta_j6_deg=angle*.4))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='joint_wrist_angle_samples', samples=rows)))
    before = (tmp_path/'0.png').read_bytes()
    report = m.measure_manifest(manifest, tmp_path/'output')
    assert (tmp_path/'0.png').read_bytes() == before
    assert not report['human_reviewed'] and report['causal_diagnosis'] is None
    assert (tmp_path/'output/review.html').exists()
    with pytest.raises(FileExistsError):
        m.measure_manifest(manifest, tmp_path/'output')
    review = dict(schema='joint_wrist_angle_review', manifest=report['manifest'], human_reviewed=False,
                  reviewed_at='synthetic-test', samples=[dict(id=r['id'], measurement=r['measurement']) for r in report['samples']])
    decision_path = tmp_path/'review.json'
    decision_path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='explicit measurement'):
        m.regress_reviewed(tmp_path/'output', decision_path)
    review['human_reviewed'] = True
    decision_path.write_text(json.dumps(review))
    result = m.regress_reviewed(tmp_path/'output', decision_path)
    assert result['groups'][0]['slope_deg_per_image_degree'] == pytest.approx(.4, abs=.02)
    assert result['causal_diagnosis'] is None
    report.pop('measurement_method')
    (tmp_path/'output/measurements.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='legacy color-only'):
        m.regress_reviewed(tmp_path/'output', decision_path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def write_video(path, count=50):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 640, 480, 'yuv420p'
        for i in range(count):
            rgb = sample(angle=i % 15)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def demo_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, evidence = tmp_path/'dataset', tmp_path/'capture'
    # Episode 7 starts at video time 1.0, not frame zero of the shared MP4.
    key = 'videos/observation.images.grasp_rgb'
    meta = dict(episode_index=7, **{'data/chunk_index':0, 'data/file_index':0,
                 key+'/chunk_index':0, key+'/file_index':0, key+'/from_timestamp':1., key+'/to_timestamp':2.})
    frames, clocks, data = [], [], []
    for i in range(25):
        t = 100+i*.04
        action = [0., 0., 0., 0., 0., 10+2*i*.04, 100. if i >= 20 else 0.]
        frames.append(dict(frame_index=i, action=action,
                           joint_command_trace=dict(accepted_monotonic_s=t+.01, return_code=0)))
        clocks.append(dict(frame_index=i, timestamps={k:{'host_receive_monotonic_s':t+offset}
                      for k,offset in [('global_rgb',0),('grasp_rgb',.005),('robot_state',.007)]}))
        data.append(dict(episode_index=7, frame_index=i, timestamp=i*.04, action=action,
                         **{'observation.state':[0.,0.,0.,0.,0.,9+2*i*.04,0.]}))
    for name, rows in [('meta/episodes/chunk-000/file-000.parquet',[meta]), ('data/chunk-000/file-000.parquet',data)]:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    write_json(evidence/'session.json',dict(attempts=[dict(status='saved_pending_finalize', episode_index=7,
               joint_frames=frames, sensor_and_suction_evidence=dict(timestamp_journal=dict(frames=clocks)))]))
    write_video(root/key/'chunk-000/file-000.mp4')
    return dict(root=str(root), evidence_root=str(evidence))


def model_fixture(tmp_path):
    root=tmp_path/'trial'
    chunk=[[0.,0.,0.,0.,0.,20+4*i*.04,0.] for i in range(50)]
    trace=[]
    for i,t in enumerate((200.,200.2,200.4)):
        trace.append(dict(index=i, sensor_times=[t,t+.005,t+.007], received_at=t+.4,
                          observed_state=[0.,0.,0.,0.,0.,19.,0.],
                          prediction=dict(action_chunk=chunk,target_input=dict(phase='approach'))))
    trace.append(dict(target_phase_command=dict(first_accepted_close_s=200.81)))
    write_jsonl(root/'trace.jsonl',trace)
    side=[dict(frame_index=i,video_time_s=i*.04,source_timestamp=200+i*.04+.005,
               repeated_source=False,filled_missing_tick=False) for i in range(25)]
    write_jsonl(root/'videos/grasp_rgb_frames.jsonl',side)
    write_video(root/'videos/grasp_rgb.mp4',25)
    return dict(root=str(root),action_fps=25)


def test_prepare_aligns_source_clocks_video_offsets_and_future_targets(tmp_path):
    demo=demo_fixture(tmp_path)
    run=model_fixture(tmp_path)
    source=tmp_path/'sources.json'
    write_json(source,dict(defaults=dict(gap_line=[[320,80],[320,400]]), demonstrations=[demo],runs=[run],
                           horizons_s=[.5], before_close_s=[1,0],sample_interval_s=.2))
    result=m.prepare_manifest(source,tmp_path/'prepared')
    expert=next(r for r in result['samples'] if r['group']=='expert')
    model=next(r for r in result['samples'] if r['group']=='model')
    assert expert['video_time_s']==1. and expert['decoded_video_time_s']==1.
    assert expert['target_bracket_frame_indices']==[12,13]
    assert expert['delta_j6_deg']==pytest.approx(1.98)  # accepted commands lag observations by .01 s
    assert model['target_chunk_indices']==[12,13]
    assert model['delta_j6_deg']==pytest.approx(3.)
    assert model['alignment_error_s']==pytest.approx(0.)
    assert model['prediction_latency_s']==pytest.approx(.4)  # receive time must not anchor plan
    assert model['roi_source']=='default_central_region_requires_review'
    assert all(r['observation_anchor_monotonic_s']+r['horizon_s'] < r['accepted_close_monotonic_s'] for r in result['samples'])
    assert any(r['reason']=='future_target_outside_chunk_or_window_or_close' for r in result['exclusions'])
    report=m.measure_manifest(tmp_path/'prepared/manifest.json',tmp_path/'measured')
    valid=next(r for r in report['samples'] if r['measurement']['valid'])
    decision=tmp_path/'decision.json'
    write_json(decision,dict(schema='joint_wrist_angle_review',manifest=report['manifest'],human_reviewed=True,
                            reviewed_at='synthetic',samples=[dict(id=valid['id'],measurement=valid['measurement'])]))
    with pytest.raises(ValueError,match='descent segment review'):
        m.regress_reviewed(tmp_path/'measured',decision)
    with pytest.raises(FileExistsError):
        m.prepare_manifest(source,tmp_path/'prepared')


def test_model_excludes_missing_duplicate_and_filled_observations(tmp_path):
    run=model_fixture(tmp_path)
    window_rows=list(m.model_candidates(dict(run,before_close_s=[1,.5]),
        dict(sample_interval_s=.01,horizons_s=[.2],max_alignment_error_s=.02),[]))
    assert [r['prediction_index'] for r in window_rows]==[0]  # future response stays inside selected segment
    root=Path(run['root'])
    side=m.read_jsonl(root/'videos/grasp_rgb_frames.jsonl')
    for r in side:
        if r['frame_index']<=6:
            r['filled_missing_tick']=True
    write_jsonl(root/'videos/grasp_rgb_frames.jsonl',side)
    config=dict(before_close_s=[1,0],sample_interval_s=.01,horizons_s=[.1],max_alignment_error_s=.02)
    skipped=[]
    rows=list(m.model_candidates(run,config,skipped))
    assert len(rows)==1 and rows[0]['prediction_index']==2
    assert len(skipped)==2 and all(r['reason']=='missing_aligned_video_frame' for r in skipped)


def test_interpolation_rejects_gaps_extrapolation_and_closed_targets():
    actions=np.zeros((3,7));actions[:,5]=[10,11,12]
    assert m.interpolate_target(np.array([1.,1.04,1.3]),actions,1.15,.12) is None
    assert m.interpolate_target(np.array([1.,1.04,1.3]),actions,.9,.12) is None
    actions[1,6]=100
    assert m.interpolate_target(np.array([1.,1.04,1.08]),actions,1.02,.12) is None


def test_training_manifest_uses_annotated_subset_and_episode_window(tmp_path):
    demo=demo_fixture(tmp_path)
    annotations=tmp_path/'annotations.json'
    write_json(annotations,dict(episodes=[dict(episode_index=7)]))
    training=tmp_path/'training.json'
    write_json(training,dict(train=[dict(demo,target_annotations=str(annotations))],validation=[]))
    source=tmp_path/'sources.json'
    write_json(source,dict(defaults=dict(gap_line=[[320,80],[320,400]]),training_sources=training.name,
                          before_close_s=[.5,0],horizons_s=[.1]))
    report=m.prepare_manifest(source,tmp_path/'prepared')
    assert all(r['frame_index']>=8 and r['episode_id'].endswith('episode-7') for r in report['samples'])
    assert m.candidate_window({'before_close_s':[3,0]}, {'episode_windows':{'7':[2,1]}},100,7)==(98,99)


def test_planned_close_then_open_is_excluded_and_duplicate_video_deduplicated(tmp_path):
    run=model_fixture(tmp_path)
    root=Path(run['root'])
    trace=m.read_jsonl(root/'trace.jsonl')
    duplicate=json.loads(json.dumps(trace[0]))
    duplicate.update(index=99,sensor_times=[200.012,200.017,200.019])
    trace.insert(1,duplicate)
    trace[2]['prediction']['action_chunk'][2][6]=100.
    write_jsonl(root/'trace.jsonl',trace)
    skipped=[]
    rows=list(m.model_candidates(run,dict(before_close_s=[1,0],sample_interval_s=.01,
                 horizons_s=[.1],max_alignment_error_s=.02),skipped))
    assert [r['prediction_index'] for r in rows]==[0,2]
    assert {r['reason'] for r in skipped}=={'duplicate_video_observation','future_target_outside_chunk_or_window_or_close'}


def test_measure_accepts_shared_gap_line_and_default_roi(tmp_path):
    Image.fromarray(sample()).save(tmp_path/'image.png')
    write_json(tmp_path/'manifest.json',dict(schema='joint_wrist_angle_samples',
        defaults=dict(gap_line=[[320,80],[320,400]]),samples=[dict(id='one',image='image.png')]))
    report=m.measure_manifest(tmp_path/'manifest.json',tmp_path/'measured')
    assert report['samples'][0]['measurement']['valid']
    assert report['samples'][0]['roi_polygon']==m.default_roi(640,480)


def test_fixed_jaw_exclusion_keeps_broad_strip_and_rejects_short_fragments():
    gap=[[320,80],[320,400]]
    roi=[[160,70],[480,70],[480,430],[160,430]]
    jaw=[[[430,220],[480,220],[480,430],[430,430]]]
    rgb=np.full((480,640,3),255,np.uint8)
    cv2.rectangle(rgb,(280,75),(390,425),(180,130,70),-1)
    cv2.rectangle(rgb,(445,230),(470,420),(180,130,70),-1)
    result,overlay=m.measure_alignment(rgb,gap,roi,jaw)
    assert result['valid'] and result['angle_deg']==pytest.approx(0,abs=.1)
    assert result['axis_span_px']>result['min_axis_span_px']
    assert result['gripper_excluded_pixels']>0
    assert not np.array_equal(rgb[260,450],overlay[260,450])
    rgb[:,270:400]=255
    result,_=m.measure_alignment(rgb,gap,roi,jaw)
    assert not result['valid']  # the jaw alone must never become the target
    cv2.rectangle(rgb,(310,250),(325,350),(180,130,70),-1)
    result,_=m.measure_alignment(rgb,gap,roi,jaw)
    assert not result['valid'] and result['rejected_short_or_broad_components']


def test_validity_counts_unique_frames_and_flags_outcome_selection_bias():
    rows=[]
    for outcome,valid in [('success',False),('failure',True)]:
        for i in range(2):
            for horizon in (.5,1.):
                rows.append(dict(image=f'{outcome}-{i}.png',episode_id=outcome,group='model',
                                 trial_outcome=outcome,horizon_s=horizon,
                                 measurement=dict(valid=valid,reason='multiple_visible_components')))
    result=m.measurement_coverage(rows)
    assert result['frames']==4 and result['program_valid_frames']==2
    assert result['sample_rows']==8
    assert result['by_model_outcome']['success']['frames']==2
    assert result['outcome_rate_gap']==1 and result['selection_bias_warning']


def test_stratification_keeps_all_slanted_episodes_before_measurement(tmp_path):
    episodes=[]
    for i,angle in enumerate([-28,-25,-23,-21,-19,-16,-12,-8,-2,2,8,12]):
        mask=np.zeros((480,640),np.uint8)
        direction=np.array([np.sin(np.radians(angle)),np.cos(np.radians(angle))])
        center=np.array([320.,240.])
        ends=[tuple(np.rint(center+t*direction).astype(int)) for t in (-170,170)]
        cv2.line(mask,*ends,255,12)
        Image.fromarray(mask).save(tmp_path/f'mask-{i}.png')
        episodes.append(dict(episode_index=i,scene_id=f't-{i}',target_id='s-1',reviewed=True,
                             frames=[dict(mask_path=f'mask-{i}.png',reviewed=True)]))
    path=tmp_path/'annotations.json';write_json(path,dict(episodes=episodes))
    sources=[(dict(root='unused',target_annotations=str(path)),'expert')]
    strata=[dict(name='slanted',angle_range_deg=[-30,-15],take=None,expected_pool_count=6),
            dict(name='straight',angle_range_deg=[-5,5],take=2),
            dict(name='moderate',angle_range_deg=[5,15],take=1)]
    selected,report=m.stratify_expert_sources(sources,strata)
    assert selected[0][0]['episode_indices'][:6]==[0,1,2,3,4,5]
    assert len(report['selected_episodes'])==9
    assert set(r['angle_stratum'] for r in report['selected_episodes'])=={'slanted','straight','moderate'}
    strata[0]['expected_pool_count']=7
    with pytest.raises(ValueError,match='coverage changed'):
        m.stratify_expert_sources(sources,strata)


def test_image_only_expert_keeps_frames_without_future_targets(tmp_path):
    source=demo_fixture(tmp_path)
    config=dict(measurement_only=True,full_preclose=True,sample_interval_s=.2)
    rows=list(m.demonstration_candidates(source,config,[]))
    assert rows[0]['frame_index']==0 and rows[-1]['frame_index']==20
    assert all(r['measurement_only'] and 'horizon_s' not in r and 'delta_j6_deg' not in r for r in rows)
    assert rows[-1]['accepted_close_monotonic_s']-rows[-1]['wrist_source_monotonic_s']<.01


def test_image_only_model_uses_video_without_predictions_or_closure(tmp_path):
    source=model_fixture(tmp_path);root=Path(source['root'])
    write_jsonl(root/'trace.jsonl',[dict(error='interrupted before prediction')])
    rows=list(m.model_candidates(dict(root=str(root)),dict(measurement_only=True,full_preclose=True,sample_interval_s=.2),[]))
    assert [r['frame_index'] for r in rows]==[0,5,10,15,20]
    assert all(r['accepted_close_monotonic_s'] is None and r['stage']=='no_close_unreviewed' for r in rows)
    assert all('prediction_index' not in r and not r['response_available'] for r in rows)
