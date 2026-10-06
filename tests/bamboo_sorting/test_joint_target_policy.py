"""Causal phase switching and training/runtime parity without models or devices."""
import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_FIELDS, JOINT_TASK, joint_contract_record
from lerobot.bamboo_sorting.joint_target_policy import (
    JointTargetDataset, SmolVLAJointTargetOfflineAdapter, TargetInputSession,
    prepare_target_source, target_policy_contract_record,
)
from lerobot.bamboo_sorting.smolvla_joint_adapter import (
    SmolVLAJointOfflineAdapter, make_joint_smolvla_config,
)


def sample(index):
    state = torch.zeros(7)
    state[-1] = 100 if index in (2, 3) else 0
    action = torch.full((50, 7), float(index))
    action[:, -1] = 100 if index in (1, 2) else 0
    return {'episode_index': 0, 'frame_index': index, 'observation.state': state,
            'action': action, 'action_is_pad': torch.arange(50) >= 5-index, 'task': JOINT_TASK,
            'observation.images.global_rgb': torch.full((3, 480, 640), 37 / 255),
            'observation.images.grasp_rgb': torch.full((3, 480, 640), 53 / 255)}


def bound_capture(tmp_path):
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    rows, clocks = [], []
    # Frame 2 arrives after close, but its camera source timestamp precedes it.
    camera_times = [10., 11., 11.1, 13., 14.]
    mask = np.zeros((480, 640), np.uint8)
    mask[10:20, 10:80] = 255
    cv2.imwrite(str(tmp_path / 'mask.png'), mask)
    labels = []
    for i in range(5):
        rows.append({'frame_index': i, 'action': sample(i)['action'][0].tolist(),
                     'joint_command_trace': {'return_code': 0, 'accepted_monotonic_s': 10.5+i}})
        clocks.append({'frame_index': i, 'timestamps': {'global_rgb': {'host_receive_monotonic_s': camera_times[i]}}})
        if i < 3:
            labels.append({'frame_index': i, 'schema': 'aubo_joint_target_frame', 'frame_id': f'0/global_rgb/{i}',
                           'target_id': 'strip_a', 'reviewed': True, 'status': 'visible', 'mask_path': 'mask.png'})
    capture = {'episode_index': 0, 'status': 'saved_pending_finalize',
               'metadata': {'target_id': 'strip_a', 'scene_id': 'pilot_001'}, 'joint_frames': rows,
               'sensor_and_suction_evidence': {'timestamp_journal': {'frames': clocks}}}
    (evidence / 'session.json').write_text(json.dumps({'attempts': [capture]}))
    spec = {'schema': 'aubo_joint_target_annotations', 'source_dataset_root': '/original/data',
            'episodes': [{'episode_index': 0, 'target_id': 'strip_a', 'scene_id': 'pilot_001',
                          'reviewed': True, 'frames': labels}]}
    annotation_path = tmp_path / 'annotations.json'
    annotation_path.write_text(json.dumps(spec))
    record = {'source_dataset_root': '/original/data', 'evidence_root': str(evidence),
              'episodes': [0], 'frames': 5}
    return record, annotation_path, camera_times


def test_complete_episode_training_runtime_parity_and_action_padding(tmp_path):
    record, path, times = bound_capture(tmp_path)
    prepared = prepare_target_source(record, path)
    assert [r['transport'] for r in prepared['target_frames']] == [False, False, False, True, True]
    source = [sample(i) for i in range(5)]
    dataset = JointTargetDataset(source, prepared['target_frames'])
    session = TargetInputSession('strip_a')
    mask = cv2.imread(str(tmp_path / 'mask.png'), 0) != 0
    # Shuffled dataset access must NOT change phase or target identity.
    expected = {i: dataset[i] for i in (4, 0, 3, 1, 2)}
    for i, raw in enumerate(source):
        info = prepared['target_frames'][i]
        frame = {'observation.state': raw['observation.state'].numpy(), 'task': JOINT_TASK,
                 **{f'observation.images.{name}': raw[f'observation.images.{name}'].mul(255).round()
                    .to(torch.uint8).permute(1, 2, 0).numpy() for name in ('global_rgb', 'grasp_rgb')}}
        runtime = session.prepare(frame, frame_id=info['frame_id'], global_timestamp=times[i],
                                  visible_mask=mask, annotation=info['annotation'])
        for name in ('target_global_rgb', 'grasp_rgb'):
            key = f'observation.images.{name}'
            torch.testing.assert_close(expected[i][key], torch.from_numpy(runtime[key]).permute(2, 0, 1).float()/255,
                                       rtol=0, atol=0)
        for key in ('observation.state', 'action', 'action_is_pad', 'observation.images.grasp_rgb'):
            assert torch.equal(expected[i][key], raw[key])
        assert 'observation.images.global_rgb' not in expected[i]
        assert torch.all(raw['observation.images.global_rgb'] == 37/255)
        session.accept_command(dict(zip(JOINT_FIELDS, raw['action'][0].tolist())), accepted_monotonic_s=10.5+i)
    assert session.accepted_close_time == 11.5  # Reopening never restarts approach.
    assert TargetInputSession('next_strip').accepted_close_time is None


@pytest.mark.parametrize('damage', ['missing', 'wrong_frame', 'wrong_target', 'draft', 'source'])
def test_target_preflight_rejects_unbound_or_unreviewed_masks(tmp_path, damage):
    record, path, _ = bound_capture(tmp_path)
    spec = json.loads(path.read_text())
    first = spec['episodes'][0]['frames'][0]
    if damage == 'missing': spec['episodes'][0]['frames'].pop(0)
    elif damage == 'wrong_frame': first['frame_id'] = '0/global_rgb/99'
    elif damage == 'wrong_target': first['target_id'] = 'strip_b'
    elif damage == 'draft': first['reviewed'] = False
    else: spec['source_dataset_root'] = '/other/data'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError): prepare_target_source(record, path)


def test_target_adapter_contracts_and_numerical_prediction():
    contract = target_policy_contract_record()
    with pytest.raises(ValueError):
        SmolVLAJointOfflineAdapter(None, None, None, contract=contract)
    with pytest.raises(ValueError):
        SmolVLAJointTargetOfflineAdapter(None, None, None, contract=joint_contract_record())
    policy = Mock(config=make_joint_smolvla_config(target_conditioned=True))
    policy.predict_action_chunk.return_value = torch.tensor([[[1., 2., 3., 4., 5., 6., 97.]]])
    adapter = SmolVLAJointTargetOfflineAdapter(policy, lambda x: x, lambda x: x, contract=contract)
    session = TargetInputSession('a')
    session.accept_command(dict(zip(JOINT_FIELDS, [0]*6+[100])), accepted_monotonic_s=1.)
    raw = {'observation.state': np.zeros(7, np.float32), 'task': JOINT_TASK,
           **{f'observation.images.{k}': np.zeros((480, 640, 3), np.uint8) for k in ('global_rgb', 'grasp_rgb')}}
    frame = session.prepare(raw, frame_id='frame', global_timestamp=2.)
    assert adapter(frame).tolist() == [[1, 2, 3, 4, 5, 6, 100]]
    inputs = policy.predict_action_chunk.call_args.args[0]
    assert set(inputs) == {'observation.state', 'task', 'observation.images.target_global_rgb', 'observation.images.grasp_rgb'}
    with pytest.raises(ValueError): adapter(raw)
    assert adapter.raw_actions is None


def test_target_capture_and_training_plan_do_not_open_devices(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'examples/phone_to_auboi10'
    capture = runpy.run_path(str(root / 'record_joint.py'))
    args = ['--target-demonstrations', '--num-episodes', '5', '--split', 'train',
            '--dataset-root', str(tmp_path/'data'), '--evidence-root', str(tmp_path/'evidence')]
    import builtins
    original = builtins.__import__
    def guarded(name, *a, **kw):
        if name.startswith(('lerobot.robots', 'lerobot.cameras', 'lerobot.teleoperators', 'pyaubo_sdk')):
            raise AssertionError(name)
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert capture['main'](args) == 0
    assert not (tmp_path/'data').exists()
    trainer = runpy.run_path(str(root / 'train_joint_smolvla.py'))
    assert trainer['main'](['--target-conditioned']) == 0
    with pytest.raises(SystemExit):
        trainer['parse_args'](['--target-conditioned', '--stage', 'preflight'])


def test_target_wire_images_require_new_key_and_preserve_pixels():
    import base64
    import runpy
    import zlib
    from pathlib import Path
    decode = runpy.run_path(str(Path(__file__).resolve().parents[2] /
        'examples/phone_to_auboi10/joint_inference_stdio.py'))['decode_images']
    original = np.full((480, 640, 3), 37, np.uint8)
    payload = base64.b64encode(zlib.compress(original.tobytes())).decode()
    images = {k: payload for k in ('target_global_rgb', 'grasp_rgb')}
    result = decode(images, 'rgb_zlib', image_keys=('target_global_rgb', 'grasp_rgb'))
    assert np.array_equal(result['observation.images.target_global_rgb'], original)
    with pytest.raises(ValueError, match='image keys'):
        decode(images, 'rgb_zlib')


def test_manifest_cannot_silently_drop_target_annotations(tmp_path):
    from lerobot.bamboo_sorting.joint_training import prepare_training_manifest
    manifest = tmp_path / 'sources.json'
    spec = {'schema_version': 'AuboJointTrainingSourcesV1',
            'train': [{'root': 'train', 'evidence_root': 'train-evidence', 'target_annotations': 'labels.json'}],
            'validation': {'root': 'val', 'evidence_root': 'val-evidence'}}
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='explicit --target-conditioned'):
        prepare_training_manifest(manifest)
    with pytest.raises(ValueError, match='every source'):
        prepare_training_manifest(manifest, target_conditioned=True)
