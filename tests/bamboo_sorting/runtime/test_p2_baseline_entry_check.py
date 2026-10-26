"""Fake-process coverage for the P2 baseline entry check script."""
import importlib.util
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pytest

from tests.bamboo_sorting.paths import REPO_ROOT

CHECK = (REPO_ROOT / 'artifacts/capture/stack/joint_stack_train_20261017/'
         'p2_baseline_entry_check/check_baseline_entry.py')
if not CHECK.is_file():
    pytest.skip('P2 baseline entry check script artifact not present', allow_module_level=True)


def load_check(monkeypatch, root):
    monkeypatch.syspath_prepend(str(REPO_ROOT / 'src'))
    spec = importlib.util.spec_from_file_location('p2_baseline_entry_check', CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(root)
    # Synthetic selection keeps software tests independent of experiment data.
    module.MANIFEST = Path(root) / 'selection_manifest.json'
    cases = [dict(case_id=case_id, scene_id=case_id.split('_')[0], split='validation',
                  source_index=index // 2, episode_index=0, frame_index=10 + index,
                  input_phase=phase, source_dataset=str(Path(root) / 'raw'),
                  state=[0.] * 6 + [100. if phase == 'transport' else 0.])
             for index, (case_id, phase) in enumerate(module.SELECTION)]
    module.MANIFEST.write_text(json.dumps(dict(cases=cases)) + '\n')
    return module


def prepare_inputs(module, monkeypatch):
    def fake_raw(case):
        images = {name: np.zeros((480, 640, 3), np.uint8) for name in ('global_rgb', 'grasp_rgb')}
        return [float(value) for value in case['state']], images
    monkeypatch.setattr(module, 'raw_observation', fake_raw)
    module.prepare()


def test_retired_manifest_keeps_explicit_frozen_record_provenance(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    prepared_path = tmp_path / 'prepared_inputs.json'
    prepared = json.loads(prepared_path.read_text())
    manifest = json.loads(module.MANIFEST.read_text())
    manifest['retired_source'] = dict(path=prepared['manifest'],
                                     sha256_recorded=prepared['manifest_sha256'], available=False)
    manifest['frozen_input_record_sha256'] = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    module.MANIFEST.write_text(json.dumps(manifest) + '\n')
    assert module.prepared_problems(prepared) == []
    assert module.source_manifest_status(prepared) == (
        'retired_source_unavailable_frozen_baseline_record_verified')
    prepared['cases'][0]['state'][0] += 1
    prepared_path.write_text(json.dumps(prepared) + '\n')
    assert module.prepared_problems(prepared)


def worker_parts(ready_overrides=None):
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK, joint_contract_record
    from lerobot.bamboo_sorting.model_profiles import BASELINE_STACK_MODEL_SHA
    chunk = [[0.] * 7 for _ in range(50)]
    ready = dict(ready=True, checkpoint_sha256=BASELINE_STACK_MODEL_SHA,
                 contract=joint_contract_record(), bf16=False, task=JOINT_TASK,
                 action_chunk_available=True, target_live=False, segmentation_checkpoint=None)
    ready.update(ready_overrides or {})
    responses = [ready] + [dict(id=index, checkpoint_sha256=BASELINE_STACK_MODEL_SHA,
                                shape=[50, 7], action=chunk[0], raw_action=chunk[0],
                                action_chunk=chunk, raw_action_chunk=chunk) for index in range(6)]
    return ready, responses


def install_worker(monkeypatch, *, packets=None, hang_on=None, ready_overrides=None):
    from lerobot.bamboo_sorting.runtime import inference
    ready, responses = worker_parts(ready_overrides)

    class Process:
        instances = []

        def __init__(self, command, **kwargs):
            self.command = command
            self.stdin = io.StringIO()
            Process.instances.append(self)

        def wait(self, timeout):
            pass

    monkeypatch.setattr(inference.subprocess, 'Popen', Process)
    messages = iter(responses)
    calls = {'count': 0}

    def receive(self, timeout):
        calls['count'] += 1
        if packets is not None:
            packets.append(self.process.stdin.getvalue())
        if hang_on == 'load' and calls['count'] == 1:
            time.sleep(2)
        if hang_on == 'predict' and calls['count'] == 2:
            time.sleep(2)
        return next(messages)

    monkeypatch.setattr(inference.InferencePipe, 'receive', receive)
    return Process, ready


def test_fake_worker_predict_then_verify_passes(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    packets = []
    Process, _ = install_worker(monkeypatch, packets=packets)
    assert module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '300']) == 0

    ready = json.loads((tmp_path / 'ready.json').read_text())
    assert ready['requested'] == dict(target_conditioned=False, return_action_chunk=True,
                                      live_target_seed=None, segmentation_checkpoint=None)
    assert ready['worker_ready']['target_live'] is False
    assert ready['worker_ready']['segmentation_checkpoint'] is None
    command = ' '.join(Process.instances[-1].command)
    assert '--return-action-chunk' in command and '--target-conditioned' not in command
    assert '--target-segmentation-checkpoint' not in command
    lines = [line for line in packets[-1].splitlines() if line]
    assert len(lines) == 6
    for line in lines:
        packet = json.loads(line)
        assert packet['codec'] == 'rgb_zlib' and packet['seed'] == 42
        assert not any(key in packet for key in
                       ('target_contract', 'target_input', 'target_seed', 'target_motion'))

    assert module.main(['--stage', 'verify']) == 0
    verify = json.loads((tmp_path / 'verify.json').read_text())
    assert verify['passed'] is True
    assert verify['within_output_budget'] is True and verify['within_hard_timeout'] is True


@pytest.mark.parametrize('hang_on', ['load', 'predict'])
def test_predict_hard_timeout_covers_loading_and_prediction(monkeypatch, tmp_path, hang_on):
    root = tmp_path / hang_on
    root.mkdir()
    module = load_check(monkeypatch, root)
    prepare_inputs(module, monkeypatch)
    install_worker(monkeypatch, hang_on=hang_on)
    with pytest.raises(TimeoutError, match='hard timeout'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '0.2'])
    error = json.loads((root / 'error.json').read_text())
    assert 'TimeoutError' in error['error']
    assert not (root / 'complete.json').exists()
    with pytest.raises(SystemExit) as stopped:
        module.main(['--stage', 'verify'])
    assert stopped.value.code == 1
    verification = json.loads((root / 'verify.json').read_text())
    assert verification['passed'] is False and verification['problems']
    assert (root / 'verify_failure.json').exists()


def _run_successful_predict(module, monkeypatch):
    install_worker(monkeypatch)
    assert module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '300']) == 0


@pytest.mark.parametrize('mutation', [
    'output_budget_flag', 'timeout_flag', 'elapsed', 'elapsed_negative', 'output_bytes',
    'hard_timeout_widened', 'frozen_mismatch', 'completed_predictions', 'hardware_access',
    'error_file', 'drop_prediction', 'missing_complete', 'ready_script', 'ready_checkpoint',
    'worker_segmentation',
])
def test_verify_rejects_failure_incomplete_or_over_budget(monkeypatch, tmp_path, mutation):
    root = tmp_path / mutation
    root.mkdir()
    module = load_check(monkeypatch, root)
    prepare_inputs(module, monkeypatch)
    _run_successful_predict(module, monkeypatch)
    complete_path = root / 'complete.json'
    ready_path = root / 'ready.json'
    complete = json.loads(complete_path.read_text())
    ready = json.loads(ready_path.read_text())
    if mutation == 'output_budget_flag':
        complete['within_output_budget'] = False
    elif mutation == 'timeout_flag':
        complete['within_hard_timeout'] = False
    elif mutation == 'elapsed':
        complete['elapsed_s'] = complete['hard_timeout_s'] + 1
    elif mutation == 'elapsed_negative':
        complete['elapsed_s'] = -5
    elif mutation == 'output_bytes':
        complete['output_bytes'] = module.MAX_OUTPUT_BYTES + 1
    elif mutation == 'hard_timeout_widened':
        complete['hard_timeout_s'] = complete['frozen_hard_timeout_s'] + 600
        complete['elapsed_s'] = complete['frozen_hard_timeout_s'] + 100
    elif mutation == 'frozen_mismatch':
        complete['frozen_hard_timeout_s'] += 1
    elif mutation == 'completed_predictions':
        complete['completed_predictions'] = 5
    elif mutation == 'hardware_access':
        complete['hardware_access'] = True
    elif mutation == 'error_file':
        (root / 'error.json').write_text(json.dumps(dict(stage='predict', error='simulated')) + '\n')
    elif mutation == 'drop_prediction':
        lines = (root / 'predictions.jsonl').read_text().splitlines()[:-1]
        (root / 'predictions.jsonl').write_text('\n'.join(lines) + '\n')
    elif mutation == 'missing_complete':
        complete_path.unlink()
    elif mutation == 'ready_script':
        ready['inference_script'] = 'wrong-worker.py'
        complete['ready'] = ready
    elif mutation == 'ready_checkpoint':
        ready['checkpoint'] = 'wrong-checkpoint'
        complete['ready'] = ready
    elif mutation == 'worker_segmentation':
        ready['worker_ready']['segmentation_checkpoint'] = 'segmenter'
        complete['ready'] = ready
    if mutation in ('ready_script', 'ready_checkpoint', 'worker_segmentation'):
        ready_path.write_text(json.dumps(ready, allow_nan=False) + '\n')
    if mutation != 'missing_complete':
        complete_path.write_text(json.dumps(complete, allow_nan=False) + '\n')
    with pytest.raises(SystemExit) as stopped:
        module.main(['--stage', 'verify'])
    assert stopped.value.code == 1
    verification = json.loads((root / 'verify.json').read_text())
    assert verification['passed'] is False and verification['problems']
    failure = json.loads((root / 'verify_failure.json').read_text())
    assert failure['passed'] is False and failure['problems']


def test_predict_rejects_widened_hard_timeout(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    def forbidden():
        raise AssertionError('worker setup reached after a widened budget')
    monkeypatch.setattr(module, 'entry_config', forbidden)
    with pytest.raises(ValueError, match='frozen budget'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '900'])
    assert (tmp_path / 'error.json').exists()
    assert not (tmp_path / 'complete.json').exists()


def test_predict_hard_timeout_covers_entry_setup(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    monkeypatch.setattr(module, 'entry_config', lambda: time.sleep(2))
    with pytest.raises(TimeoutError, match='hard timeout'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '0.2'])
    assert (tmp_path / 'error.json').exists()
    assert not (tmp_path / 'complete.json').exists()


def test_predict_rejects_segmentation_worker(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    install_worker(monkeypatch, ready_overrides={'segmentation_checkpoint': 'segmenter'})
    with pytest.raises(ValueError, match='target/segmentation support'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '300'])
    assert (tmp_path / 'error.json').exists()
    assert not (tmp_path / 'complete.json').exists()


def test_predict_rejects_target_live_worker(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    install_worker(monkeypatch, ready_overrides={'target_live': True})
    with pytest.raises(ValueError, match='live target protocol'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '300'])
    assert (tmp_path / 'error.json').exists()
    assert not (tmp_path / 'complete.json').exists()


def test_predict_rejects_drifted_entry_identity(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    original = module.entry_config

    def drifted():
        _, config, plan = original()
        config['pipe_options']['checkpoint'] = 'wrong-checkpoint'
        return None, config, plan

    monkeypatch.setattr(module, 'entry_config', drifted)
    with pytest.raises(ValueError, match='entry options'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '300'])
    assert (tmp_path / 'error.json').exists()
    assert not (tmp_path / 'complete.json').exists()


def test_close_timeout_fails_without_complete(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    install_worker(monkeypatch)
    from lerobot.bamboo_sorting.runtime import inference
    monkeypatch.setattr(inference.InferencePipe, 'close', lambda self: time.sleep(2))
    with pytest.raises(TimeoutError, match='hard timeout'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '0.5'])
    error = json.loads((tmp_path / 'error.json').read_text())
    assert 'TimeoutError' in error['error'] and 'hard timeout' in error['error']
    assert not (tmp_path / 'complete.json').exists()


def test_close_error_does_not_mask_primary_timeout(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    install_worker(monkeypatch, hang_on='predict')
    from lerobot.bamboo_sorting.runtime import inference
    def failing_close(self):
        raise RuntimeError('close failed')
    monkeypatch.setattr(inference.InferencePipe, 'close', failing_close)
    with pytest.raises(TimeoutError, match='hard timeout'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '0.2'])
    error = json.loads((tmp_path / 'error.json').read_text())
    assert error['error'].startswith('TimeoutError') and 'close failed' not in error['error']


@pytest.mark.parametrize('mutation', ['complete_list', 'ready_list', 'budget_list',
                                      'budget_widened', 'requested_list', 'worker_ready_list'])
def test_verify_rejects_non_object_records(monkeypatch, tmp_path, mutation):
    root = tmp_path / mutation
    root.mkdir()
    module = load_check(monkeypatch, root)
    prepare_inputs(module, monkeypatch)
    _run_successful_predict(module, monkeypatch)
    complete_path = root / 'complete.json'
    ready_path = root / 'ready.json'
    prepared_path = root / 'prepared_inputs.json'
    if mutation == 'complete_list':
        complete_path.write_text('[]\n')
    elif mutation == 'ready_list':
        ready_path.write_text('[]\n')
    elif mutation == 'budget_list':
        prepared = json.loads(prepared_path.read_text())
        prepared['budget'] = []
        prepared_path.write_text(json.dumps(prepared, allow_nan=False) + '\n')
    elif mutation == 'budget_widened':
        prepared = json.loads(prepared_path.read_text())
        prepared['budget']['hard_timeout_s'] = 900
        prepared_path.write_text(json.dumps(prepared, allow_nan=False) + '\n')
    else:
        ready = json.loads(ready_path.read_text())
        complete = json.loads(complete_path.read_text())
        ready['requested' if mutation == 'requested_list' else 'worker_ready'] = []
        complete['ready'] = ready
        ready_path.write_text(json.dumps(ready, allow_nan=False) + '\n')
        complete_path.write_text(json.dumps(complete, allow_nan=False) + '\n')
    with pytest.raises(SystemExit) as stopped:
        module.main(['--stage', 'verify'])
    assert stopped.value.code == 1
    assert json.loads((root / 'verify.json').read_text())['passed'] is False


def test_predict_rejects_tampered_prepared_budget(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    prepared_path = tmp_path / 'prepared_inputs.json'
    prepared = json.loads(prepared_path.read_text())
    prepared['budget']['hard_timeout_s'] = 900
    prepared_path.write_text(json.dumps(prepared, allow_nan=False) + '\n')
    with pytest.raises(ValueError, match='not the frozen'):
        module.main(['--stage', 'predict', '--confirm-real-gpu', '--max-seconds', '900'])
    assert (tmp_path / 'error.json').exists()
    assert not (tmp_path / 'complete.json').exists()


@pytest.mark.parametrize('mutation', ['missing_input', 'modified_input', 'modified_state'])
def test_verify_rejects_missing_or_modified_frozen_input(monkeypatch, tmp_path, mutation):
    root = tmp_path / mutation
    root.mkdir()
    module = load_check(monkeypatch, root)
    prepare_inputs(module, monkeypatch)
    _run_successful_predict(module, monkeypatch)
    prepared_path = root / 'prepared_inputs.json'
    prepared = json.loads(prepared_path.read_text())
    image = root / prepared['cases'][0]['images']['global_rgb']['path']
    if mutation == 'missing_input':
        image.unlink()
    elif mutation == 'modified_input':
        image.write_bytes(image.read_bytes() + b'changed')
    else:
        prepared['cases'][0]['state'][0] += 1
        prepared_path.write_text(json.dumps(prepared, allow_nan=False) + '\n')
    with pytest.raises(SystemExit) as stopped:
        module.main(['--stage', 'verify'])
    assert stopped.value.code == 1
    assert json.loads((root / 'verify.json').read_text())['passed'] is False


def test_reverify_archives_previous_pass_and_failure_is_canonical(monkeypatch, tmp_path):
    module = load_check(monkeypatch, tmp_path)
    prepare_inputs(module, monkeypatch)
    _run_successful_predict(module, monkeypatch)
    assert module.main(['--stage', 'verify']) == 0
    assert module.main(['--stage', 'verify']) == 0
    assert list(tmp_path.glob('verify_previous_*.json'))
    assert json.loads((tmp_path / 'verify.json').read_text())['passed'] is True
    complete_path = tmp_path / 'complete.json'
    complete = json.loads(complete_path.read_text())
    complete['failed'] = True
    complete_path.write_text(json.dumps(complete, allow_nan=False) + '\n')
    with pytest.raises(SystemExit) as stopped:
        module.main(['--stage', 'verify'])
    assert stopped.value.code == 1
    assert json.loads((tmp_path / 'verify.json').read_text())['passed'] is False
    previous = sorted(tmp_path.glob('verify_previous_*.json'))
    assert previous and all(json.loads(path.read_text())['passed'] is True for path in previous)
