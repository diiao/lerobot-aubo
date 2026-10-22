import importlib.util
from pathlib import Path

import pytest
from tests.bamboo_sorting.paths import REPO_ROOT


def load_entry(monkeypatch):
    directory = REPO_ROOT / 'examples/phone_to_auboi10'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('release_gripper_test', directory / 'release_joint_gripper.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_never_connects(monkeypatch):
    module = load_entry(monkeypatch)
    def forbidden():
        raise AssertionError('hardware accessed')
    monkeypatch.setattr(module, 'ReadOnlyStation', forbidden)
    assert module.main([]) == 0


@pytest.mark.parametrize('initial', ['unset', 'closed', 'open', 'unsafe', 'both_high', 'write_failure', 'bad_readback'])
def test_release_is_bounded_and_disconnects(monkeypatch, initial):
    module = load_entry(monkeypatch)
    calls = []
    released = [False]
    io = object()

    class Station:
        io_readback = io

        def connect(self, *, include_cameras):
            assert include_cameras is False
            calls.append('connect')

        def current(self, *, require_stationary):
            assert require_stationary is True
            if initial == 'unsafe':
                raise RuntimeError('unsafe')
            if initial == 'both_high':
                raise ValueError('unknown suction output pair: (True, True)')
            if not released[0] and initial in ('unset', 'write_failure', 'bad_readback'):
                raise ValueError('unknown suction output pair: (False, False)')
            value = 0 if initial == 'open' or (released[0] and initial != 'bad_readback') else 100
            return [0] * 6 + [value], [0] * 3, 0

        def close(self):
            calls.append('close')

    def release(actual_io):
        assert actual_io is io
        calls.append('release')
        if initial == 'write_failure':
            raise RuntimeError('write failed')
        released[0] = True

    monkeypatch.setattr(module, 'ReadOnlyStation', Station)
    monkeypatch.setattr(module, 'release_once', release)
    if initial in ('unsafe', 'both_high', 'write_failure', 'bad_readback'):
        with pytest.raises((RuntimeError, ValueError)):
            module.main(['--execute'])
    else:
        assert module.main(['--execute']) == 0
    assert calls == (['connect', 'close'] if initial in ('open', 'unsafe', 'both_high')
                     else ['connect', 'release', 'close'])
