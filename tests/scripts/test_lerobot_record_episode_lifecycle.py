# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline integration tests for the generic record episode lifecycle hook.

Everything here is mocked: robots, teleoperators, datasets, and lifecycle
observers are in-memory fakes. No camera, AUBO, DO/IO, motion, real dataset on
disk, network, training, or inference is involved.

Coverage:
- ``lifecycle=None`` keeps the original save/finalize/rerecord/stop behavior
  exactly (regression proof for the record() extraction).
- A frame observer is passed to the real record_loop per episode attempt and
  never to the reset loop (which runs with ``dataset=None``).
- rerecord/stop/save/finalize notifications and their deterministic exception
  semantics (observer errors never mask in-flight original errors).
- The record() entrypoint itself, with all hardware/dataset factories
  monkeypatched, keeps the original call counts with no lifecycle observer.
"""

import logging
from types import SimpleNamespace

import pytest

import lerobot.scripts.lerobot_record as lerobot_record
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    make_default_teleop_action_processor,
)
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    RecordConfig,
    finalize_recorded_dataset,
    record,
    record_episode_sessions,
)
from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

FPS = 30.0
SINGLE_TASK = "sort bamboo"


def _gripper_action(value: float) -> dict:
    return {"ee.gripper_pos": value}


class FakeRobot:
    """Duck-typed robot with scripted per-cycle traces and event triggers."""

    name = "fake_robot"

    def __init__(
        self,
        *,
        traces=(),
        exit_afters=(),
        rerecord_afters=(),
        stop_afters=(),
        send_error: BaseException | None = None,
        error_after: int | None = None,
        reset_error: BaseException | None = None,
    ):
        self.traces = list(traces)
        self.exit_afters = set(exit_afters)
        self.rerecord_afters = set(rerecord_afters)
        self.stop_afters = set(stop_afters)
        self.send_error = send_error
        self.error_after = error_after
        self.reset_error = reset_error
        self.reset_calls = 0
        self.sent_actions: list[dict] = []
        self.last_gripper_command_trace: dict | None = None
        self.events: dict | None = None

    def reset(self) -> None:
        self.reset_calls += 1
        if self.reset_error is not None:
            raise self.reset_error

    def get_observation(self) -> dict:
        return {"a": 0.0, "b": 0.0}

    def send_action(self, action: dict) -> dict:
        self.sent_actions.append(action)
        index = len(self.sent_actions)
        self.last_gripper_command_trace = self.traces[index - 1] if index <= len(self.traces) else None
        if self.send_error is not None and self.error_after is not None and index >= self.error_after:
            raise self.send_error
        events = self.events
        if events is not None:
            if index in self.exit_afters:
                events["exit_early"] = True
            if index in self.rerecord_afters:
                events["rerecord_episode"] = True
                events["exit_early"] = True
            if index in self.stop_afters:
                events["stop_recording"] = True
                events["exit_early"] = True
        return action


class FakeTeleop(Teleoperator):
    config_class = TeleoperatorConfig
    name = "fake_teleop"

    def __init__(self, actions: list[dict]):
        # Deliberately skip Teleoperator.__init__ (no calibration dirs on disk).
        self._actions = iter(actions)
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {"ee.gripper_pos": float}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        raise NotImplementedError

    def configure(self) -> None:
        raise NotImplementedError

    def get_action(self) -> dict:
        return next(self._actions)

    def send_feedback(self, feedback: dict) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        self._connected = False


class FakeDataset:
    """Duck-typed LeRobotDataset stand-in counting every lifecycle call."""

    def __init__(
        self,
        *,
        fps: float = FPS,
        save_error: Exception | None = None,
        finalize_error: Exception | None = None,
        add_frame_error: Exception | None = None,
    ):
        self.fps = fps
        self.features = {
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action.ee.gripper_pos": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["ee.gripper_pos"],
            },
        }
        self.episode_buffer: dict | None = None
        self.save_calls = 0
        self.clear_calls = 0
        self.finalize_calls = 0
        self.meta_total_episodes = 0
        self.save_error = save_error
        self.finalize_error = finalize_error
        self.add_frame_error = add_frame_error
        self.frames: list[dict] = []

    @property
    def num_episodes(self) -> int:
        return self.meta_total_episodes

    def add_frame(self, frame: dict) -> None:
        if self.episode_buffer is None:
            self.episode_buffer = {"size": 0, "episode_index": self.meta_total_episodes}
        if self.add_frame_error is not None:
            raise self.add_frame_error
        self.episode_buffer["size"] += 1
        self.frames.append(frame)

    def clear_episode_buffer(self) -> None:
        self.clear_calls += 1
        self.episode_buffer = None

    def save_episode(self) -> None:
        self.save_calls += 1
        if self.save_error is not None:
            raise self.save_error
        self.episode_buffer = None
        self.meta_total_episodes += 1

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self.finalize_error is not None:
            raise self.finalize_error


class CountingObserver:
    """Generic frame observer counting hook invocations per stage."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def _record(self, stage: str, candidate_frame_index: int) -> None:
        self.calls.append((stage, candidate_frame_index))

    def before_send(self, **kwargs) -> None:
        self._record("before_send", kwargs["candidate_frame_index"])

    def on_send_success(self, **kwargs) -> None:
        self._record("on_send_success", kwargs["candidate_frame_index"])

    def on_send_failure(self, **kwargs) -> None:
        self._record("on_send_failure", kwargs["candidate_frame_index"])

    def on_add_frame_success(self, **kwargs) -> None:
        self._record("on_add_frame_success", kwargs["candidate_frame_index"])

    def on_add_frame_failure(self, **kwargs) -> None:
        self._record("on_add_frame_failure", kwargs["candidate_frame_index"])


class RecordingLifecycle:
    """Generic lifecycle observer recording every hook (optionally failing)."""

    def __init__(
        self,
        *,
        frame_observers=(),
        rerecord_error: BaseException | None = None,
        incomplete_error: BaseException | None = None,
        save_error: BaseException | None = None,
        finalize_success_error: BaseException | None = None,
        finalize_failure_error: BaseException | None = None,
    ):
        self._frame_observers = list(frame_observers)
        self.rerecord_error = rerecord_error
        self.incomplete_error = incomplete_error
        self.save_error = save_error
        self.finalize_success_error = finalize_success_error
        self.finalize_failure_error = finalize_failure_error
        self.frame_observer_requests = 0
        self.rerecord_calls: list[int] = []
        self.incomplete_calls: list[tuple[int, str]] = []
        self.save_calls: list[int] = []
        self.finalize_success_calls: list[int] = []
        self.finalize_failure_calls: list[BaseException] = []

    def episode_frame_observer(self):
        self.frame_observer_requests += 1
        if self._frame_observers:
            return self._frame_observers.pop(0)
        return None

    def on_episode_rerecord(self, *, robot, dataset, episode_index) -> None:
        self.rerecord_calls.append(episode_index)
        if self.rerecord_error is not None:
            raise self.rerecord_error

    def on_episode_incomplete(self, *, robot, dataset, episode_index, reason) -> None:
        self.incomplete_calls.append((episode_index, reason))
        if self.incomplete_error is not None:
            raise self.incomplete_error

    def save_episode(self, *, robot, dataset, episode_index) -> None:
        self.save_calls.append(episode_index)
        if self.save_error is not None:
            raise self.save_error
        dataset.save_episode()

    def on_dataset_finalize_success(self, *, dataset, recorded_episode_count) -> None:
        self.finalize_success_calls.append(recorded_episode_count)
        if self.finalize_success_error is not None:
            raise self.finalize_success_error

    def on_dataset_finalize_failure(self, *, dataset, error) -> None:
        self.finalize_failure_calls.append(error)
        if self.finalize_failure_error is not None:
            raise self.finalize_failure_error


def _run_sessions(
    *,
    robot: FakeRobot,
    teleop_actions: list[dict],
    dataset: FakeDataset,
    lifecycle=None,
    num_episodes: int = 1,
    reset_time_s: float = 0.0,
):
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    robot.events = events
    recorded = record_episode_sessions(
        robot=robot,
        teleop=FakeTeleop(teleop_actions),
        policy=None,
        preprocessor=None,
        postprocessor=None,
        teleop_action_processor=make_default_teleop_action_processor(),
        robot_action_processor=make_default_robot_action_processor(),
        robot_observation_processor=make_default_robot_observation_processor(),
        dataset=dataset,
        events=events,
        fps=FPS,
        num_episodes=num_episodes,
        episode_time_s=60,
        reset_time_s=reset_time_s,
        single_task=SINGLE_TASK,
        play_sounds=False,
        display_data=False,
        display_compressed_images=False,
        lifecycle=lifecycle,
    )
    return recorded, events


# ------------------------------------------------- lifecycle=None regression


def test_lifecycle_none_two_episodes_keeps_original_save_counts() -> None:
    robot = FakeRobot(exit_afters={2, 4})
    dataset = FakeDataset()

    recorded, _ = _run_sessions(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)] * 4,
        dataset=dataset,
        lifecycle=None,
        num_episodes=2,
    )

    assert recorded == 2
    assert dataset.save_calls == 2
    assert dataset.clear_calls == 0
    assert len(dataset.frames) == 4
    finalize_recorded_dataset(dataset, lifecycle=None, recorded_episode_count=recorded)
    assert dataset.finalize_calls == 1


def test_lifecycle_none_rerecord_keeps_original_behavior() -> None:
    robot = FakeRobot(rerecord_afters={2}, exit_afters={2, 4})
    dataset = FakeDataset()

    recorded, events = _run_sessions(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)] * 4,
        dataset=dataset,
        lifecycle=None,
        num_episodes=1,
    )

    assert recorded == 1
    assert dataset.save_calls == 1
    assert dataset.clear_calls == 1
    assert events["rerecord_episode"] is False
    assert len(dataset.frames) == 4  # 2 per attempt, buffer cleared in between


def test_lifecycle_none_stop_mid_episode_still_saves_original_behavior() -> None:
    # Historical behavior: a stopped episode's buffer is still saved once.
    robot = FakeRobot(stop_afters={4}, exit_afters={2, 4})
    dataset = FakeDataset()

    recorded, events = _run_sessions(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)] * 4,
        dataset=dataset,
        lifecycle=None,
        num_episodes=2,
    )

    assert events["stop_recording"] is True
    assert recorded == 2
    assert dataset.save_calls == 2


# ---------------------------------------------------- frame observer routing


def test_frame_observer_reaches_record_loop_but_never_reset_loop() -> None:
    observer = CountingObserver()
    lifecycle = RecordingLifecycle(frame_observers=[observer, observer])
    robot = FakeRobot(exit_afters={2, 3, 5})
    dataset = FakeDataset()

    recorded, _ = _run_sessions(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)] * 5,
        dataset=dataset,
        lifecycle=lifecycle,
        num_episodes=2,
        reset_time_s=1.0,
    )

    # Sends 1-2 episode 0, send 3 reset loop, sends 4-5 episode 1.
    assert len(robot.sent_actions) == 5
    assert recorded == 2
    assert dataset.save_calls == 2
    assert lifecycle.frame_observer_requests == 2
    stages = [stage for stage, _ in observer.calls]
    # Exactly two full cycles per episode; the reset loop ran with
    # dataset=None and therefore produced zero observer calls.
    cycle = ["before_send", "on_send_success", "on_add_frame_success"]
    assert stages == cycle + cycle + cycle + cycle
    # Candidate indices restart at 0 per episode: two cycles with candidates
    # 0 and 1 per episode; the reset-loop send produced none.
    assert [index for _, index in observer.calls] == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]


# --------------------------------------------------------- lifecycle present


def test_rerecord_notifies_once_and_clears_buffer_once() -> None:
    lifecycle = RecordingLifecycle()
    robot = FakeRobot(rerecord_afters={2}, exit_afters={2, 4})
    dataset = FakeDataset()

    recorded, _ = _run_sessions(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)] * 4,
        dataset=dataset,
        lifecycle=lifecycle,
        num_episodes=1,
    )

    assert recorded == 1
    assert lifecycle.frame_observer_requests == 2
    assert lifecycle.rerecord_calls == [0]
    assert dataset.clear_calls == 1
    assert lifecycle.save_calls == [0]
    assert dataset.save_calls == 1


def test_stop_before_save_marks_incomplete_and_never_saves() -> None:
    lifecycle = RecordingLifecycle()
    robot = FakeRobot(stop_afters={4}, exit_afters={2, 4})
    dataset = FakeDataset()

    recorded, _ = _run_sessions(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)] * 4,
        dataset=dataset,
        lifecycle=lifecycle,
        num_episodes=2,
    )

    assert recorded == 1
    assert dataset.save_calls == 1
    assert lifecycle.save_calls == [0]
    assert lifecycle.incomplete_calls == [(1, "operator stop before save")]
    finalize_recorded_dataset(dataset, lifecycle=lifecycle, recorded_episode_count=recorded)
    assert dataset.finalize_calls == 1
    assert lifecycle.finalize_success_calls == [1]


def test_record_loop_failure_notifies_incomplete_and_propagates() -> None:
    lifecycle = RecordingLifecycle()
    robot = FakeRobot(send_error=RuntimeError("motion boom"), error_after=1)
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="motion boom"):
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 2,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=1,
        )

    assert dataset.save_calls == 0
    assert lifecycle.save_calls == []
    assert len(lifecycle.incomplete_calls) == 1
    index, reason = lifecycle.incomplete_calls[0]
    assert index == 0
    assert "record_loop raised RuntimeError" in reason
    # The episode loop aborted, but the caller still finalizes the dataset in
    # its finally block and the observer is notified of the success.
    finalize_recorded_dataset(dataset, lifecycle=lifecycle, recorded_episode_count=0)
    assert dataset.finalize_calls == 1
    assert lifecycle.finalize_success_calls == [0]


def test_send_failure_skips_add_frame_and_save_without_lifecycle() -> None:
    robot = FakeRobot(send_error=RuntimeError("motion boom"), error_after=1)
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="motion boom"):
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 2,
            dataset=dataset,
            lifecycle=None,
            num_episodes=1,
        )

    assert dataset.save_calls == 0
    assert len(dataset.frames) == 0


# --------------------------------------------------- observer error semantics


def test_observer_error_during_failure_notification_never_masks_original(caplog) -> None:
    lifecycle = RecordingLifecycle(incomplete_error=RuntimeError("observer exploded"))
    robot = FakeRobot(send_error=RuntimeError("motion boom"), error_after=1)
    dataset = FakeDataset()

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="motion boom"):
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 2,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=1,
        )

    assert dataset.save_calls == 0
    assert any("on_episode_incomplete" in record.message for record in caplog.records)


def test_observer_error_during_rerecord_propagates() -> None:
    lifecycle = RecordingLifecycle(rerecord_error=RuntimeError("rerecord hook boom"))
    robot = FakeRobot(rerecord_afters={2}, exit_afters={2, 4})
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="rerecord hook boom"):
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 4,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=1,
        )

    # The buffer clear never ran because the notification aborted the loop.
    assert dataset.clear_calls == 0
    assert dataset.save_calls == 0


def test_observer_error_during_stop_incomplete_propagates_without_inflight() -> None:
    lifecycle = RecordingLifecycle(incomplete_error=RuntimeError("incomplete hook boom"))
    robot = FakeRobot(stop_afters={2})
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="incomplete hook boom"):
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 2,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=1,
        )

    assert dataset.save_calls == 0


# --------------------------------------------------------------- finalize


def test_finalize_failure_notifies_observer_and_propagates() -> None:
    lifecycle = RecordingLifecycle()
    dataset = FakeDataset(finalize_error=RuntimeError("finalize crashed"))

    with pytest.raises(RuntimeError, match="finalize crashed"):
        finalize_recorded_dataset(dataset, lifecycle=lifecycle, recorded_episode_count=0)

    assert dataset.finalize_calls == 1
    assert len(lifecycle.finalize_failure_calls) == 1
    assert isinstance(lifecycle.finalize_failure_calls[0], RuntimeError)
    assert lifecycle.finalize_success_calls == []


def test_finalize_failure_observer_error_never_masks_finalize_error() -> None:
    lifecycle = RecordingLifecycle(finalize_failure_error=RuntimeError("observer exploded"))
    dataset = FakeDataset(finalize_error=RuntimeError("finalize crashed"))

    with pytest.raises(RuntimeError, match="finalize crashed"):
        finalize_recorded_dataset(dataset, lifecycle=lifecycle, recorded_episode_count=0)

    assert len(lifecycle.finalize_failure_calls) == 1


def test_finalize_success_observer_error_propagates_without_inflight_error() -> None:
    lifecycle = RecordingLifecycle(finalize_success_error=RuntimeError("promotion boom"))
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="promotion boom"):
        finalize_recorded_dataset(dataset, lifecycle=lifecycle, recorded_episode_count=0)

    assert dataset.finalize_calls == 1


def test_finalize_success_observer_error_only_logged_with_inflight_error(caplog) -> None:
    lifecycle = RecordingLifecycle(
        save_error=RuntimeError("save failed"),
        finalize_success_error=RuntimeError("promotion boom"),
    )
    robot = FakeRobot(exit_afters={1})
    dataset = FakeDataset()

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="save failed"):
        try:
            _run_sessions(
                robot=robot,
                teleop_actions=[_gripper_action(0.0)],
                dataset=dataset,
                lifecycle=lifecycle,
                num_episodes=1,
            )
        finally:
            finalize_recorded_dataset(
                dataset, lifecycle=lifecycle, recorded_episode_count=0
            )

    # The save error kept propagating; the observer error was only logged.
    assert dataset.save_calls == 0
    assert dataset.finalize_calls == 1
    assert any("on_dataset_finalize_success" in record.message for record in caplog.records)


# ------------------------------------------------------- record() entrypoint


class _FakeLeRobotDataset(FakeDataset):
    """Fake dataset class standing in for LeRobotDataset (create + resume)."""

    created: list = []

    def __init__(self, *args, **kwargs):
        kwargs.pop("fps", None)
        super().__init__(fps=FPS)

    @classmethod
    def create(cls, *args, **kwargs):
        instance = cls()
        cls.created.append(instance)
        return instance


class _DummyVideoEncodingManager:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_record_entrypoint_without_lifecycle_keeps_original_counts(monkeypatch, tmp_path) -> None:
    """record() itself, with every hardware factory mocked, is unchanged."""
    robot = FakeRobot(exit_afters={2, 4})
    # Entrypoint fakes need the real robot surface record() touches.
    robot.action_features = {"ee.gripper_pos": float}
    robot.observation_features = {"a": float, "b": float}
    robot.cameras = []
    robot._connected = False

    def connect():
        robot._connected = True

    def disconnect():
        robot._connected = False

    robot.connect = connect
    robot.disconnect = disconnect
    robot.is_connected = lambda: robot._connected
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(
        lerobot_record,
        "make_teleoperator_from_config",
        lambda cfg: FakeTeleop([_gripper_action(0.0)] * 4),
    )
    monkeypatch.setattr(
        lerobot_record,
        "make_processors_for_teleop_robot_pair",
        lambda **kwargs: (
            make_default_teleop_action_processor(),
            make_default_robot_action_processor(),
            make_default_robot_observation_processor(),
        ),
    )
    monkeypatch.setattr(lerobot_record, "sanity_check_dataset_name", lambda *a, **k: None)
    monkeypatch.setattr(
        lerobot_record, "sanity_check_dataset_robot_compatibility", lambda *a, **k: None
    )
    monkeypatch.setattr(lerobot_record, "LeRobotDataset", _FakeLeRobotDataset)

    def fake_keyboard_listener():
        robot.events = events
        return (None, events)

    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", fake_keyboard_listener)
    monkeypatch.setattr(lerobot_record, "VideoEncodingManager", _DummyVideoEncodingManager)
    monkeypatch.setattr(lerobot_record, "log_say", lambda *a, **k: None)
    monkeypatch.setattr(lerobot_record, "init_logging", lambda *a, **k: None)
    monkeypatch.setattr(lerobot_record, "is_headless", lambda: True)

    cfg = RecordConfig(
        robot=SimpleNamespace(type="fake_robot"),
        dataset=DatasetRecordConfig(
            repo_id="fake/repo",
            single_task=SINGLE_TASK,
            root=tmp_path,
            fps=int(FPS),
            episode_time_s=60,
            reset_time_s=0,
            num_episodes=2,
            push_to_hub=False,
        ),
        teleop=SimpleNamespace(type="fake_teleop"),
        play_sounds=False,
    )

    result = record(cfg)

    assert len(_FakeLeRobotDataset.created) == 1
    dataset = _FakeLeRobotDataset.created[0]
    assert result is dataset
    assert dataset.save_calls == 2
    assert dataset.finalize_calls == 1
    assert dataset.clear_calls == 0
    assert len(dataset.frames) == 4


# ------------------------------------------------------- reset-phase failures


def test_reset_record_loop_failure_marks_episode_incomplete_and_preserves_original() -> None:
    original = RuntimeError("reset loop boom")
    lifecycle = RecordingLifecycle()
    robot = FakeRobot(send_error=original, error_after=3, exit_afters={2})
    dataset = FakeDataset()

    with pytest.raises(RuntimeError) as excinfo:
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 4,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=2,
            reset_time_s=1.0,
        )

    assert excinfo.value is original
    assert type(excinfo.value) is RuntimeError
    assert dataset.save_calls == 0
    assert lifecycle.save_calls == []
    assert len(lifecycle.incomplete_calls) == 1
    index, reason = lifecycle.incomplete_calls[0]
    assert index == 0
    assert "reset record_loop raised RuntimeError" in reason


def test_robot_reset_failure_marks_episode_incomplete_and_preserves_original() -> None:
    original = RuntimeError("reset boom")
    lifecycle = RecordingLifecycle()
    robot = FakeRobot(exit_afters={2}, reset_error=original)
    robot.name = "unitree_g1"
    dataset = FakeDataset()

    with pytest.raises(RuntimeError) as excinfo:
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 2,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=2,
            reset_time_s=0.0,
        )

    assert excinfo.value is original
    assert type(excinfo.value) is RuntimeError
    assert robot.reset_calls == 1
    assert dataset.save_calls == 0
    assert lifecycle.save_calls == []
    assert len(lifecycle.incomplete_calls) == 1
    index, reason = lifecycle.incomplete_calls[0]
    assert index == 0
    assert "robot.reset raised RuntimeError" in reason


@pytest.mark.parametrize(
    "failure",
    ["robot.reset", "reset record_loop"],
    ids=["robot_reset", "reset_record_loop"],
)
def test_reset_phase_observer_failure_never_masks_original_or_double_notifies(
    failure, caplog
) -> None:
    original = RuntimeError("reset original")
    observer_error = KeyboardInterrupt("observer interrupt")
    lifecycle = RecordingLifecycle(incomplete_error=observer_error)
    if failure == "robot.reset":
        robot = FakeRobot(exit_afters={2}, reset_error=original)
        robot.name = "unitree_g1"
        reset_time_s = 0.0
        actions = [_gripper_action(0.0)] * 2
    else:
        robot = FakeRobot(send_error=original, error_after=3, exit_afters={2})
        reset_time_s = 1.0
        actions = [_gripper_action(0.0)] * 4
    dataset = FakeDataset()

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError) as excinfo,
    ):
        _run_sessions(
            robot=robot,
            teleop_actions=actions,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=2,
            reset_time_s=reset_time_s,
        )

    assert excinfo.value is original
    assert dataset.save_calls == 0
    assert len(lifecycle.incomplete_calls) == 1
    assert any("on_episode_incomplete" in record.message for record in caplog.records)


# --------------------------------------------- BaseException observer contract


_OBSERVER_ERRORS = (
    KeyboardInterrupt("observer interrupt"),
    SystemExit("observer exit"),
    RuntimeError("observer exploded"),
)


@pytest.mark.parametrize(
    "observer_error",
    _OBSERVER_ERRORS,
    ids=["KeyboardInterrupt", "SystemExit", "RuntimeError"],
)
def test_observer_baseexception_propagates_without_original(observer_error) -> None:
    lifecycle = RecordingLifecycle(incomplete_error=observer_error)
    robot = FakeRobot(stop_afters={2})
    dataset = FakeDataset()

    with pytest.raises(type(observer_error)) as excinfo:
        _run_sessions(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)] * 2,
            dataset=dataset,
            lifecycle=lifecycle,
            num_episodes=1,
        )

    assert excinfo.value is observer_error
    assert dataset.save_calls == 0


@pytest.mark.parametrize(
    "observer_error",
    _OBSERVER_ERRORS,
    ids=["KeyboardInterrupt", "SystemExit", "RuntimeError"],
)
@pytest.mark.parametrize(
    "notification",
    ["record_failure", "finalize_failure", "finalize_success_inflight"],
    ids=["record_failure", "finalize_failure", "finalize_success_inflight"],
)
def test_observer_baseexception_never_masks_original(
    observer_error, notification, caplog
) -> None:
    original = RuntimeError("original boom")
    dataset = FakeDataset()

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as excinfo:
        if notification == "record_failure":
            lifecycle = RecordingLifecycle(incomplete_error=observer_error)
            robot = FakeRobot(send_error=original, error_after=1)
            _run_sessions(
                robot=robot,
                teleop_actions=[_gripper_action(0.0)] * 2,
                dataset=dataset,
                lifecycle=lifecycle,
                num_episodes=1,
            )
        elif notification == "finalize_failure":
            lifecycle = RecordingLifecycle(finalize_failure_error=observer_error)
            failing = FakeDataset(finalize_error=original)
            finalize_recorded_dataset(failing, lifecycle=lifecycle, recorded_episode_count=0)
        else:
            lifecycle = RecordingLifecycle(
                save_error=original, finalize_success_error=observer_error
            )
            robot = FakeRobot(exit_afters={1})
            try:
                _run_sessions(
                    robot=robot,
                    teleop_actions=[_gripper_action(0.0)],
                    dataset=dataset,
                    lifecycle=lifecycle,
                    num_episodes=1,
                )
            finally:
                finalize_recorded_dataset(
                    dataset, lifecycle=lifecycle, recorded_episode_count=0
                )

    assert excinfo.value is original
    assert type(excinfo.value) is RuntimeError
