from __future__ import annotations

import numpy as np

from lerobot.bamboo_sorting import (
    CandidateStatus,
    CommandedGripperState,
    GraspCandidate,
    GraspOutcome,
    GraspOutcomeStatus,
    PlanningResult,
    PointCloudQualityReport,
    PreprocessedScene,
    SceneFrame,
    SortingState,
    SortingTaskRunner,
)


def _candidate() -> GraspCandidate:
    return GraspCandidate(
        "candidate-0",
        "segment-0",
        np.eye(4),
        np.eye(4),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        CandidateStatus.EXECUTION_VALIDATED,
        {"test": 1.0},
        {"test": True},
    )


class Sensor:
    def __init__(self) -> None:
        self.count = 0

    def capture(self) -> SceneFrame:
        self.count += 1
        return SceneFrame(np.empty((0, 3)), float(self.count), "base")


class Pipeline:
    def __init__(self, candidate: GraspCandidate | None) -> None:
        self.candidate = candidate

    def plan(self, frame: SceneFrame) -> PlanningResult:
        quality = PointCloudQualityReport(0, 0, 0, 0, 0.0, 0.0)
        scene = PreprocessedScene(frame, frame.points_xyz, None, "base", quality, np.zeros(3), np.ones(3))
        candidates = () if self.candidate is None else (self.candidate,)
        return PlanningResult(scene, (), (), candidates)


class Gripper:
    def __init__(self) -> None:
        self.commanded_state = CommandedGripperState.OPEN
        self.calls: list[str] = []

    def close(self) -> None:
        self.calls.append("close")
        self.commanded_state = CommandedGripperState.CLOSED

    def open(self) -> None:
        self.calls.append("open")
        self.commanded_state = CommandedGripperState.OPEN


class Executor:
    def __init__(self, fail_at: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError("synthetic motion error")

    def move_pregrasp(self, candidate) -> None:
        self._call("pregrasp")

    def approach(self, candidate) -> None:
        self._call("approach")

    def lift(self, candidate) -> None:
        self._call("lift")

    def transfer_to_collection(self, candidate) -> None:
        self._call("transfer")

    def retreat(self, candidate) -> None:
        self._call("retreat")

    def recover(self, candidate, outcome) -> None:
        self._call("recover")

    def stop(self) -> None:
        self.calls.append("stop")


class Verifier:
    def __init__(self, status: GraspOutcomeStatus) -> None:
        self.status = status

    def verify(self, candidate) -> GraspOutcome:
        return GraspOutcome(self.status, "synthetic_visual_result", {"visual": True})


def test_success_follows_fixed_sequence_and_recaptures():
    sensor, gripper, executor = Sensor(), Gripper(), Executor()
    runner = SortingTaskRunner(
        sensor, Pipeline(_candidate()), gripper, executor, Verifier(GraspOutcomeStatus.SUCCESS)
    )
    result = runner.run_cycle()
    assert result.states == tuple(state for state in SortingState if state != SortingState.ABORTED)
    assert executor.calls == ["pregrasp", "approach", "lift", "transfer", "retreat"]
    assert gripper.calls == ["close", "open"]
    assert sensor.count == 2
    assert result.post_action_frame is not None


def test_uncertain_does_not_release_or_transfer_and_uses_recovery():
    sensor, gripper, executor = Sensor(), Gripper(), Executor()
    result = SortingTaskRunner(
        sensor, Pipeline(_candidate()), gripper, executor, Verifier(GraspOutcomeStatus.UNCERTAIN)
    ).run_cycle()
    assert result.outcome.status == GraspOutcomeStatus.UNCERTAIN
    assert executor.calls == ["pregrasp", "approach", "lift", "recover"]
    assert gripper.calls == ["close"]
    assert SortingState.TRANSFER not in result.states
    assert sensor.count == 2


def test_motion_exception_stops_and_is_not_reported_as_success():
    sensor, gripper, executor = Sensor(), Gripper(), Executor(fail_at="approach")
    result = SortingTaskRunner(
        sensor, Pipeline(_candidate()), gripper, executor, Verifier(GraspOutcomeStatus.SUCCESS)
    ).run_cycle()
    assert result.outcome.status == GraspOutcomeStatus.UNCERTAIN
    assert result.outcome.safety_stop
    assert result.states[-1] == SortingState.ABORTED
    assert executor.calls[-1] == "stop"
    assert gripper.calls == []


def test_no_valid_candidate_causes_no_motion_and_recaptures():
    sensor, gripper, executor = Sensor(), Gripper(), Executor()
    result = SortingTaskRunner(
        sensor, Pipeline(None), gripper, executor, Verifier(GraspOutcomeStatus.SUCCESS)
    ).run_cycle()
    assert result.outcome.status == GraspOutcomeStatus.FAILURE
    assert executor.calls == []
    assert gripper.calls == []
    assert sensor.count == 2
