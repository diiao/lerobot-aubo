"""Fail-closed deterministic sorting-cycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .gripper import CommandedGripperState
from .interfaces import GraspVerifier, MotionExecutor, PneumaticGripper, SceneSensor
from .pipeline import GeometricPlanningPipeline, PlanningResult
from .types import CandidateStatus, GraspCandidate, GraspOutcome, GraspOutcomeStatus, SceneFrame


class SortingState(str, Enum):
    PERCEIVE = "perceive"
    PLAN = "plan"
    PRE_GRASP = "pre_grasp"
    APPROACH = "approach"
    CLOSE = "close"
    LIFT = "lift"
    VERIFY = "verify"
    TRANSFER = "transfer"
    RELEASE = "release"
    RETREAT = "retreat"
    ABORTED = "aborted"


@dataclass(frozen=True)
class SortingCycleResult:
    outcome: GraspOutcome
    states: tuple[SortingState, ...]
    planning: PlanningResult | None
    candidate: GraspCandidate | None
    post_action_frame: SceneFrame | None = None


class SortingTaskRunner:
    """Run one cycle. No implementation in this package can connect hardware."""

    def __init__(
        self,
        sensor: SceneSensor,
        pipeline: GeometricPlanningPipeline,
        gripper: PneumaticGripper,
        executor: MotionExecutor,
        verifier: GraspVerifier,
    ) -> None:
        self.sensor = sensor
        self.pipeline = pipeline
        self.gripper = gripper
        self.executor = executor
        self.verifier = verifier

    def run_cycle(self) -> SortingCycleResult:
        states: list[SortingState] = []
        planning: PlanningResult | None = None
        candidate: GraspCandidate | None = None
        attempted_motion = False
        try:
            states.append(SortingState.PERCEIVE)
            frame = self.sensor.capture()
            states.append(SortingState.PLAN)
            planning = self.pipeline.plan(frame)
            candidate = next(
                (
                    item
                    for item in planning.candidates
                    if item.status == CandidateStatus.EXECUTION_VALIDATED
                ),
                None,
            )
            if candidate is None:
                refreshed = self.sensor.capture()
                return SortingCycleResult(
                    GraspOutcome(GraspOutcomeStatus.FAILURE, "no_execution_validated_candidate"),
                    tuple(states),
                    planning,
                    None,
                    refreshed,
                )

            attempted_motion = True
            states.append(SortingState.PRE_GRASP)
            self.executor.move_pregrasp(candidate)
            states.append(SortingState.APPROACH)
            self.executor.approach(candidate)
            states.append(SortingState.CLOSE)
            self.gripper.close()
            states.append(SortingState.LIFT)
            self.executor.lift(candidate)
            states.append(SortingState.VERIFY)
            outcome = self.verifier.verify(candidate)

            if outcome.status == GraspOutcomeStatus.SUCCESS:
                states.append(SortingState.TRANSFER)
                self.executor.transfer_to_collection(candidate)
                states.append(SortingState.RELEASE)
                self.gripper.open()
                states.append(SortingState.RETREAT)
                self.executor.retreat(candidate)
            else:
                states.append(SortingState.RETREAT)
                self.executor.recover(candidate, outcome)

            # A commanded state is deliberately not used to upgrade the outcome.
            _ = self.gripper.commanded_state == CommandedGripperState.CLOSED
            refreshed = self.sensor.capture()
            return SortingCycleResult(outcome, tuple(states), planning, candidate, refreshed)
        except Exception as exc:
            states.append(SortingState.ABORTED)
            stop_error = None
            try:
                self.executor.stop()
            except Exception as stop_exc:
                stop_error = f"{type(stop_exc).__name__}:{stop_exc}"
            refreshed = None
            if attempted_motion:
                try:
                    refreshed = self.sensor.capture()
                except Exception:
                    pass
            return SortingCycleResult(
                GraspOutcome(
                    GraspOutcomeStatus.UNCERTAIN,
                    f"unhandled_exception:{type(exc).__name__}",
                    evidence={"message": str(exc), "stop_error": stop_error},
                    safety_stop=True,
                ),
                tuple(states),
                planning,
                candidate,
                refreshed,
            )
