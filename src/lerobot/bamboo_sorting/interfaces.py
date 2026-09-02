"""Protocols separating task logic from physical hardware and ROS 2."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .gripper import CommandedGripperState
from .types import GraspCandidate, GraspOutcome, SceneFrame


@runtime_checkable
class SceneSensor(Protocol):
    def capture(self) -> SceneFrame:
        """Return one synchronized scene without initiating robot motion."""


@runtime_checkable
class PneumaticGripper(Protocol):
    @property
    def commanded_state(self) -> CommandedGripperState:
        """Return the last commanded state, never a measured grasp result."""

    def open(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class MotionValidator(Protocol):
    def rejection_reasons(self, candidate: GraspCandidate) -> tuple[str, ...]:
        """Return hard reachability/collision rejection reasons."""


@runtime_checkable
class MotionExecutor(Protocol):
    """Real implementations must enforce speed, workspace and collision limits."""

    def move_pregrasp(self, candidate: GraspCandidate) -> None: ...

    def approach(self, candidate: GraspCandidate) -> None: ...

    def lift(self, candidate: GraspCandidate) -> None: ...

    def transfer_to_collection(self, candidate: GraspCandidate) -> None: ...

    def retreat(self, candidate: GraspCandidate) -> None: ...

    def recover(self, candidate: GraspCandidate, outcome: GraspOutcome) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class GraspVerifier(Protocol):
    def verify(self, candidate: GraspCandidate) -> GraspOutcome:
        """Use visual/sensor evidence; commanded gripper state is insufficient."""
