"""Binary pneumatic gripper control with mutually exclusive digital outputs."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


GRIPPER_OPEN_VALUE = 0.0
GRIPPER_CLOSED_VALUE = 100.0


class CommandedGripperState(str, Enum):
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSED = "closed"


class StandardDigitalOutput(Protocol):
    def setStandardDigitalOutput(self, pin: int, value: bool) -> object: ...


class DigitalOutputPneumaticGripper:
    """Drive the validated two-output AUBO pneumatic gripper wiring.

    The class tracks commands only.  It does not claim that the fingers moved or
    that an object was grasped.  The opposite coil is always de-energized before
    the requested coil is energized, so both outputs are never intentionally on.
    """

    def __init__(self, io: StandardDigitalOutput, *, close_pin: int = 2, open_pin: int = 3) -> None:
        if close_pin == open_pin:
            raise ValueError("close_pin and open_pin must be different")
        if close_pin < 0 or open_pin < 0:
            raise ValueError("digital output pins must be non-negative")
        self._io = io
        self.close_pin = close_pin
        self.open_pin = open_pin
        self._commanded_state = CommandedGripperState.UNKNOWN

    @property
    def commanded_state(self) -> CommandedGripperState:
        return self._commanded_state

    @property
    def commanded_position(self) -> float | None:
        if self._commanded_state == CommandedGripperState.OPEN:
            return GRIPPER_OPEN_VALUE
        if self._commanded_state == CommandedGripperState.CLOSED:
            return GRIPPER_CLOSED_VALUE
        return None

    def open(self) -> None:
        self._drive(CommandedGripperState.OPEN)

    def close(self) -> None:
        self._drive(CommandedGripperState.CLOSED)

    def command_position(
        self, value: float, *, open_threshold: float = 20.0, close_threshold: float = 60.0
    ) -> None:
        if value > close_threshold:
            self.close()
        elif value < open_threshold:
            self.open()

    def deenergize(self) -> None:
        """Turn both coils off without claiming a physical finger state."""
        self._io.setStandardDigitalOutput(self.close_pin, False)
        self._io.setStandardDigitalOutput(self.open_pin, False)
        self._commanded_state = CommandedGripperState.UNKNOWN

    def _drive(self, target: CommandedGripperState) -> None:
        if target == self._commanded_state:
            return
        requested_pin = self.open_pin if target == CommandedGripperState.OPEN else self.close_pin
        opposite_pin = self.close_pin if target == CommandedGripperState.OPEN else self.open_pin
        try:
            self._io.setStandardDigitalOutput(opposite_pin, False)
            self._io.setStandardDigitalOutput(requested_pin, True)
        except Exception:
            self._commanded_state = CommandedGripperState.UNKNOWN
            try:
                self._io.setStandardDigitalOutput(requested_pin, False)
            except Exception:
                pass
            raise
        self._commanded_state = target
