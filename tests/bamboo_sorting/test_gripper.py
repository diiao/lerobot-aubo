import pytest

from lerobot.bamboo_sorting.gripper import CommandedGripperState, DigitalOutputPneumaticGripper


class FakeIo:
    def __init__(self, fail_on: tuple[int, bool] | None = None) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.fail_on = fail_on

    def setStandardDigitalOutput(self, pin: int, value: bool) -> None:
        self.calls.append((pin, value))
        if self.fail_on == (pin, value):
            raise RuntimeError("simulated I/O failure")


def test_open_close_mapping_is_mutually_exclusive_and_idempotent():
    io = FakeIo()
    gripper = DigitalOutputPneumaticGripper(io, close_pin=2, open_pin=3)

    gripper.close()
    gripper.close()
    assert io.calls == [(3, False), (2, True)]
    assert gripper.commanded_state == CommandedGripperState.CLOSED
    assert gripper.commanded_position == 100.0

    gripper.open()
    assert io.calls[-2:] == [(2, False), (3, True)]
    assert gripper.commanded_state == CommandedGripperState.OPEN
    assert gripper.commanded_position == 0.0


def test_deadband_does_not_change_outputs():
    io = FakeIo()
    gripper = DigitalOutputPneumaticGripper(io)
    gripper.command_position(50.0)
    assert io.calls == []
    assert gripper.commanded_state == CommandedGripperState.UNKNOWN


def test_io_exception_fails_to_unknown_without_energizing_opposite_direction():
    io = FakeIo(fail_on=(2, True))
    gripper = DigitalOutputPneumaticGripper(io)

    with pytest.raises(RuntimeError, match="simulated"):
        gripper.close()

    assert gripper.commanded_state == CommandedGripperState.UNKNOWN
    assert (3, True) not in io.calls
    assert io.calls == [(3, False), (2, True), (2, False)]


def test_deenergize_turns_off_only_the_two_configured_pins():
    io = FakeIo()
    gripper = DigitalOutputPneumaticGripper(io, close_pin=2, open_pin=3)
    gripper.deenergize()
    assert io.calls == [(2, False), (3, False)]
