from lerobot.bamboo_sorting.gripper import CommandedGripperState, DigitalOutputPneumaticGripper
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot


class FakeIo:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def setStandardDigitalOutput(self, pin: int, value: bool) -> None:
        self.calls.append((pin, value))


def test_aubo_compatibility_aliases_keep_zero_open_and_hundred_closed():
    io = FakeIo()
    robot = object.__new__(AuboI10Robot)
    robot._gripper_closed_commanded = False
    robot.pneumatic_gripper = DigitalOutputPneumaticGripper(io)

    robot._control_softpaws_based_on_gripper(100.0)
    assert robot.is_gripper_closed
    assert robot.is_suction_on  # Historical alias mirrors the same commanded state.
    assert robot.gripper_commanded_state == CommandedGripperState.CLOSED

    robot._control_suction_based_on_gripper(0.0)
    assert not robot.is_gripper_closed
    assert not robot.is_suction_on
    assert robot.gripper_commanded_state == CommandedGripperState.OPEN
