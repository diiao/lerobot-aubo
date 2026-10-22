"""Shared record-loop fakes used by observer and AUBO recording tests."""

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig


def no_transition_trace(on: bool) -> dict:
    return {
        "requested_transition": False,
        "target_suction_on": on,
        "requested_do": None,
        "do_writes": [],
        "do_write_attempt_count": 0,
        "do_api_success": None,
        "do_readback_supported": True,
        "do_readback": None,
        "do_readback_after_failure": None,
        "do_readback_error": None,
        "controller_output_state_known": False,
        "controller_output_matches_requested": None,
        "partial_write_possible": False,
        "commanded_state_before": on,
        "commanded_state_after": on,
        "error": None,
    }


class FakeTeleop(Teleoperator):
    config_class = TeleoperatorConfig
    name = "fake_teleop"

    def __init__(self, actions: list[dict]):
        # Deliberately skip Teleoperator.__init__ (no calibration dirs on disk).
        self._actions = iter(actions)

    @property
    def action_features(self) -> dict:
        return {"ee.gripper_pos": float}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self, calibrate: bool = True) -> None:
        raise NotImplementedError

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
        pass
