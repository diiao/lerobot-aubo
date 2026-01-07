from dataclasses import dataclass
from typing import Any, Dict

from lerobot.robots import Robot, RobotConfig
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots.config import RobotConfig

from ..config_aubo_i10 import AuboI10Config

class AuboI10Robot(Robot):
    config_class = AuboI10Config
    name = "aubo_i10"

    def __init__(self, config: AuboI10Config):
        super().__init__(config)
        self.config = config

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        pass

    @property
    def action_features(self) -> dict[str, type]:
        pass

    @property
    def is_connected(self) -> bool:
        pass

    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        pass

    def send_action(self, action: dict[str, Any]) -> [str, Any]:
        pass

    def disconnect(self) -> None:
        pass
    