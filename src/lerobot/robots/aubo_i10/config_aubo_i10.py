from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("aubo_i10")
@dataclass
class AuboI10Config(RobotConfig):
    pass

    