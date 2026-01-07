from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("aubo_i10")
@dataclass
class AuboI10Config(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = False

    