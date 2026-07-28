from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("aubo_i10")
@dataclass
class AuboI10Config(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    control_fps: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_fps <= 0:
            raise ValueError("control_fps must be greater than zero")
