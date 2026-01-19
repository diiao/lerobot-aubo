from dataclasses import dataclass
from lerobot.processor import (
    ProcessorStep,
    EnvTransition,
    RobotActionProcessorStep,
    RobotAction
)

@ProcessorStepRegistry.register(name="so101_ee_to_auboi10_ee")
@dataclass
class SO101EEToAuboi10EE(RobotActionProcessorStep):
    def action(self, action: RobotAction) -> RobotAction:
        scale = 3
        new_action = action.copy()
        new_action["ee.x"] = action["ee.x"] * scale
        new_action["ee.y"] = action["ee.y"] * scale
        new_action["ee.z"] = action["ee.z"] * scale - 0.3
        new_action["ee.wx"] = action["ee.wx"]
        new_action["ee.wy"] = action["ee.wy"]
        new_action["ee.wz"] = action["ee.wz"]
        new_action["ee.gripper_pos"] = action["ee.gripper_pos"]
        return new_action
            
