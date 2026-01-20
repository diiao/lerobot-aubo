from dataclasses import dataclass
from lerobot.processor import (
    ProcessorStep,
    EnvTransition,
    RobotActionProcessorStep,
    RobotAction,
    ProcessorStepRegistry
)

@ProcessorStepRegistry.register(name="so101_ee_to_auboi10_ee")
@dataclass
class SO101EEToAuboi10EE(RobotActionProcessorStep):
    def action(self, action: RobotAction) -> RobotAction:
        scale = 3
        new_action = action.copy()
        new_action["ee.x"] = action["ee.x"] * scale
        new_action["ee.y"] = action["ee.y"] * scale
        new_action["ee.z"] = action["ee.z"] * scale
        new_action["ee.wx"] = -3.14
        new_action["ee.wy"] = 0
        new_action["ee.wz"] = 0
        new_action["ee.gripper_pos"] = action["ee.gripper_pos"]
        return new_action
    
    def transform_features(self):
        pass
            
