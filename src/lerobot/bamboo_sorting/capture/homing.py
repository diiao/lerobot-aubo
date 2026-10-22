"""Normal capture homing without the legacy record.py global mode.

The recovery-pose workflow in record.py stays separate; this helper preserves
its normal-mode speed, steps, gripper release and return semantics.
"""

import logging
import math
import time

from ..start_poses import NORMAL_START_DEG

NORMAL_MOVE_SPEED_FRACTION = 0.5


def wait_for_move_completion(motion, start_timeout_s: float = 5.0) -> bool:
    """Wait until moveJoint starts and finishes; do not continue if it never starts."""
    deadline = time.monotonic() + start_timeout_s
    while motion.getExecId() == -1:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    while motion.getExecId() != -1:
        time.sleep(0.05)
    return True


def move_joint_and_wait(motion, target_deg, speed_fraction, speed_deg_s, accel_deg_s2) -> bool:
    """Move to one joint target at explicit limits; success requires arrival."""
    target_rad = [math.radians(value) for value in target_deg]
    motion.setSpeedFraction(speed_fraction)
    motion.moveJoint(
        target_rad,
        math.radians(speed_deg_s),
        math.radians(accel_deg_s2),
        0,
        0,
    )
    return wait_for_move_completion(motion)


def return_to_normal_start(robot):
    """Normal-mode homing: moveJoint to NORMAL_START_DEG, then release gripper."""
    logging.info("归位：moveJoint 到标准起始位...")
    motion = robot.robot_interface.getMotionControl()
    # moveJoint requires non-servo mode.
    if motion.isServoModeEnabled():
        motion.setServoMode(False)
        time.sleep(0.5)

    if not move_joint_and_wait(
        motion,
        NORMAL_START_DEG,
        NORMAL_MOVE_SPEED_FRACTION,
        speed_deg_s=80,
        accel_deg_s2=60,
    ):
        logging.error("未确认机械臂已到达标准起始位；本次归位失败")
        return False

    try:
        if not robot.suction_release():
            logging.error("归位完成，但吸盘释放失败")
            return False
        logging.info("归位完成，吸盘已释放")
    except Exception as e:
        logging.error("吸盘释放失败: %s", e)
        return False
    return True
