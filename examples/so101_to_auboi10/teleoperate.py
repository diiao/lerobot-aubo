from lerobot.robots.aubo_i10 import AuboI10Config, AuboI10Robot

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline, RobotAction, RobotObservation
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
)
from lerobot.robots.aubo_i10.robot_processor import SO101EEToAuboi10EE
from lerobot.processor.converters import (
    robot_action_to_transition,
    transition_to_robot_action,
    robot_action_observation_to_transition
)
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

import time
import logging



FPS = 30

def main():
    init_logging()
    leader_config = SO101LeaderConfig(
        port="/dev/ttyACM0", id="so101_leader", use_degrees=True
    )
    follower_config = AuboI10Config(
        id="aubo_i10"
    )

    leader = SO101Leader(leader_config)
    follower = AuboI10Robot(follower_config)

    leader_kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys())
    )

    leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            )
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    leader_ee_to_follower_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            SO101EEToAuboi10EE()
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    leader.connect()
    follower.connect()

    # Init rerun viewer
    init_rerun(session_name="so101_auboi10_EE_teleop")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        robot_obs = follower.get_observation()

        # Get teleop observation
        leader_joints_obs = leader.get_action()
        logging.info(f"leader joints obs: {leader_joints_obs}")

        # teleop joints -> teleop EE action
        leader_ee_act = leader_to_ee(leader_joints_obs)
        logging.debug(f"leader EE act: {leader_ee_act}")

        # teleop EE -> follower EE
        follower_ee_act = leader_ee_to_follower_ee(leader_ee_act)
        logging.debug(f"follower EE act: {follower_ee_act}")

        # Send action to robot
        _ = follower.send_action(follower_ee_act)

        # Visualize
        log_rerun_data(observation=leader_ee_act, action=follower_ee_act)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
        



