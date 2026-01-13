# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
import numpy as np

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
    aubo_i10,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 100
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in teleop.action_features)
    start = time.perf_counter()
    
    from typing import Dict, Tuple

    def lerp_map(m: float, m1: float, m2: float, s1: float, s2: float, clamp: bool = True) -> float:
        """
        根据两点 (m1->s1), (m2->s2) 做线性映射。
        clamp=True 时会把 m 限制在 [min(m1,m2), max(m1,m2)] 之间。
        """
        if clamp:
            lo, hi = (m1, m2) if m1 < m2 else (m2, m1)
            if m < lo: m = lo
            if m > hi: m = hi

        k = (s2 - s1) / (m2 - m1)
        return s1 + (m - m1) * k


    # 你给的 5 行对应关系：每行是 (m1, m2, s1, s2)
    # 左侧为主臂(lero)，右侧为从臂(aubo)
    MAPPINGS: Dict[str, Tuple[float, float, float, float]] = {
        "shoulder_pan.pos":  (-100,  100,   91.93,  -97.69),
        "shoulder_lift.pos": ( 100, -100,  -59.73,   73.75),
        "elbow_flex.pos":    ( 100, -100,  157.50,  -62.81),
        "wrist_flex.pos":    ( 100, -100,  -82.69,  141.77),
        "wrist_roll.pos":    (-100,  100, -164.00,   42.17),
    }
    # MAPPINGS: Dict[str, Tuple[float, float, float, float]] = {
    #     "shoulder_pan.pos":  (-60,  60,   91.93,  -97.69),
    #     "shoulder_lift.pos": ( 60, -60,  -59.73,   73.75),
    #     "elbow_flex.pos":    ( 60, -60,  157.50,  -62.81),
    #     "wrist_flex.pos":    ( 60, -60,  -82.69,  141.77),
    #     "wrist_roll.pos":    (-60,  60, -164.00,   42.17),
    # }


    def master_to_slave(master: Dict[str, float]) -> Dict[str, float]:
        """
        输入：主臂关节值（-100~100）
        输出：从臂关节值（aubo角度/位置，单位按你表里的数值）
        """
        slave: Dict[str, float] = {}
        for joint, (m1, m2, s1, s2) in MAPPINGS.items():
            if joint not in master:
                raise KeyError(f"Missing master joint: {joint}")
            slave[joint] = lerp_map(master[joint], m1, m2, s1, s2, clamp=True)
        return slave
    

    # 工具函数：将rpy角度转换为旋转矩阵
    def rpy_to_rotation(roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x

    # 工具函数：创建齐次变换矩阵
    def create_transform(xyz, rpy):
        roll, pitch, yaw = rpy
        R = rpy_to_rotation(roll, pitch, yaw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T

    # 正向运动学函数
    def forward_kinematics(joint_angles_deg):
        """
        根据关节角度计算末端位置和姿态
        joint_angles: 长度为6的列表，包含关节1-6的角度值
        返回：末端的位置(xyz)和姿态(四元数或rpy)
        """

        joint_angles = np.deg2rad(joint_angles_deg)
        # 从URDF中提取的关节参数
        joint_params = [
            # joint 1: base -> shoulder
            {"origin_xyz": [0.0207909, -0.0230745, 0.0948817],
            "origin_rpy": [-3.14159, 6.03684e-16, 1.5708],
            "axis": [0, 0, 1]},
            
            # joint 2: shoulder -> upper_arm
            {"origin_xyz": [-0.0303992, -0.0182778, -0.0542],
            "origin_rpy": [-1.5708, -1.5708, 0],
            "axis": [0, 0, 1]},
            
            # joint 3: upper_arm -> lower_arm
            {"origin_xyz": [-0.11257, -0.028, 2.46331e-16],
            "origin_rpy": [-1.22818e-15, 5.75928e-16, 1.5708],
            "axis": [0, 0, 1]},
            
            # joint 4: lower_arm -> wrist
            {"origin_xyz": [-0.1349, 0.0052, 1.65232e-16],
            "origin_rpy": [3.2474e-15, 2.86219e-15, -1.5708],
            "axis": [0, 0, 1]},
            
            # joint 5: wrist -> gripper
            {"origin_xyz": [0, -0.0611, 0.0181],
            "origin_rpy": [1.5708, 1.5708, 3.14159],
            "axis": [0, 0, 1]},
            
            # joint 6: gripper -> jaw
            {"origin_xyz": [0.0202, 0.0188, -0.0234],
            "origin_rpy": [1.5708, -5.14108e-17, -1.38655e-14],
            "axis": [0, 0, 1]}
        ]
        
        # 初始化总变换矩阵为单位矩阵
        T_total = np.eye(4)
        
        # 计算每个关节的变换并连乘
        for i in range(6):
            params = joint_params[i]
            angle = joint_angles[i]
            
            # 关节固定变换（来自origin）
            T_origin = create_transform(params["origin_xyz"], params["origin_rpy"])
            
            # 关节旋转变换
            R_joint = rpy_to_rotation(0, 0, angle)  # 绕z轴旋转
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_joint
            
            # 合并变换
            T_total = T_total @ T_origin @ T_joint
        
        # 提取末端位置
        position = T_total[:3, 3]
        
        # 提取末端姿态（旋转矩阵）
        orientation_matrix = T_total[:3, :3]
        
        return position, orientation_matrix
    

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default

        obs = robot.get_observation()
        obs = {"test":111}

        # Get teleop action
        raw_action = teleop.get_action()
        print(f"\n\nraw_action: {raw_action}")
        position, orientation = forward_kinematics(list(raw_action.values()))
        position = position.tolist()
        print("末端位置：", position)
        print(type(position))
        print("末端姿态：")
        print(orientation)

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))
        # print(f"\n\nteleop_action: {teleop_action}")

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        # print(f"\n\nrobot_action_to_send: {robot_action_to_send}")

        robot_action_to_send.pop("gripper.pos",None)

        # slave_data = master_to_slave(robot_action_to_send)
        # print(f"\n\nslave_data: {slave_data}")

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        position = [i*3 for i in position]
        position[2] = position[2] - 0.3
        print(f"send_position: {position}")
        _ = robot.send_action(position)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
