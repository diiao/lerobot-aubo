from dataclasses import dataclass
from typing import Any, Dict

from lerobot.robots import Robot, RobotConfig
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots.config import RobotConfig

from .config_aubo_i10 import AuboI10Config
import pyaubo_sdk, time, math

class AuboI10Robot(Robot):
    config_class = AuboI10Config
    name = "aubo_i10"

    def __init__(self, config: AuboI10Config):
        super().__init__(config)
        self.config = config

        self.robot_ip = "192.168.31.200"  # 机械臂 IP 地址
        self.robot_port = 30004  # 端口号


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
        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        self.robot_rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 超时
        self.robot_rpc_client.connect(self.robot_ip, self.robot_port)  # 接口调用: 连接 RPC 服务
        if self.robot_rpc_client.hasConnected():
            print("RPC客户端连接成功!")
            self.robot_rpc_client.login("aubo", "123456")  # 接口调用: 登录
            if self.robot_rpc_client.hasLogined():
                print("RPC客户端登录成功!")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        pass

    # 阻塞
    def wait_arrival(self, robot_interface):
        max_retry_count = 5
        cnt = 0

        # 接口调用: 获取当前的运动指令 ID
        exec_id = robot_interface.getMotionControl().getExecId()

        # 等待机械臂开始运动
        while exec_id == -1:
            if cnt > max_retry_count:
                return -1
            time.sleep(0.001)
            cnt += 1
            exec_id = robot_interface.getMotionControl().getExecId()

        # 等待机械臂运动完成
        while robot_interface.getMotionControl().getExecId() != -1:
            time.sleep(0.001)

        return 0
    
    # def send_action(self, action: dict[str, Any]) -> [str, Any]:
    #     import math
    #     print(f"\n\naction: {action}")
    #     for joint, value in action.items():
    #         action[joint] = value * (math.pi / 180)
    #     q = list(action.values())
    #     q.insert(4, math.pi * 0.5)
        
    #     robot_name = self.robot_rpc_client.getRobotNames()[0]

    #     robot_interface = self.robot_rpc_client.getRobotInterface(robot_name)
    #     robot_interface.getMotionControl().setSpeedFraction(0.95)
    #     robot_interface.getMotionControl() \
    #     .moveJoint(q, 80 * (math.pi / 180), 60 * (math.pi / 180), 0, 0)

    #     ret = self.wait_arrival(robot_interface)
    #     return q
    def send_action(self, action):
        robot_name = self.robot_rpc_client.getRobotNames()[0]
        robot_interface = self.robot_rpc_client.getRobotInterface(robot_name)
        robot_interface.getMotionControl().setSpeedFraction(0.95)
        action.append(-3.134)
        action.append(0.004)
        action.append(1.567)
        robot_interface.getMotionControl().moveLine(action, 1.2, 0.25, 0, 0)
        ret = self.wait_arrival(robot_interface)
        return action
        


    def disconnect(self) -> None:
        self.robot_rpc_client.logout()  # 退出登录
        self.robot_rpc_client.disconnect()  # 断开连接
    