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

        self.robot_rpc_client = pyaubo_sdk.RpcClient()


    @property
    def observation_features(self) -> dict[str, type | tuple]:
        pass

    @property
    def action_features(self) -> dict[str, type]:
        pass

    @property
    def is_connected(self) -> bool:
        return self.robot_rpc_client.hasConnected()

    def connect(self, calibrate: bool = True) -> None:
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

    def get_robot_status(self):
        robot_name = self.robot_rpc_client.getRobotNames()[0]  # 接口调用: 获取机器人的名字
        # 接口调用: 获取机器人的名字
        name = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getName()
        print("机器人的名字:", name)
        # 接口调用: 获取机器人的自由度
        dof = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getDof()
        print("机器人的自由度:", dof)
        # 接口调用: 获取机器人的伺服控制周期（从硬件抽象层读取）
        cycle_time = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getCycletime()
        print("伺服控制周期:", cycle_time)
        # 接口调用: 获取默认的工具端加速度，单位: m/s^2
        default_tool_acc = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getDefaultToolAcc()
        print("默认的工具端加速度:", default_tool_acc)
        # 接口调用: 获取默认的工具端速度，单位: m/s
        default_tool_speed = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getDefaultToolSpeed()
        print("默认的工具端速度:", default_tool_speed)
        # 接口调用: 获取默认的关节加速度，单位: rad/s
        default_joint_acc = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getDefaultJointAcc()
        print("默认的关节加速度", default_joint_acc)
        # 接口调用: 获取默认的关节速度，单位: rad/s
        default_joint_speed = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getDefaultJointSpeed()
        print("默认的关节加速度", default_joint_speed)
        # 接口调用: 获取机器人类型代码
        robot_type = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getRobotType()
        print("机器人类型代码", robot_type)
        # 接口调用: 获取机器人子类型代码
        sub_robot_type = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getRobotSubType()
        print("机器人子类型代码", sub_robot_type)
        # 接口调用: 获取控制柜类型代码
        control_box_type = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getControlBoxType()
        print("控制柜类型代码", control_box_type)
        # 接口调用: 获取安装位姿(机器人的基坐标系相对于世界坐标系)
        mounting_pose = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getMountingPose()
        print("安装位姿(机器人的基坐标系相对于世界坐标系)", mounting_pose)
        # 接口调用: 设置碰撞灵敏度等级
        level = 6
        self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().setCollisionLevel(level)
        print("设置碰撞灵敏度等级:", level)
        # 接口调用: 获取碰撞灵敏度等级
        level = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getCollisionLevel()
        print("碰撞灵敏度等级", level)
        # 接口调用: 设置碰撞停止类型
        collision_stop_type = 1
        self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().setCollisionStopType(collision_stop_type)
        print("设置碰撞停止类型", collision_stop_type)
        # 接口调用: 获取碰撞停止类型
        collision_stop_type = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getCollisionStopType()
        print("碰撞停止类型", collision_stop_type)
        # 接口调用: 获取机器人DH参数
        kin_param = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getKinematicsParam(True)
        print("机器人DH参数", kin_param)
        # 接口调用: 获取指定温度下的DH参数补偿值
        temperature = 20
        kin_compensate = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getKinematicsCompensate(
            temperature)
        print("指定温度下的DH参数补偿值", kin_compensate)
        # 接口调用: 获取可用的末端力矩传感器的名字
        tcp_force_sensor_name = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getTcpForceSensorNames()
        print("可用的末端力矩传感器的名字", tcp_force_sensor_name)
        # 接口调用: 获取末端力矩偏移
        tcp_force_offset = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getTcpForceOffset()
        print("末端力矩偏移", tcp_force_offset)
        # 接口调用: 获取可用的底座力矩传感器的名字
        base_force_sensor_names = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getBaseForceSensorNames()
        print("可用的底座力矩传感器的名字", base_force_sensor_names)
        # 接口调用: 获取底座力矩偏移
        base_force_offset = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getBaseForceOffset()
        print("底座力矩偏移", base_force_offset)
        # 接口调用: 获取安全参数校验码 CRC32
        check_sum = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getSafetyParametersCheckSum()
        print("安全参数校验码 CRC32", check_sum)
        # 接口调用: 获取关节最大位置（物理极限）
        joint_max_positions = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getJointMaxPositions()
        print("关节最大位置（物理极限）", joint_max_positions)
        # 接口调用: 获取关节最小位置（物理极限）
        joint_min_positions = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getJointMinPositions()
        print("关节最小位置（物理极限）", joint_min_positions)
        # 接口调用: 获取关节最大速度（物理极限）
        joint_max_speeds = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getJointMaxSpeeds()
        print("关节最大速度（物理极限）", joint_max_speeds)
        # 接口调用: 获取关节最大加速度（物理极限）
        joint_max_acc = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getJointMaxAccelerations()
        print("关节最大加速度（物理极限）", joint_max_acc)
        # 接口调用: 获取TCP最大速度（物理极限）
        tcp_max_speeds = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getTcpMaxSpeeds()
        print("TCP最大速度（物理极限）", tcp_max_speeds)
        # 接口调用: 获取TCP最大加速度（物理极限）
        tcp_max_acc = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getTcpMaxAccelerations()
        print("TCP最大加速度（物理极限）", tcp_max_acc)
        # 接口调用:获取机器人的安装姿态
        gravity = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getGravity()
        print("机器人的安装姿态", gravity)
        # 接口调用: 获取TCP偏移
        tcp_offset = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getTcpOffset()
        print("TCP偏移", tcp_offset)
        # 接口调用: 获取末端负载
        payload = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getPayload()
        print("末端负载如下:")
        print("mass:", payload[0])
        print("cog:", payload[1])
        print("aom:", payload[2])
        print("inertia:", payload[3])
        # 接口调用: 获取固件升级的进程
        firmware_update_process = self.robot_rpc_client.getRobotInterface(robot_name).getRobotConfig().getFirmwareUpdateProcess()
        print("固件升级的进程", firmware_update_process)