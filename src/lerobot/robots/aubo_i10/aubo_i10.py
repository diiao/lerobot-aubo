from dataclasses import dataclass
from typing import Any, Dict

from lerobot.robots import Robot, RobotConfig
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots.config import RobotConfig
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)

from .config_aubo_i10 import AuboI10Config
import pyaubo_sdk, time, math
import logging

class AuboI10Robot(Robot):
    config_class = AuboI10Config
    name = "aubo_i10"

    def __init__(self, config: AuboI10Config):
        super().__init__(config)
        self.config = config

        self.robot_ip = "192.168.31.200"  # 机械臂 IP 地址
        self.robot_port = 30004  # 端口号

        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        # 全局变量，供其他函数使用
        self.robot_name = None
        self.robot_interface = None
        #软爪控制相关属性#################################################################################################################
        self.softpaws_open_pin = 4  # 数字输出DO04
        self.softpaws_close_pin = 5  # 数字输出DO05
        self.is_softpaws_open = False
        self.is_softpaws_close = False
        self.io_control = None

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
                # 登录成功后初始化全局变量
                self.robot_name = self.robot_rpc_client.getRobotNames()[0]
                self.robot_interface = self.robot_rpc_client.getRobotInterface(self.robot_name)
                print(f"{'='*8} Robot status {'='*8}")
                self.get_robot_status()
                print(f"{'='*8} End robot status {'='*8}")
                 # 初始化IO控制接口##################################################################################################

                self.io_control = self.robot_interface.getIoControl()
                print(f"软爪控制初始化完成 - 打开引脚: {self.softpaws_open_pin}, 关闭引脚: {self.softpaws_close_pin}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        # TODO(Rory):  Get the video from Mech-Mind camera
        return {"test": 1}
    
    def send_action(self, action: RobotAction) -> RobotAction:
        self.robot_interface.getMotionControl().setSpeedFraction(0.85)
        position = [
            action["ee.x"],
            action["ee.y"],
            action["ee.z"],
            action["ee.wx"],
            action["ee.wy"],
            action["ee.wz"],
        ]
        # ============ 软爪控制部分 ============###############################################################################
        #gripper = action["ee.gripper_pos"] # TODO(Rory): make gripper useful
        # 从action中获取夹爪位置（百分比 0-100）
        print(action)
        gripper_pos = action.get("ee.gripper_pos", 0)
        print(f"\n发送动作 - 夹爪位置: {gripper_pos}%")
        # 根据夹爪位置控制软爪
        self._control_softpaws_based_on_gripper(gripper_pos)
        # ============ 软爪控制部分结束 ============#########################################################
        self.robot_interface.getMotionControl().moveLine(position, 1.2, 0.25, 0, 0)
        _ = self.wait_arrival(self.robot_interface)
        return action
########################################################################
    def _control_softpaws_based_on_gripper(self, gripper_pos: float):
    
        try:
            if gripper_pos > 60:
                print("leader夹爪超过60度，打开aubo夹爪")
                if hasattr(self, 'softpaws_open'):
                    # 确保软爪处于open状态，如果当前是close状态则先关闭close
                    if hasattr(self, 'is_softpaws_close') and self.is_softpaws_close:
                        self.softpaws_close_off()
                    # 如果软爪当前不是open状态，则执行open
                    if hasattr(self, 'is_softpaws_open') and not self.is_softpaws_open:
                        self.softpaws_open()
                else:
                    print("aubo_i10没有软爪控制功能")
                    
            elif gripper_pos < 20:
                print("Leader 夹爪关闭到20度以下，关闭 aubo 软爪")
                if hasattr(self, 'softpaws_close'):
                    # 确保软爪处于close状态，如果当前是open状态则先关闭open
                    if hasattr(self, 'is_softpaws_open') and self.is_softpaws_open:
                        self.softpaws_open_off()
                    # 如果软爪当前不是close状态，则执行close
                    if hasattr(self, 'is_softpaws_close') and not self.is_softpaws_close:
                        self.softpaws_close()
                else:
                    print("警告：aubo robot 没有软爪控制方法")
                    
            else:
                # 20-60度之间，关闭所有夹爪状态
                print("Leader 夹爪在20-60度之间，关闭aubo软爪所有状态")
                #if hasattr(robot, 'softpaws_open_off') and hasattr(robot, 'is_softpaws_open') and robot.is_softpaws_open:
                self.softpaws_open_off()
                #if hasattr(robot, 'softpaws_close_off') and hasattr(robot, 'is_softpaws_close') and robot.is_softpaws_close:
                self.softpaws_close_off()
            
        except Exception as e:
            print(f"控制软爪时发生错误: {e}")
##############################################################################
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
        # 使用全局变量
        name = self.robot_interface.getRobotConfig().getName()
        print("机器人的名字:", name)
        # 接口调用: 获取机器人的自由度
        dof = self.robot_interface.getRobotConfig().getDof()
        print("机器人的自由度:", dof)
        # 接口调用: 获取机器人的伺服控制周期（从硬件抽象层读取）
        cycle_time = self.robot_interface.getRobotConfig().getCycletime()
        print("伺服控制周期:", cycle_time)
        # 接口调用: 获取默认的工具端加速度，单位: m/s^2
        default_tool_acc = self.robot_interface.getRobotConfig().getDefaultToolAcc()
        print("默认的工具端加速度:", default_tool_acc)
        # 接口调用: 获取默认的工具端速度，单位: m/s
        default_tool_speed = self.robot_interface.getRobotConfig().getDefaultToolSpeed()
        print("默认的工具端速度:", default_tool_speed)
        # 接口调用: 获取默认的关节加速度，单位: rad/s
        default_joint_acc = self.robot_interface.getRobotConfig().getDefaultJointAcc()
        print("默认的关节加速度", default_joint_acc)
        # 接口调用: 获取默认的关节速度，单位: rad/s
        default_joint_speed = self.robot_interface.getRobotConfig().getDefaultJointSpeed()
        print("默认的关节加速度", default_joint_speed)
        # 接口调用: 获取机器人类型代码
        # robot_type = self.robot_interface.getRobotConfig().getRobotType()
        # print("机器人类型代码", robot_type)
        # # 接口调用: 获取机器人子类型代码
        # sub_robot_type = self.robot_interface.getRobotConfig().getRobotSubType()
        # print("机器人子类型代码", sub_robot_type)
        # # 接口调用: 获取控制柜类型代码
        # control_box_type = self.robot_interface.getRobotConfig().getControlBoxType()
        # print("控制柜类型代码", control_box_type)
        # # 接口调用: 获取安装位姿(机器人的基坐标系相对于世界坐标系)
        # mounting_pose = self.robot_interface.getRobotConfig().getMountingPose()
        # print("安装位姿(机器人的基坐标系相对于世界坐标系)", mounting_pose)
        # # 接口调用: 设置碰撞灵敏度等级
        # level = 6
        # self.robot_interface.getRobotConfig().setCollisionLevel(level)
        # print("设置碰撞灵敏度等级:", level)
        # # 接口调用: 获取碰撞灵敏度等级
        # level = self.robot_interface.getRobotConfig().getCollisionLevel()
        # print("碰撞灵敏度等级", level)
        # # 接口调用: 设置碰撞停止类型
        # collision_stop_type = 1
        # self.robot_interface.getRobotConfig().setCollisionStopType(collision_stop_type)
        # print("设置碰撞停止类型", collision_stop_type)
        # # 接口调用: 获取碰撞停止类型
        # collision_stop_type = self.robot_interface.getRobotConfig().getCollisionStopType()
        # print("碰撞停止类型", collision_stop_type)
        # # 接口调用: 获取机器人DH参数
        # kin_param = self.robot_interface.getRobotConfig().getKinematicsParam(True)
        # print("机器人DH参数", kin_param)
        # # 接口调用: 获取指定温度下的DH参数补偿值
        # temperature = 20
        # kin_compensate = self.robot_interface.getRobotConfig().getKinematicsCompensate(
        #     temperature)
        # print("指定温度下的DH参数补偿值", kin_compensate)
        # # 接口调用: 获取可用的末端力矩传感器的名字
        # tcp_force_sensor_name = self.robot_interface.getRobotConfig().getTcpForceSensorNames()
        # print("可用的末端力矩传感器的名字", tcp_force_sensor_name)
        # # 接口调用: 获取末端力矩偏移
        # tcp_force_offset = self.robot_interface.getRobotConfig().getTcpForceOffset()
        # print("末端力矩偏移", tcp_force_offset)
        # # 接口调用: 获取可用的底座力矩传感器的名字
        # base_force_sensor_names = self.robot_interface.getRobotConfig().getBaseForceSensorNames()
        # print("可用的底座力矩传感器的名字", base_force_sensor_names)
        # # 接口调用: 获取底座力矩偏移
        # base_force_offset = self.robot_interface.getRobotConfig().getBaseForceOffset()
        # print("底座力矩偏移", base_force_offset)
        # 接口调用: 获取安全参数校验码 CRC
    ###############################################################################################
    def softpaws_open(self):
        try:
            if self.io_control is None:
                print("错误: IO控制接口未初始化，请先调用connect方法")
                return False
            if self.is_softpaws_close:
                self.softpaws_close_off()
                time.sleep(0.05)
            self.io_control.setStandardDigitalOutput(self.softpaws_open_pin, True)
            self.is_softpaws_open = True
            print("softpaws is open")
            return True
        except Exception as e:
            print(f"打开软爪失败: {e}")
            return False

    def softpaws_open_off(self):
        try:
            if self.io_control is None:
                print("错误: IO控制接口未初始化，请先调用connect方法")
                return False
            self.io_control.setStandardDigitalOutput(self.softpaws_open_pin, False)
            self.is_softpaws_open = False
            print("softpaws open off")
            return True
        except Exception as e:
            print(f"关闭软爪失败: {e}")
            return False
            
    def softpaws_close(self):
        try:
            if self.io_control is None:
                print("错误: IO控制接口未初始化，请先调用connect方法")
                return False
            if self.is_softpaws_open:
                self.softpaws_open_off()
                time.sleep(0.05)
            self.io_control.setStandardDigitalOutput(self.softpaws_close_pin, True)
            self.is_softpaws_close = True
            print("softpaws is close")
            return True
        except Exception as e:
            print(f"关闭软爪失败: {e}")
            return False
        
    def softpaws_close_off(self):
        try:
            if self.io_control is None:
                print("错误: IO控制接口未初始化，请先调用connect方法")
                return False
            self.io_control.setStandardDigitalOutput(self.softpaws_close_pin, False)
            self.is_softpaws_close = False
            print("softpaws close off")
            return True
        except Exception as e:
            print(f"关闭软爪失败: {e}")
            return False