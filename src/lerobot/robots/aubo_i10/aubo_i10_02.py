from typing import Any, Dict
import logging
import math
import time

from lerobot.robots import Robot
from .config_aubo_i10 import AuboI10Config
import pyaubo_sdk


class AuboI10Robot02(Robot):
    """
    AuboI10Robot 的变体版本 02
    映射规则:SO101 leader 5个关节 → Aubo J1, J2, J3, J4, J6
    Aubo J5(第五轴)固定为 self.fixed_axis5_deg 度
    用于测试不同关节映射方案
    """

    config_class = AuboI10Config
    name = "aubo_i10_02"   # 与原版区分开

    def __init__(self, config: AuboI10Config):
        super().__init__(config)
        self.config = config

        self.robot_ip = "192.168.31.200"
        self.robot_port = 30004

        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        self.robot_name = None
        self.robot_interface = None

        # 软爪相关
        self.softpaws_open_pin = 4
        self.softpaws_close_pin = 5
        self.is_softpaws_open = False
        self.is_softpaws_close = False
        self.io_control = None

        # 固定 Aubo 的第五轴角度（单位：度）
        # 建议值：0.0（默认）、90.0、-90.0、180.0 等，根据工具朝向调整
        self.fixed_axis5_deg = 0.0   # ← 你可以随时修改这个值

    @property
    def is_connected(self) -> bool:
        return self.robot_rpc_client.hasConnected()

    def connect(self, calibrate: bool = True) -> None:
        self.robot_rpc_client.setRequestTimeout(1000)
        self.robot_rpc_client.connect(self.robot_ip, self.robot_port)
        if self.robot_rpc_client.hasConnected():
            print("RPC客户端连接成功!")
            self.robot_rpc_client.login("aubo", "123456")
            if self.robot_rpc_client.hasLogined():
                print("RPC客户端登录成功!")
                self.robot_name = self.robot_rpc_client.getRobotNames()[0]
                self.robot_interface = self.robot_rpc_client.getRobotInterface(self.robot_name)
                print(f"{'='*8} Robot status {'='*8}")
                self.get_robot_status()
                print(f"{'='*8} End robot status {'='*8}")

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
        # TODO: 后续加上真实关节状态、相机等观测
        return {"test": 1}

    def send_action(self, action: dict) -> dict:
        """
        leader 的 5 个关节 → Aubo 的 J1,J2,J3,J4,J6
        Aubo J5 使用固定值 self.fixed_axis5_deg
        """
        if not self.is_connected:
            logging.error("机器人未连接，无法发送动作")
            return action

        motion = self.robot_interface.getMotionControl()
        motion.setSpeedFraction(0.20)  # 建议调试时从 0.15~0.30 开始

        # leader 常见的关节名称（如果实际不同，请在这里修改）
        leader_joint_keys = [
            'shoulder_pan.pos',     # → Aubo J1
            'shoulder_lift.pos',    # → Aubo J2
            'elbow_flex.pos',       # → Aubo J3
            'wrist_flex.pos',       # → Aubo J4
            'wrist_roll.pos',       # → Aubo J6
        ]

        try:
            leader_joints_deg = [action[key] for key in leader_joint_keys]
        except KeyError as e:
            logging.error(f"缺少关节键: {e}，实际 action: {action}")
            return action

        # 构建 Aubo 6关节列表（度）
        aubo_joints_deg = [
            leader_joints_deg[0],      # J1
            leader_joints_deg[1],      # J2
            leader_joints_deg[2],      # J3
            leader_joints_deg[3],      # J4
            self.fixed_axis5_deg,      # J5 ← 固定
            leader_joints_deg[4],      # J6 ← leader 的 wrist_roll
        ]

        # 转换为弧度
        aubo_joints_rad = [math.radians(deg) for deg in aubo_joints_deg]

        logging.debug(f"发送关节 (弧度): {aubo_joints_rad}")

        # 发送关节运动指令
        motion.movej(aubo_joints_rad, 0.3, 0.3, 0)  # 非阻塞，速度30%，加速度30%

        # 如果需要阻塞等待完成，可以打开下面这行（遥操作通常不阻塞）
        # _ = self.wait_arrival(self.robot_interface)

        # 处理夹爪
        gripper_pos = action.get('gripper.pos', action.get('ee.gripper_pos', 0))
        logging.info(f"夹爪位置: {gripper_pos}")
        self._control_softpaws_based_on_gripper(gripper_pos)

        # 返回发送的内容（用于记录或可视化）
        sent = {
            'j1': aubo_joints_deg[0],
            'j2': aubo_joints_deg[1],
            'j3': aubo_joints_deg[2],
            'j4': aubo_joints_deg[3],
            'j5_fixed': aubo_joints_deg[4],
            'j6': aubo_joints_deg[5],
            'gripper.pos': gripper_pos
        }
        return sent

    def _control_softpaws_based_on_gripper(self, gripper_pos: float):
        try:
            if gripper_pos > 60:
                if self.is_softpaws_close:
                    self.softpaws_close_off()
                if not self.is_softpaws_open:
                    self.softpaws_open()
            elif gripper_pos < 20:
                if self.is_softpaws_open:
                    self.softpaws_open_off()
                if not self.is_softpaws_close:
                    self.softpaws_close()
            else:
                self.softpaws_open_off()
                self.softpaws_close_off()
        except Exception as e:
            logging.error(f"控制软爪时发生错误: {e}")

    def softpaws_open(self):
        try:
            if self.io_control is None:
                print("错误: IO控制接口未初始化")
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
                print("错误: IO控制接口未初始化")
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
                print("错误: IO控制接口未初始化")
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
                print("错误: IO控制接口未初始化")
                return False
            self.io_control.setStandardDigitalOutput(self.softpaws_close_pin, False)
            self.is_softpaws_close = False
            print("softpaws close off")
            return True
        except Exception as e:
            print(f"关闭软爪失败: {e}")
            return False

    def disconnect(self) -> None:
        if self.robot_rpc_client.hasLogined():
            self.robot_rpc_client.logout()
        self.robot_rpc_client.disconnect()

    def wait_arrival(self, robot_interface):
        max_retry_count = 5
        cnt = 0

        motion = robot_interface.getMotionControl()
        exec_id = motion.getExecId()

        while exec_id == -1:
            if cnt > max_retry_count:
                return -1
            time.sleep(0.001)
            cnt += 1
            exec_id = motion.getExecId()

        while motion.getExecId() != -1:
            time.sleep(0.001)

        return 0

    def get_robot_status(self):
        if not self.robot_interface:
            print("机器人接口未初始化")
            return

        config = self.robot_interface.getRobotConfig()
        print("机器人的名字:", config.getName())
        print("机器人的自由度:", config.getDof())
        print("伺服控制周期:", config.getCycletime())
        print("默认的工具端加速度:", config.getDefaultToolAcc())
        print("默认的工具端速度:", config.getDefaultToolSpeed())
        print("默认的关节加速度:", config.getDefaultJointAcc())
        print("默认的关节速度:", config.getDefaultJointSpeed())

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {}

    @property
    def action_features(self) -> dict[str, type]:
        return {}