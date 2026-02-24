from typing import Any, Dict
import logging
import math
import time

from lerobot.robots import Robot
from .config_aubo_i10 import AuboI10Config
import pyaubo_sdk


class AuboI10Robot(Robot):
    """
    AuboI10Robot 的变体版本
    映射规则:SO101 leader 5个关节 → Aubo J1, J2, J3, J4, J6
    Aubo J5(第五轴)固定为 self.fixed_axis5_deg 度
    用于测试不同关节映射方案
    """

    config_class = AuboI10Config
    name = "aubo_i10"   

    def __init__(self, config: AuboI10Config):
        super().__init__(config)
        self.config = config

        from lerobot.cameras.utils import make_cameras_from_configs   

        self.cameras = (
            make_cameras_from_configs(config.cameras)
            if hasattr(config, 'cameras') and config.cameras
            else {}
        )
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
        self.fixed_axis5_deg = 90.0   # ← 你可以随时修改这个值

        
    @property
    def is_connected(self) -> bool:
        return self.robot_rpc_client.hasConnected()

    def connect(self, calibrate: bool = True) -> None:
        # connect cameras
        for cam_name, cam in self.cameras.items():
            try:
                cam.connect()
                logging.info(f"{cam_name} connect success")
            except Exception as e:
                logging.error(f"{cam_name} connect fail: {e}")
        # connect robot
        self.robot_rpc_client.setRequestTimeout(1000)
        self.robot_rpc_client.connect(self.robot_ip, self.robot_port)
        if self.robot_rpc_client.hasConnected():
            self.robot_rpc_client.login("aubo", "123456")
            if self.robot_rpc_client.hasLogined():
                self.robot_name = self.robot_rpc_client.getRobotNames()[0]
                self.robot_interface = self.robot_rpc_client.getRobotInterface(self.robot_name)
                self.get_robot_status()
                self.io_control = self.robot_interface.getIoControl()
                logging.info(f"Robot connect success: {self.robot_name}")
                

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        start = time.perf_counter()
        obs_dict = {}
        if self.is_connected and self.robot_interface:
            try:
                # 获取当前关节位置（弧度）
                robot_state = self.robot_interface.getRobotState()
                joints_rad = robot_state.getJointPositions()  # 返回 list[float]，6个值
                # 转成度数（更直观）
                joints_deg = [math.degrees(rad) for rad in joints_rad]
                obs_dict["J1"]  = joints_deg[0]   
                obs_dict["J2"]  = joints_deg[1]   
                obs_dict["J3"]  = joints_deg[2]   
                obs_dict["J4"]  = joints_deg[3]    
                obs_dict["J5"]  = joints_deg[4]
                obs_dict["J6"]  = joints_deg[5]

                # 获取夹爪状态
                # 0.0 = 关闭，100.0 = 打开，50.0 = 中间状态
                if self.is_softpaws_open:
                    obs_dict["gripper_pos"] = 100.0
                elif self.is_softpaws_close:
                    obs_dict["gripper_pos"] = 0.0
                else:
                    obs_dict["gripper_pos"] = 50.0

            except Exception as e:
                logging.error(f"读取关节状态失败: {e}")
        else:
            logging.warning("机器人未连接，无法读取关节状态")

        import os
        save_dir = "debug_camera_images"
        os.makedirs(save_dir, exist_ok=True)  # 提前创建目录，避免每次循环都创建

        for cam_key, cam in self.cameras.items():
            start_cam = time.perf_counter()
            try:
                img = cam.async_read()   # 假设返回 numpy array (h,w,c) 或 PIL Image
                if img is not None:
                    obs_dict[cam_key] = img
                else:
                    obs_dict[cam_key] = None
                    logging.warning(f"相机 {cam_key} 本次无图像")
            except Exception as e:
                logging.error(f"读取相机 {cam_key} 失败: {e}")
                obs_dict[cam_key] = None

            dt_cam = (time.perf_counter() - start_cam) * 1e3
            logging.debug(f"读取 {cam_key}: {dt_cam:.1f}ms")

        dt = (time.perf_counter() - start) * 1e3
        logging.debug(f"get_observation 总耗时: {dt:.1f}ms")
        obs_dict["observation.image.handeye"] = obs_dict.pop("handeye", None)
        obs_dict["observation.image.fixed"]   = obs_dict.pop("fixed", None)
        return obs_dict

    def send_action(self, action: dict) -> dict:
        """
        action应包含六个关节角度以及一个gripper的角度
        角度单位为度
        """
        # 从 action 中提取各关节角度（单位：度）
        j1_deg = action.get("J1", 0.0)
        j2_deg = action.get("J2", 0.0)
        j3_deg = action.get("J3", 0.0)
        j4_deg = action.get("J4", 0.0)
        j5_deg = action.get("J5", 0.0)
        j6_deg = action.get("J6", 0.0)

        # 将关节角度转换为弧度
        j1_rad = math.radians(j1_deg)
        j2_rad = math.radians(j2_deg)
        j3_rad = math.radians(j3_deg)
        j4_rad = math.radians(j4_deg)
        j5_rad = math.radians(j5_deg)
        j6_rad = math.radians(j6_deg)

        # 构建弧度列表
        aubo_joints_rad = [j1_rad, j2_rad, j3_rad, j4_rad, j5_rad, j6_rad]
        motion = self.robot_interface.getMotionControl()
        motion.setSpeedFraction(1)  
        motion.moveJoint(aubo_joints_rad, 0.8, 0.8, 0.0, 0.0)  

        gripper_pos = action.get("gripper_pos", 0.0)
        self._control_softpaws_based_on_gripper(gripper_pos)
        return action # sure?


        if not self.is_connected:
            logging.error("机器人未连接，无法发送动作")
            return action

        motion = self.robot_interface.getMotionControl()
        motion.setSpeedFraction(1)  

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
            logging.debug("Leader 关节角度（度）:")
            for key, value in zip(leader_joint_keys, leader_joints_deg):
                logging.debug(f"  {key:18} : {value:8.3f}°")
        except KeyError as e:
            logging.error(f"缺少关节键: {e}，实际 action: {action}")
            return action
        
        # 校正参数（根据你的数据初步估计，可继续调）
        directions = [-1, -1, 1, -1, 1]      
        offsets     = [ 0,  0, 100, 90, 40]     
        scale       = 1                  

        # 校正 + 缩放
        corrected = []
        for i in range(5):
            deg = leader_joints_deg[i]
            deg = deg * directions[i] + offsets[i]
            deg *= scale
            corrected.append(deg)

        # 构建 Aubo 6关节列表（度）
        aubo_joints_deg = [
            corrected[0],              # J1
            corrected[1],              # J2
            corrected[2],              # J3
            corrected[3],              # J4
            self.fixed_axis5_deg,      # J5 固定
            corrected[4],              # J6
        ]

        # 转换为弧度
        aubo_joints_rad = [math.radians(deg) for deg in aubo_joints_deg]

        # 发送关节运动指令
        motion.moveJoint(aubo_joints_rad, 0.8, 0.8, 0.0, 0.0)  

        # 处理夹爪
        gripper_pos = action.get('gripper.pos', action.get('ee.gripper_pos', 0))
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
                # print("错误: IO控制接口未初始化")
                return False
            if self.is_softpaws_close:
                self.softpaws_close_off()
                time.sleep(0.05)
            self.io_control.setStandardDigitalOutput(self.softpaws_open_pin, True)
            self.is_softpaws_open = True
            return True
        except Exception as e:
            return False

    def softpaws_open_off(self):
        try:
            if self.io_control is None:
                return False
            self.io_control.setStandardDigitalOutput(self.softpaws_open_pin, False)
            self.is_softpaws_open = False
            return True
        except Exception as e:
            return False

    def softpaws_close(self):
        try:
            if self.io_control is None:
                return False
            if self.is_softpaws_open:
                self.softpaws_open_off()
                time.sleep(0.05)
            self.io_control.setStandardDigitalOutput(self.softpaws_close_pin, True)
            self.is_softpaws_close = True
            return True
        except Exception as e:
            return False

    def softpaws_close_off(self):
        try:
            if self.io_control is None:
                return False
            self.io_control.setStandardDigitalOutput(self.softpaws_close_pin, False)
            self.is_softpaws_close = False
            return True
        except Exception as e:
            return False

    def disconnect(self) -> None:
        if self.robot_rpc_client.hasLogined():
            self.robot_rpc_client.logout()
        self.robot_rpc_client.disconnect()


##############################
#开启阻塞，会导致遥操时真机卡顿
    # def wait_arrival(self, robot_interface):
    #     max_retry_count = 5
    #     cnt = 0

    #     motion = robot_interface.getMotionControl()
    #     exec_id = motion.getExecId()

    #     while exec_id == -1:
    #         if cnt > max_retry_count:
    #             return -1
    #         time.sleep(0.001)
    #         cnt += 1
    #         exec_id = motion.getExecId()

    #     while motion.getExecId() != -1:
    #         time.sleep(0.001)

    #     return 0

    def get_robot_status(self):
        if not self.robot_interface:
            # print("机器人接口未初始化")
            return


    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        机器人产生的观测数据的格式，包括关节和相机
        """
        return {
            "J1": float,
            "J2": float,
            "J3": float,
            "J4": float,
            "J5": float,
            "J6": float,
            "gripper_pos": float,
            "observation.image.handeye": (480, 640, 3),  
            "observation.image.fixed":   (480, 640, 3),
        }

    @property
    def action_features(self) -> dict[str, type | tuple]:
        """
        机器人可以执行的动作的格式，包括关节和夹爪
        """
        return {
            "J1": float,     
            "J2": float,     
            "J3": float,  
            "J4": float,     
            "J5": float,     
            "J6": float,     
            "gripper_pos": float,          
        }