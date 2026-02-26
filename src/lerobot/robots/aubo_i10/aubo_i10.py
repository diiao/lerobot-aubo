from typing import Any, Dict
import logging
import math
import time
import numpy as np

from lerobot.robots import Robot
from .config_aubo_i10 import AuboI10Config
import pyaubo_sdk


class AuboI10Robot(Robot):
    """
    AuboI10Robot 的变体版本
    支持两种控制模式：
    1. 关节角度控制：直接控制 J1-J6 关节角度
    2. 末端位姿控制：使用 Aubo 的 moveLine 接口进行直线运动

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

        # 运动控制参数
        self.line_velocity = 0.25  # 直线运动速度 (m/s)
        self.line_acceleration = 1.2  # 直线运动加速度 (m/s²)
        self.joint_velocity = 80 * (math.pi / 180)  # 关节运动速度 (rad/s)
        self.joint_acceleration = 60 * (math.pi / 180)  # 关节运动加速度 (rad/s²)

        # 伺服模式参数
        self.servo_time = 0.05  # 伺服运动周期，单位：秒（对应 20Hz）
        self.servo_blend_radius = 0.0  # 混合半径
        self.servo_max_queue_retry = 5  # 队列满时的最大重试次数
        self.is_servo_mode_enabled = False  # 伺服模式状态

        
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

    def enable_servo_mode(self) -> bool:
        """
        开启伺服模式

        Returns:
            bool: 成功返回 True，失败返回 False
        """
        if not self.is_connected or not self.robot_interface:
            logging.error("机器人未连接，无法开启伺服模式")
            return False

        try:
            motion = self.robot_interface.getMotionControl()
            motion.setServoMode(True)

            # 等待伺服模式开启，最多重试5次
            retry_count = 0
            max_retries = 5
            while not motion.isServoModeEnabled():
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(f"开启伺服模式失败！当前伺服模式状态：{motion.isServoModeEnabled()}")
                    return False
                time.sleep(0.005)

            self.is_servo_mode_enabled = True
            logging.info("伺服模式已开启")
            return True

        except Exception as e:
            logging.error(f"开启伺服模式时发生错误: {e}")
            return False

    def disable_servo_mode(self) -> bool:
        """
        关闭伺服模式

        Returns:
            bool: 成功返回 True，失败返回 False
        """
        if not self.is_connected or not self.robot_interface:
            logging.warning("机器人未连接")
            return False

        try:
            motion = self.robot_interface.getMotionControl()
            motion.setServoMode(False)

            # 等待伺服模式关闭，最多重试5次
            retry_count = 0
            max_retries = 5
            while motion.isServoModeEnabled():
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(f"关闭伺服模式失败！当前伺服模式状态：{motion.isServoModeEnabled()}")
                    return False
                time.sleep(0.005)

            self.is_servo_mode_enabled = False
            logging.info("伺服模式已关闭")
            return True

        except Exception as e:
            logging.error(f"关闭伺服模式时发生错误: {e}")
            return False

    def get_observation(self) -> dict[str, Any]:
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

        obs_dict["observation.image.handeye"] = obs_dict.pop("handeye", None)
        obs_dict["observation.image.fixed"]   = obs_dict.pop("fixed", None)
        return obs_dict

    def send_action(self, action: dict) -> dict:
        """
        支持两种控制模式：
        1. 关节角度控制：action 包含 J1-J6 和 gripper_pos，角度单位为度
        2. 末端位姿控制：action 包含 ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz 和 gripper_pos
           位置单位为米，姿态单位为弧度

        使用伺服模式进行实时控制
        """
        if not self.is_connected:
            logging.error("机器人未连接，无法发送动作")
            return action

        try:
            motion = self.robot_interface.getMotionControl()
            motion.setSpeedFraction(0.25)

            # 如果伺服模式未开启，则开启
            if not self.is_servo_mode_enabled:
                if not self.enable_servo_mode():
                    logging.error("无法开启伺服模式，退出")
                    return action

            # 检测控制模式：末端位姿控制优先
            ee_keys = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"]
            joint_keys = ["J1", "J2", "J3", "J4", "J5", "J6"]

            if all(key in action for key in ee_keys):
                # 末端位姿控制模式（笛卡尔伺服运动）
                self._send_ee_action_servo(action, motion)
            elif all(key in action for key in joint_keys):
                # 关节角度控制模式（关节伺服运动）
                self._send_joint_action_servo(action, motion)
            else:
                logging.warning(f"action 中缺少必要的控制参数，action keys: {list(action.keys())}")
                return action

            # 处理夹爪
            gripper_pos = action.get("gripper_pos", action.get("ee.gripper_pos", 0.0))
            self._control_softpaws_based_on_gripper(gripper_pos)

        except Exception as e:
            logging.error(f"发送动作失败: {e}")

        return action

    def _send_joint_action(self, action: dict, motion):
        """
        发送关节角度控制指令（普通模式，已废弃）

        Args:
            action: 包含 J1-J6 的动作字典
            motion: 运动控制接口
        """
        # 从 action 中提取各关节角度（单位：度）
        j1_deg = action.get("J1", 0.0)
        j2_deg = action.get("J2", 0.0)
        j3_deg = action.get("J3", 0.0)
        j4_deg = action.get("J4", 0.0)
        j5_deg = action.get("J5", self.fixed_axis5_deg)
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

        # 发送关节运动指令
        motion.moveJoint(aubo_joints_rad, self.joint_velocity, self.joint_acceleration, 0.0, 0.0)

        logging.debug(f"关节运动: J1={j1_deg:.2f}°, J2={j2_deg:.2f}°, J3={j3_deg:.2f}°, "
                     f"J4={j4_deg:.2f}°, J5={j5_deg:.2f}°, J6={j6_deg:.2f}°")

    def _send_joint_action_servo(self, action: dict, motion):
        """
        发送关节角度伺服控制指令（伺服模式）

        Args:
            action: 包含 J1-J6 的动作字典
            motion: 运动控制接口
        """
        # 从 action 中提取各关节角度（单位：度）
        j1_deg = action.get("J1", 0.0)
        j2_deg = action.get("J2", 0.0)
        j3_deg = action.get("J3", 0.0)
        j4_deg = action.get("J4", 0.0)
        j5_deg = action.get("J5", self.fixed_axis5_deg)
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

        # 发送关节伺服运动指令，处理队列满的情况
        # servoJoint(joints, acc, vel, time, blend_radius, max_radius)
        # 参数说明：
        # - joints: 目标关节位置（弧度）
        # - acc: 加速度 (rad/s²)
        # - vel: 速度 (rad/s)
        # - time: 运动时间 (s)，必须匹配控制周期
        # - blend_radius: 混合半径
        # - max_radius: 最大半径
        retry_count = 0
        while retry_count < self.servo_max_queue_retry:
            ret = motion.servoJoint(
                aubo_joints_rad,
                self.joint_acceleration,
                self.joint_velocity,
                self.servo_time,
                self.servo_blend_radius,
                200
            )

            if ret == 2:  # 队列满
                retry_count += 1
                if retry_count >= self.servo_max_queue_retry:
                    logging.warning(f"关节伺服队列持续满载，已重试 {retry_count} 次")
                time.sleep(0.005)
            else:
                break

        logging.debug(f"关节伺服运动: J1={j1_deg:.2f}°, J2={j2_deg:.2f}°, J3={j3_deg:.2f}°, "
                     f"J4={j4_deg:.2f}°, J5={j5_deg:.2f}°, J6={j6_deg:.2f}°")

    def _send_ee_action(self, action: dict, motion):
        """
        发送末端位姿控制指令（普通模式，已废弃）

        Args:
            action: 包含 ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz 的动作字典
                   这些值表示相对于当前位置的增量，单位：米和弧度
            motion: 运动控制接口
        """
        # 获取当前TCP位姿
        current_pose = self.robot_interface.getRobotState().getTcpPose()

        # 提取相对增量
        # 位置单位：米，姿态单位：弧度
        dx = float(action.get("ee.x", 0.0))
        dy = float(action.get("ee.y", 0.0))
        dz = float(action.get("ee.z", 0.0))
        drx = float(action.get("ee.wx", 0.0))
        dry = float(action.get("ee.wy", 0.0))
        drz = float(action.get("ee.wz", 0.0))

        # 计算目标位姿 = 当前位姿 + 增量
        target_pose = [
            current_pose[0] + dx,
            current_pose[1] + dy,
            current_pose[2] + dz,
            current_pose[3] + drx,
            current_pose[4] + dry,
            current_pose[5] + drz
        ]

        # 发送直线运动指令
        motion.moveLine(target_pose, self.line_acceleration, self.line_velocity, 0, 0)

        logging.debug(f"相对增量移动: delta=[{dx:.3f}, {dy:.3f}, {dz:.3f}]m, "
                     f"delta_rot=[{drx:.3f}, {dry:.3f}, {drz:.3f}]rad")
        logging.debug(f"当前位姿: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}]m, "
                     f"[{current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]rad")
        logging.debug(f"目标位姿: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]m, "
                     f"[{target_pose[3]:.3f}, {target_pose[4]:.3f}, {target_pose[5]:.3f}]rad")

    def _send_ee_action_servo(self, action: dict, motion):
        """
        发送末端位姿伺服控制指令（伺服模式）
        只使用位置增量，姿态保持机器人当前姿态不变

        Args:
            action: 包含 ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz 的动作字典
                   只使用位置增量 (ee.x, ee.y, ee.z)，姿态增量被忽略
            motion: 运动控制接口
        """
        current_pose = self.robot_interface.getRobotState().getTcpPose()
        logging.debug(f"当前位姿: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}]m, "
                     f"[{current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]rad")

        dx = float(action.get("ee.x", 0.0))
        dy = float(action.get("ee.y", 0.0))
        dz = float(action.get("ee.z", 0.0))
        dwx = float(action.get("ee.wx", 0.0))
        dry = float(action.get("ee.wy", 0.0))
        drz = float(action.get("ee.wz", 0.0))

        target_pose = [
            current_pose[0] + dx,
            current_pose[1] + dy,
            current_pose[2] + dz, 
            current_pose[3] + dwx,
            current_pose[4] + dry,
            current_pose[5] + drz
        ]

        retry_count = 0
        while retry_count < self.servo_max_queue_retry:
            ret = motion.servoCartesian(
                target_pose,
                self.line_acceleration,
                self.line_velocity,
                self.servo_time,
                self.servo_blend_radius,
                0.0
            )

            if ret == 2:
                retry_count += 1
                if retry_count >= self.servo_max_queue_retry:
                    logging.warning(f"笛卡尔伺服队列持续满载，已重试 {retry_count} 次")
                time.sleep(0.005)
            else:
                break

        logging.debug(f"笛卡尔伺服运动(仅位置): delta=[{dx:.3f}, {dy:.3f}, {dz:.3f}]m")
        logging.debug(f"当前位姿: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}]m, "
                     f"[{current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]rad")
        logging.debug(f"目标位姿: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]m, "
                     f"[{target_pose[3]:.3f}, {target_pose[4]:.3f}, {target_pose[5]:.3f}]rad (姿态保持不变)")

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
        # 断开连接前关闭伺服模式
        if self.is_servo_mode_enabled:
            logging.info("断开连接前关闭伺服模式")
            self.disable_servo_mode()

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