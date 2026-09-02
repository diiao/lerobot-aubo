from typing import Any, Dict
import logging
import math
import time
import numpy as np

from lerobot.robots import Robot
from .config_aubo_i10 import AuboI10Config
from lerobot.utils.rotation import Rotation
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

        # 吸盘 IO 配置（双引脚互斥控制）
        # 端口 2 = 吸（真空开启）；端口 3 = 不吸（真空关闭/释放）
        self.suction_on_pin = 2    # 拉高 → 吸
        self.suction_off_pin = 3   # 拉高 → 不吸
        self.is_suction_on = False
        self.io_control = None

        # 固定 Aubo 的第五轴角度（单位：度）
        # 建议值：0.0（默认）、90.0、-90.0、180.0 等，根据工具朝向调整
        self.fixed_axis5_deg = 90.0   # ← 你可以随时修改这个值

        # SO101 leader → Aubo 关节映射校准（workstation 默认值，可微调）
        # 顺序对应 leader: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
        #        映射到  Aubo: J1, J2, J3, J4, J6   （J5 固定为 fixed_axis5_deg）
        self.so101_directions = [-1, -1, 1, -1, 1]
        self.so101_offsets = [0, 0, 100, 90, 40]
        self.so101_scale = 1.0

        # 运动控制参数
        self.line_velocity = 0.25  # 直线运动速度 (m/s)
        self.line_acceleration = 1.2  # 直线运动加速度 (m/s²)
        self.joint_velocity = 80 * (math.pi / 180)  # 关节运动速度 (rad/s)
        self.joint_acceleration = 60 * (math.pi / 180)  # 关节运动加速度 (rad/s²)

        # 伺服模式参数
        # 依据 SDK 文档(motion_control.h servoCartesian)："目前可用参数只有 pose 和 t"，
        # 即本版固件 servoCartesian 只有 pose 和 t 生效，lookahead_time/gain 暂被忽略。
        #  - t(运行时间) 最优值 = 连续调用间隔 = 控制循环周期。
        # 原先 t=0.05 > 循环间隔 0.033，导致伺服队列持续积压 → 满(ret=2) → 重试 sleep
        #   → 运动一顿一顿且滞后。改成 0.033 让喂入≈消耗，队列稳定，运动连续顺滑。
        # servo_time 从配置的控制频率计算，避免录制/推理 FPS 改动后忘记同步。
        self.servo_time = 1.0 / config.control_fps
        self.servo_lookahead_time = 0.0  # 前瞻时间：本版固件忽略，保留待高版本固件启用[0.03,0.2]
        self.servo_gain = 0.0  # 比例增益：本版固件忽略，保留待高版本固件启用[100,200]
        self.servo_blend_radius = 0.0  # 混合半径(关节伺服用)
        self.servo_max_queue_retry = 5  # 队列满时的最大重试次数
        self.is_servo_mode_enabled = False  # 伺服模式状态

        # 绝对笛卡尔伺服的姿态连续性：记录上一帧实际发送的 rotvec，
        # 用于把新目标 rotvec 对齐到同一表示分支（解决竖直向下 θ=π 奇异点跳变）。
        self._prev_sent_rotvec = None

        
    @property
    def is_connected(self) -> bool:
        return self.robot_rpc_client.hasConnected()

    def connect(self, calibrate: bool = True) -> None:
        try:
            # A configured learning camera is mandatory. Warmup verifies that the
            # device can deliver a real frame before the robot is connected.
            for cam_name, cam in self.cameras.items():
                try:
                    cam.connect(warmup=True)
                except Exception as exc:
                    raise ConnectionError(
                        f"Failed to connect configured camera '{cam_name}': {exc}"
                    ) from exc
                logging.info(f"{cam_name} connect success")

            self.robot_rpc_client.setRequestTimeout(1000)
            self.robot_rpc_client.connect(self.robot_ip, self.robot_port)
            if not self.robot_rpc_client.hasConnected():
                raise ConnectionError(
                    f"Failed to connect Aubo robot at {self.robot_ip}:{self.robot_port}"
                )

            self.robot_rpc_client.login("aubo", "123456")
            if not self.robot_rpc_client.hasLogined():
                raise ConnectionError("Connected to Aubo RPC, but login failed")

            self.robot_name = self.robot_rpc_client.getRobotNames()[0]
            self.robot_interface = self.robot_rpc_client.getRobotInterface(self.robot_name)
            self.get_robot_status()
            self.io_control = self.robot_interface.getIoControl()
            logging.info(f"Robot connect success: {self.robot_name}")
        except Exception:
            # Never leave camera handles or a partial RPC session alive after a
            # failed startup. Most importantly, do not continue recording with a
            # missing camera.
            self._disconnect_cameras()
            if self.robot_rpc_client.hasConnected():
                self.robot_rpc_client.disconnect()
            raise
                

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
                robot_state = self.robot_interface.getRobotState()
                
                joints_rad = robot_state.getJointPositions()
                joints_deg = [math.degrees(rad) for rad in joints_rad]
                obs_dict["J1"]  = joints_deg[0]   
                obs_dict["J2"]  = joints_deg[1]   
                obs_dict["J3"]  = joints_deg[2]   
                obs_dict["J4"]  = joints_deg[3]    
                obs_dict["J5"]  = joints_deg[4]
                obs_dict["J6"]  = joints_deg[5]

                tcp_pose = robot_state.getTcpPose()
                obs_dict["ee.x"] = float(tcp_pose[0])
                obs_dict["ee.y"] = float(tcp_pose[1])
                obs_dict["ee.z"] = float(tcp_pose[2])
                obs_dict["ee.wx"] = float(tcp_pose[3])
                obs_dict["ee.wy"] = float(tcp_pose[4])
                obs_dict["ee.wz"] = float(tcp_pose[5])

                obs_dict["gripper_pos"] = 100.0 if self.is_suction_on else 0.0

            except Exception as e:
                logging.error(f"读取关节状态失败: {e}")
        else:
            logging.warning("机器人未连接，无法读取关节状态")

        for cam_key, cam in self.cameras.items():
            start_cam = time.perf_counter()
            try:
                img = cam.read_latest()
            except Exception as latest_error:
                # read_latest 失败时回退到 async_read（阻塞等待新帧）
                try:
                    img = cam.async_read(timeout_ms=200)
                except Exception as e:
                    raise RuntimeError(
                        f"相机 {cam_key} 读取失败；为避免黑帧污染数据，已中止本轮"
                    ) from e
                else:
                    logging.debug(
                        "相机 %s 的最新帧不可用，已等待下一帧: %s",
                        cam_key,
                        latest_error,
                    )
            obs_dict[cam_key] = img
            dt_cam = (time.perf_counter() - start_cam) * 1e3
            logging.debug(f"读取 {cam_key}: {dt_cam:.1f}ms")

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

            # 检查伺服模式状态：同时验证本地标志和机器人实际状态
            # 机器人伺服模式可能因超时自动失效（等待按键期间无命令发送），
            # 此时本地 is_servo_mode_enabled 仍为 True，需同步修正
            if not self.is_servo_mode_enabled or not motion.isServoModeEnabled():
                self.is_servo_mode_enabled = False  # 与机器人实际状态同步
                if not self.enable_servo_mode():
                    logging.error("无法开启伺服模式，退出")
                    return action

            # 检测控制模式：末端位姿控制优先
            ee_keys = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"]
            joint_keys = ["J1", "J2", "J3", "J4", "J5", "J6"]
            so101_keys = [
                "shoulder_pan.pos", "shoulder_lift.pos",
                "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
            ]

            if action.get("ee_mode") == "abs_j6yaw" and all(key in action for key in ee_keys):
                # 位置跟随 + J6 直控偏航模式：IK 求 J1-J5 保持固定竖直姿态(pitch/roll锁死)，
                # J6 单独叠加手机左右旋转 → 末端只绕法兰轴(竖直)转，位置跟手机走。
                self._send_position_j6yaw(action, motion)
            elif action.get("ee_mode") == "absolute" and all(key in action for key in ee_keys):
                # 末端位姿绝对控制模式（笛卡尔伺服，完整 6 维位姿，无 delta 累积）
                self._send_ee_action_servo_absolute(action, motion)
            elif all(key in action for key in ee_keys):
                # 末端位姿增量控制模式（笛卡尔伺服运动，ee.* 为相对当前 TCP 的增量）
                self._send_ee_action_servo(action, motion)
            elif all(key in action for key in so101_keys):
                # SO101 leader 关节直连模式：内部做 SO101→Aubo J1-J6 映射后关节伺服
                self._send_so101_joint_action(action, motion)
            elif all(key in action for key in joint_keys):
                # 关节角度控制模式（关节伺服运动）
                self._send_joint_action_servo(action, motion)
            else:
                logging.warning(f"action 中缺少必要的控制参数，action keys: {list(action.keys())}")
                return action

            # 处理夹爪（兼容 SO101 的 gripper.pos / 关节模式的 gripper_pos / EE 的 ee.gripper_pos）
            gripper_pos = action.get(
                "gripper.pos",
                action.get("gripper_pos", action.get("ee.gripper_pos", 0.0)),
            )
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

    def _send_position_j6yaw(self, action: dict, motion):
        """位置跟随 + J6 直控偏航。

        - 位置 (ee.x/y/z) 与固定姿态 (ee.wx/wy/wz，按下瞬间锁存的竖直向下姿态) 一起
          做逆解 → J1..J6，保证 pitch/roll 锁死、末端始终竖直向下。
        - 再把 J6 直接设为绝对目标 ee.j6_target(=按下瞬间 J6 + 手机偏航累计, 弧度)：
          末端只绕法兰轴(工具竖直时即基座 Z/竖直方向)旋转，别的姿态不动。
        - 用关节伺服 (servoJoint) 下发。IK 以当前关节为种子，帧间连续平滑。

        这样彻底绕开笛卡尔姿态伺服里 IK 把偏航分配到 J4/J5(表现为绕基座 Y 俯仰)
        的问题——偏航只由 J6 承担。J6 用绝对目标 ee.j6_target，IK 只负责 J1-J5。
        """
        target_pose = [
            float(action.get("ee.x", 0.0)),
            float(action.get("ee.y", 0.0)),
            float(action.get("ee.z", 0.0)),
            float(action.get("ee.wx", 0.0)),
            float(action.get("ee.wy", 0.0)),
            float(action.get("ee.wz", 0.0)),
        ]

        current_q = self.robot_interface.getRobotState().getJointPositions()
        res = self.robot_interface.getRobotAlgorithm().inverseKinematics(current_q, target_pose)
        q_sol, errno = list(res[0]), int(res[1])
        if errno != 0:
            logging.warning(
                f"逆解失败 errno={errno}，跳过本帧。目标 pos="
                f"[{target_pose[0]:.3f},{target_pose[1]:.3f},{target_pose[2]:.3f}]m"
            )
            return

        # J6 用绝对目标（IK 只负责 J1-J5 的位置与竖直姿态；J6 单独承担手机偏航）。
        # 工具竖直时 J6 轴=竖直方向，故这就是绕竖直方向旋转。
        if "ee.j6_target" in action:
            q_sol[5] = float(action["ee.j6_target"])

        retry_count = 0
        while retry_count < self.servo_max_queue_retry:
            ret = motion.servoJoint(
                q_sol,
                self.joint_acceleration,
                self.joint_velocity,
                self.servo_time,
                self.servo_blend_radius,
                200
            )
            if ret == 2:  # 队列满
                retry_count += 1
                if retry_count >= self.servo_max_queue_retry:
                    logging.warning(f"J6偏航关节伺服队列持续满载，已重试 {retry_count} 次")
                time.sleep(0.005)
            else:
                if ret != 0:
                    logging.warning(f"J6偏航关节伺服返回非 0 码 ret={ret}")
                break

        logging.debug(
            f"位置+J6偏航: pos=[{target_pose[0]:.3f},{target_pose[1]:.3f},{target_pose[2]:.3f}]m, "
            f"J6={math.degrees(q_sol[5]):.1f}°"
        )

    def _send_so101_joint_action(self, action: dict, motion):
        """
        SO101 leader 关节直连模式：把 leader 的 5 个关节角度（度）按方向/偏置/缩放
        映射到 Aubo 的 J1,J2,J3,J4,J6，J5 固定为 fixed_axis5_deg，然后复用关节伺服通路下发。

        映射顺序对应：
            leader  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
            aubo    J1,           J2,            J3,         J4,         J6
        校准参数 so101_directions / so101_offsets / so101_scale 在 __init__ 中可调。
        """
        leader_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos",
            "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
        ]
        try:
            leader_deg = [action[k] for k in leader_keys]
        except KeyError as e:
            logging.error(f"SO101 关节直连缺少键: {e}，实际 action: {list(action.keys())}")
            return

        corrected = [
            leader_deg[i] * self.so101_directions[i] + self.so101_offsets[i]
            for i in range(5)
        ]
        corrected = [d * self.so101_scale for d in corrected]

        joint_action = {
            "J1": corrected[0],
            "J2": corrected[1],
            "J3": corrected[2],
            "J4": corrected[3],
            "J5": self.fixed_axis5_deg,
            "J6": corrected[4],
        }
        self._send_joint_action_servo(joint_action, motion)

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
        发送末端位姿伺服控制指令（伺服模式，delta 模式）

        action 中的 ee.x/ee.y/ee.z 为位置增量（米），ee.wx/ee.wy/ee.wz 为姿态
        增量（rotvec 弧度）。target = current_pose + delta，每帧重读 current_pose，
        保证目标始终接近当前位姿，servoCartesian 连续可接受。

        Args:
            action: 包含 ee.x..ee.wz 的动作字典（均为相对增量）
            motion: 运动控制接口
        """
        current_pose = self.robot_interface.getRobotState().getTcpPose()

        dx = float(action.get("ee.x", 0.0))
        dy = float(action.get("ee.y", 0.0))
        dz = float(action.get("ee.z", 0.0))
        dwx = float(action.get("ee.wx", 0.0))
        dwy = float(action.get("ee.wy", 0.0))
        dwz = float(action.get("ee.wz", 0.0))

        # 旋转 delta 用矩阵复合应用（基座系叠加），避免 rotvec 线性相加在大角度
        # （rotvec 模长 > π）下失效的问题。target_R = R(delta) @ R(current)，
        # 转回 rotvec 后用 _align_rotvec 对齐到当前 TCP 的表示分支——servoCartesian
        # 对 rotvec 表示敏感，未对齐的等价表示会被 IK 拒绝 (ret=-5)。
        from lerobot.robots.aubo_i10.robot_processor import _align_rotvec

        cur_rotvec = np.array(current_pose[3:6], dtype=float)
        delta_rotvec = np.array([dwx, dwy, dwz], dtype=float)
        if float(np.linalg.norm(delta_rotvec)) > 1e-9:
            target_R = Rotation.from_rotvec(delta_rotvec) * Rotation.from_rotvec(cur_rotvec)
            tgt_rotvec = _align_rotvec(target_R.as_rotvec(), cur_rotvec)
        else:
            tgt_rotvec = cur_rotvec

        target_pose = [
            current_pose[0] + dx,
            current_pose[1] + dy,
            current_pose[2] + dz,
            float(tgt_rotvec[0]),
            float(tgt_rotvec[1]),
            float(tgt_rotvec[2]),
        ]

        # 伺服模式：acc=vel=0；启用 lookahead_time + gain 平滑轨迹（抑制卡顿），
        # time=控制周期。非零 acc/vel 会触发参数非法 (ret=-5)，故 a=v=0。
        retry_count = 0
        while retry_count < self.servo_max_queue_retry:
            ret = motion.servoCartesian(
                target_pose,
                0.0,
                0.0,
                self.servo_time,
                self.servo_lookahead_time,
                self.servo_gain
            )

            if ret == 2:
                retry_count += 1
                if retry_count >= self.servo_max_queue_retry:
                    logging.warning(f"笛卡尔伺服队列持续满载，已重试 {retry_count} 次")
                time.sleep(0.005)
            else:
                if ret != 0:
                    logging.warning(
                        f"笛卡尔伺服返回非 0 码 ret={ret} (servo_enabled={motion.isServoModeEnabled()})，"
                        f"delta pos=[{dx:.3f},{dy:.3f},{dz:.3f}]m, "
                        f"delta rot=[{dwx:.3f},{dwy:.3f},{dwz:.3f}]rad, "
                        f"当前=[{current_pose[0]:.3f},{current_pose[1]:.3f},{current_pose[2]:.3f}]m, "
                        f"rotvec=[{current_pose[3]:.3f},{current_pose[4]:.3f},{current_pose[5]:.3f}]rad"
                    )
                break

        logging.debug(f"笛卡尔伺服: delta=[{dx:.3f},{dy:.3f},{dz:.3f}]m, "
                     f"drot=[{dwx:.3f},{dwy:.3f},{dwz:.3f}]rad, "
                     f"目标=[{target_pose[0]:.3f},{target_pose[1]:.3f},{target_pose[2]:.3f}]m")

    def _continuous_rotvec(self, target_rotvec, prev_rotvec):
        """把 target_rotvec 表示成与 prev_rotvec 最接近的等价 rotvec。

        同一旋转有多种等价 rotvec 表示：绕轴 ±2π，以及 θ=π 时的反向轴 (-v)。
        servoCartesian 对 rotvec 数值敏感，且"末端竖直向下"恒为 θ=π（rotvec 模长≡π，
        正处于奇异点）。若每帧直接用 as_rotvec() 主值，绕基座 Z 旋转时轴会在 XY 平面
        扫动并在某处翻转符号，产生 ~2π 的数值跳变 → IK 解出俯仰/大跳动 → 又抖又俯仰。

        本函数枚举 target 旋转的等价表示（含 -v，用旋转角一致性过滤，保证只保留真正
        等价的表示），选出与 prev_rotvec 数值最近的一个，保证帧间 rotvec 连续。
        """
        R = Rotation.from_rotvec(np.asarray(target_rotvec, dtype=float))
        v0 = R.as_rotvec()
        n = float(np.linalg.norm(v0))
        if n < 1e-9:
            return v0
        axis = v0 / n
        prev = np.asarray(prev_rotvec, dtype=float)
        candidates = []
        for base in (v0, -v0):
            for k in (-1, 0, 1):
                c = base + 2.0 * np.pi * k * axis
                # 仅保留与目标旋转真正一致的等价表示（-v 只有在 θ≈π 时成立）
                diff = float(np.linalg.norm((Rotation.from_rotvec(c) * R.inv()).as_rotvec()))
                if diff < 1e-4:
                    candidates.append(c)
        if not candidates:
            candidates = [v0]
        return np.asarray(min(candidates, key=lambda c: float(np.linalg.norm(c - prev))), dtype=float)

    def _send_ee_action_servo_absolute(self, action: dict, motion):
        """
        发送末端位姿绝对伺服控制指令（伺服模式）

        与 _send_ee_action_servo 不同，action 中的 ee.x..ee.wz 为 Aubo 基座系下的
        绝对目标位姿（位置单位米，姿态 rotvec 弧度），完整 6 维直接传给 servoCartesian，
        不做 current+delta、不丢弃 wx/wy。用于 FK 末端到末端映射、手机遥操竖直锁定等
        需要全姿态控制的场景。

        姿态连续性：新目标 rotvec 对齐到"上一帧已发送的 rotvec"（而非实测 current，
        后者带噪声且滞后，在 θ=π 奇异点会诱发俯仰跳动），保证伺服平滑。

        Args:
            action: 包含 ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz 的动作字典
            motion: 运动控制接口
        """
        target_rotvec = np.array(
            [
                float(action.get("ee.wx", 0.0)),
                float(action.get("ee.wy", 0.0)),
                float(action.get("ee.wz", 0.0)),
            ],
            dtype=float,
        )

        # 首帧用实测 TCP 姿态播种参考分支，之后始终跟踪上一帧发送值
        if self._prev_sent_rotvec is None:
            current_pose = self.robot_interface.getRobotState().getTcpPose()
            self._prev_sent_rotvec = np.array(current_pose[3:6], dtype=float)

        aligned_rotvec = self._continuous_rotvec(target_rotvec, self._prev_sent_rotvec)
        self._prev_sent_rotvec = aligned_rotvec

        target_pose = [
            float(action.get("ee.x", 0.0)),
            float(action.get("ee.y", 0.0)),
            float(action.get("ee.z", 0.0)),
            float(aligned_rotvec[0]),
            float(aligned_rotvec[1]),
            float(aligned_rotvec[2]),
        ]

        # servoCartesian 伺服模式：acc=vel=0；启用 lookahead_time + gain 平滑轨迹(抑制卡顿)。
        # 非零 acc/vel 在伺服模式下会触发参数非法 (ret=-5)，故用纯时间 + 前瞻/增益平滑。
        retry_count = 0
        while retry_count < self.servo_max_queue_retry:
            ret = motion.servoCartesian(
                target_pose,
                0.0,
                0.0,
                self.servo_time,
                self.servo_lookahead_time,
                self.servo_gain
            )

            if ret == 2:  # 队列满
                retry_count += 1
                if retry_count >= self.servo_max_queue_retry:
                    logging.warning(f"绝对笛卡尔伺服队列持续满载，已重试 {retry_count} 次")
                time.sleep(0.005)
            else:
                if ret != 0:
                    logging.warning(
                        f"绝对笛卡尔伺服返回非 0 码 ret={ret} (servo_enabled={motion.isServoModeEnabled()})，目标 "
                        f"pos=[{target_pose[0]:.3f},{target_pose[1]:.3f},{target_pose[2]:.3f}]m, "
                        f"rotvec=[{target_pose[3]:.3f},{target_pose[4]:.3f},{target_pose[5]:.3f}]rad"
                    )
                break

        logging.debug(f"绝对笛卡尔伺服: 目标 pos=[{target_pose[0]:.3f},{target_pose[1]:.3f},"
                     f"{target_pose[2]:.3f}]m, rotvec=[{target_pose[3]:.3f},{target_pose[4]:.3f},"
                     f"{target_pose[5]:.3f}]rad")

    def _control_suction_based_on_gripper(self, gripper_pos: float):
        """根据 gripper_pos 控制吸盘。
        
        gripper_pos > 60  → A 键按下 → 激活真空（吸）
        gripper_pos < 20  → B 键按下 → 关闭真空（放）
        20 ≤ pos ≤ 60     → 无按键   → 保持当前状态（不切换）
        """
        try:
            if gripper_pos > 60:
                if not self.is_suction_on:
                    self.suction_activate()
            elif gripper_pos < 20:
                if self.is_suction_on:
                    self.suction_release()
            # neutral: maintain current state
        except Exception as e:
            logging.error(f"控制吸盘时发生错误: {e}")

    # 兼容旧版调用名
    _control_softpaws_based_on_gripper = _control_suction_based_on_gripper

    def suction_activate(self):
        """开启真空吸附：端口2拉高、端口3拉低。"""
        try:
            if self.io_control is None:
                return False
            self.io_control.setStandardDigitalOutput(self.suction_off_pin, False)
            self.io_control.setStandardDigitalOutput(self.suction_on_pin, True)
            self.is_suction_on = True
            logging.info("吸盘：真空开启（吸）端口2=ON, 端口3=OFF")
            return True
        except Exception as e:
            logging.error(f"吸盘激活失败: {e}")
            return False

    def suction_release(self):
        """关闭真空（释放物体）：端口3拉高、端口2拉低。"""
        try:
            if self.io_control is None:
                return False
            self.io_control.setStandardDigitalOutput(self.suction_on_pin, False)
            self.io_control.setStandardDigitalOutput(self.suction_off_pin, True)
            self.is_suction_on = False
            logging.info("吸盘：真空关闭（放）端口2=OFF, 端口3=ON")
            return True
        except Exception as e:
            logging.error(f"吸盘释放失败: {e}")
            return False

    def _disconnect_cameras(self) -> None:
        for cam_name, cam in self.cameras.items():
            if not cam.is_connected:
                continue
            try:
                cam.disconnect()
            except Exception:
                logging.exception("断开相机 %s 失败", cam_name)

    def disconnect(self) -> None:
        # 断开连接前关闭伺服模式
        if self.is_servo_mode_enabled:
            logging.info("断开连接前关闭伺服模式")
            self.disable_servo_mode()

        self._disconnect_cameras()
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
        机器人产生的观测数据的格式，包括关节、末端位姿和相机
        """
        return {
            "J1": float,
            "J2": float,
            "J3": float,
            "J4": float,
            "J5": float,
            "J6": float,
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.wx": float,
            "ee.wy": float,
            "ee.wz": float,
            "gripper_pos": float,
            "handeye": (480, 640, 3),
            "fixed": (480, 640, 3),
        }

    @property
    def action_features(self) -> dict[str, type | tuple]:
        """
        机器人可以执行的动作的格式，末端位姿控制模式
        """
        return {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.wx": float,
            "ee.wy": float,
            "ee.wz": float,
            "ee.gripper_pos": float,
        }
