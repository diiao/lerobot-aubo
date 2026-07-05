"""SO101 leader → Aubo I10 关节直连遥操（workstation 风格）。

链路:
  leader.get_action()  →  5 关节度 + gripper.pos
    → follower.send_action(leader_action)
        内部做 SO101→Aubo J1-J6 的方向/偏置映射后 servoJoint 伺服
        （J5 固定为 fixed_axis5_deg，J6 跟随 wrist_roll）
        软爪随 gripper.pos 开合

设计要点（丝滑的关键）:
  - 主循环 30Hz 采集 leader，发送线程 30Hz 固定轮询下发，连续喂点让 servoJoint 队列不空转；
  - 小死区 0.1° 仅用于过滤 leader 抖动、决定是否更新 latest_action，发送线程照常 30Hz 发最新值；
  - 不需要任何 processor / FK / calibration.json。

用法:
  PYTHONPATH=src python examples/so101_to_auboi10/teleoperate.py
"""
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, ColorMode
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

import threading
import time
import logging

FPS = 30  # 主循环与发送频率（Hz）
DEADZONE_DEG = 0.1  # leader 关节死区（度），过滤抖动；workstation 用 0.05


def main():
    init_logging()
    logging.getLogger().setLevel(logging.WARNING)  # 减少不必要的日志输出

    leader_config = SO101LeaderConfig(
        port="/dev/ttyACM0",
        id="so101_leader",
        use_degrees=True,
    )

    follower_config = AuboI10Config(id="aubo_i10")

    custom_config = OpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        color_mode=ColorMode.RGB,
    )

    leader = SO101Leader(leader_config)
    follower = AuboI10Robot(follower_config)
    custom_camera = OpenCVCamera(custom_config)

    # 1. 核心控制连接放在主线程（必须先连好才能进入循环）
    print("正在连接 leader 和 follower...")
    leader.connect()
    follower.connect()
    print("leader 和 follower 连接成功")

    # 2. 摄像头连接放到后台线程（不阻塞主循环）
    def connect_camera():
        try:
            print("开始连接摄像头...")
            custom_camera.connect()
            print("摄像头连接成功")
        except Exception as e:
            print(f"摄像头连接失败（程序继续运行）: {e}")

    camera_thread = threading.Thread(target=connect_camera, daemon=True)
    camera_thread.start()

    # 3. 立即进入主循环，不等待摄像头连接完成
    stop_event = threading.Event()  # 停止信号
    lock = threading.Lock()         # 保护 latest_action
    latest_action = None            # 最新的动作（dict）

    def send_thread():
        """30Hz 固定轮询下发最新动作，连续喂点保证 servoJoint 丝滑。"""
        interval = 1.0 / FPS
        while not stop_event.is_set():
            with lock:
                if latest_action is not None:
                    action_to_send = latest_action.copy()
                else:
                    action_to_send = None

            if action_to_send is not None:
                try:
                    t_send = time.perf_counter()
                    follower.send_action(action_to_send)
                    dt_send = (time.perf_counter() - t_send) * 1000
                    print(f"[Send] {dt_send:.1f}ms | 示例: {list(action_to_send.values())[:3]}...")
                except Exception as e:
                    print(f"send_action 失败: {e}")

            precise_sleep(interval, 0.0)

    send_thread_obj = threading.Thread(target=send_thread, daemon=True)
    send_thread_obj.start()

    try:
        frame_count = 0

        while True:
            t0 = time.perf_counter()

            leader_action = leader.get_action()

            if frame_count % 60 == 0:
                print(f"[Leader] 获取的值: {[(k, f'{v:.2f}') for k, v in leader_action.items()]}")

            # 死区滤波：仅决定是否更新 latest_action；发送线程照常 30Hz 下发当前最新值
            with lock:
                if latest_action is None:
                    latest_action = leader_action.copy()
                    print(f"[Init] 首次 leader_action: {list(leader_action.values())[:3]}...")
                else:
                    deltas = {
                        k: abs(leader_action[k] - latest_action[k])
                        for k in leader_action
                        if k != "gripper.pos" and k in latest_action
                    }
                    max_delta = max(deltas.values()) if deltas else 0
                    if max_delta > DEADZONE_DEG:
                        latest_action = leader_action.copy()

            dt_total = (time.perf_counter() - t0) * 1000
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Loop: {dt_total:.1f}ms")

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n遥操作停止")

    finally:
        stop_event.set()  # 通知线程退出
        time.sleep(0.1)   # 给线程时间退出
        leader.disconnect()
        follower.disconnect()
        custom_camera.disconnect()


if __name__ == "__main__":
    main()
