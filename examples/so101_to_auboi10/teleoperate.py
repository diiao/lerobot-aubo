from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera
# from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

import threading
import time
import logging

FPS = 30  # 30Hz 的发送频率（可以根据需要调整）

def main():
    init_logging()
    logging.getLogger().setLevel(logging.DEBUG)

    leader_config = SO101LeaderConfig(
        port="/dev/ttyACM0",
        id="so101_leader",
        use_degrees=True
    )

    follower_config = AuboI10Config(
        id="aubo_i10"        
    )

    custom_config = OpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        # width=640,
        # height=480,
        color_mode=ColorMode.RGB,
        # rotation=Cv2Rotation.ROTATE_90
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
    # init_rerun(session_name="so101_aubo_joint_teleop_v02")

    stop_event = threading.Event()                # 停止信号
    lock = threading.Lock()                       # 保护 latest_action
    latest_action = None                          # 最新的动作（dict）
    send_frequency = 30                           # 发送频率 30Hz

    def send_thread():
        interval = 1.0 / send_frequency

        while not stop_event.is_set():
            with lock:
                if latest_action is not None:
                    action_to_send = latest_action.copy()

                    try:
                        t_send = time.perf_counter()
                        follower.send_action(action_to_send)
                        dt_send = (time.perf_counter() - t_send) * 1000
                        print(f"[Send] {dt_send:.1f}ms | 示例: {list(action_to_send.values())[:3]}...")  # 调试
                    except Exception as e:
                        print(f"send_action 失败: {e}")

            # 固定频率轮询
            precise_sleep(interval, 0.0)

    # 启动发送线程
    send_thread_obj = threading.Thread(target=send_thread, daemon=True)
    send_thread_obj.start()




    try:
        frame_count = 0
    
        while True:
            t0 = time.perf_counter()

            t1 = time.perf_counter()
            leader_action = leader.get_action()
            dt_get = (time.perf_counter() - t1) * 1000  # 修正 t1 - t0 → time.perf_counter() - t1


            # 只更新 latest_action + 死区滤波
            with lock:
                if latest_action is None:
                    # 第一次直接更新
                    latest_action = leader_action.copy()
                else:
                    # 计算最大变化（忽略 gripper 如果有）
                    max_delta = max(
                        abs(leader_action[k] - latest_action[k])
                        for k in leader_action
                        if k != 'gripper.pos' and k in latest_action
                    )
                    if max_delta > 1.0:  # 调这个阈值：1.0~3.0，根据 leader 抖动
                        latest_action = leader_action.copy()

            dt_total = (time.perf_counter() - t0) * 1000
            frame_count += 1

            if frame_count % 30 == 0:  # 每3秒打印一次（因为FPS=10）
                print(f"Loop: {dt_total:.1f}ms | get_action: {dt_get:.1f}ms")

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