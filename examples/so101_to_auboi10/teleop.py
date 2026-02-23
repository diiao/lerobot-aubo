from lerobot.robots.aubo_i10.aubo_i10_02 import AuboI10Robot02, AuboI10Config
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

import time
import logging

FPS = 30

def main():
    init_logging()

    leader_config = SO101LeaderConfig(
        port="/dev/ttyACM0",
        id="so101_leader",
        use_degrees=True
    )

    follower_config = AuboI10Config(
        id="aubo_i10_02"           # 改这里，方便区分
    )

    custom_config = OpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=480,
        height=640,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.ROTATE_90
    )

    leader = SO101Leader(leader_config)
    follower = AuboI10Robot02(follower_config)   # ← 这里改成 AuboI10Robot02
    custom_camera = OpenCVCamera(custom_config)

    leader.connect()
    follower.connect()
    custom_camera.connect()

    init_rerun(session_name="so101_aubo_joint_teleop_v02")

    print("开始遥操作（使用 AuboI10Robot02 版本，第5轴固定）")
    print(f"固定第5轴角度: {follower.fixed_axis5_deg} 度")

    try:
        while True:
            t0 = time.perf_counter()

            leader_action = leader.get_action()
            logging.info(f"Leader joints: {leader_action}")

            sent_action = follower.send_action(leader_action)

            log_rerun_data(observation=leader_action, action=sent_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n遥操作停止")

    finally:
        leader.disconnect()
        follower.disconnect()
        custom_camera.disconnect()


if __name__ == "__main__":
    main()