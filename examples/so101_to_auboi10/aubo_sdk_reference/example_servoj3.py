#! /usr/bin/env python
# coding=utf-8

"""
功能：机械臂 servoJ 叠加力控示例

步骤:
第一步: 设置 RPC 超时、连接 RPC 服务、机械臂登录
第二步: servoJ 叠加力控示例
第三步: RPC 退出登录、断开连接
"""

import time
import math
import pyaubo_sdk
import csv
from typing import List, Sequence, Tuple, Any, Optional

robot_ip = "127.0.0.1"  # 服务器 IP 地址
robot_port = 30004  # 端口号
M_PI = 3.14159265358979323846
tcp_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mass = 0.0
com = [0.0, 0.0, 0.0]
t = 0.05
robot_rpc_client = pyaubo_sdk.RpcClient()
EMBEDDED: bool = True # 是否使用机械臂内置力传感器
SENSOR_VENDOR: str = "kw_ftsensor"  # 外置力传感器厂商名称


# 阻塞
def wait_arrival(robot_interface):
    max_retry_count = 5
    cnt = 0

    # 接口调用: 获取当前的运动指令 ID
    exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂开始运动
    while exec_id == -1:
        if cnt > max_retry_count:
            return -1
        time.sleep(0.05)
        cnt += 1
        exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂运动完成
    while robot_interface.getMotionControl().getExecId() != -1:
        time.sleep(0.05)

    return 0

def calculate_distance(p1: Sequence[float], p2: Optional[Sequence[float]] = None) -> float:
    """
    计算两点之间的欧氏距离，默认第二点为原点
    """
    if p2 is None:
        p2 = (0.0,) * 6
    if len(p1) != 6 or len(p2) != 6:
        return 0.0
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(3)))

def waitServoJointComplete(cli: Any, timeout_ms=10000) -> int:
    # 接口调用：获取机器人的名字
    robot_name = cli.getRobotNames()[0]
    robot_interface = cli.getRobotInterface(robot_name)
    # 记录开始时间
    start_time = time.perf_counter()

    # 第一步：等待运动剩余时间为0（循环检查getMotionLeftTime(0)）
    while robot_interface.getMotionControl().getMotionLeftTime(0) != 0:
        # 计算已耗时（毫秒）
        current_time = time.perf_counter()
        elapsed_ms = (current_time - start_time) * 1000
        # 检查是否超时
        if elapsed_ms >= timeout_ms:
            print(f"Warning: WaitServoJointComplete timeout after {elapsed_ms:.0f} ms")
            return 1
        # 每5ms检查一次
        time.sleep(0.005)
    # 第二步：等待机械臂进入稳定状态（isSteady()为True）
    while not robot_interface.getRobotState().isSteady():
        time.sleep(0.005)

    return 0

def switch_servoj_mode(cli: Any, mode: int) -> int:
    # 接口调用：获取机器人的名字
    robot_name = cli.getRobotNames()[0]
    robot_interface = cli.getRobotInterface(robot_name)
    mc = robot_interface.getMotionControl()
    mc.setServoModeSelect(mode)
    i = 0
    while mc.getServoModeSelect() != mode:
        i = i + 1
        if i > 20:
            print("Servo Mode is ", mc.getServoModeSelect())
            return -1
        time.sleep(0.007)
    return 0

def fc_enable(cli: Any)->int:
    # 接口调用：获取机器人的名字
    robot_name = cli.getRobotNames()[0]
    robot = cli.getRobotInterface(robot_name)
    # 开启力控
    ret = robot.getForceControl().fcEnable()
    if ret != 0:
        print("力控开启失败, 错误码: ", ret)
        return -1

def fc_disable(cli: Any)->int:
    # 接口调用：获取机器人的名字
    robot_name = cli.getRobotNames()[0]
    robot = cli.getRobotInterface(robot_name)
    # 关闭力控
    ret = robot.getForceControl().fcDisable()
    if ret != 0:
        print("力控关闭失败, 错误码: ", ret)
        return -1

def tcp_sensor_test(cli: Any, csv_path: str = "force.csv"):
    # 接口调用：获取机器人的名字
    robot_name = cli.getRobotNames()[0]

    # 选择力传感器类型
    if EMBEDDED:
        sensor_pose = [0, 0, 0, 0, 0, 0]
        cli.getRobotInterface(robot_name) \
            .getRobotConfig() \
            .selectTcpForceSensor("embedded")
    else:
        sensor_pose = [0, 0, 0.047, 0, 0, 0]
        cli.getRobotInterface(robot_name) \
            .getRobotConfig() \
            .selectTcpForceSensor(SENSOR_VENDOR)

    # 写入 CSV
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # 表头
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
        f.flush()

        while True:
            sensor_data = cli.getRobotInterface(robot_name) \
                             .getRobotState() \
                             .getTcpForceSensors()

            print("--------------------------------------")
            for i, v in enumerate(sensor_data):
                print(f"获取TCP力传感器读数: {i + 1}: {v}")
            print("--------------------------------------")

            writer.writerow(list(sensor_data))
            f.flush()

            time.sleep(0.005)

# --------- 力传感器标定 ----------
def tcp_sensor_calibration(cli: Any, joints: List[List[float]]) -> Tuple[List[float], List[float], float]:
    robot_name = cli.getRobotNames()[0]
    robot_interface = cli.getRobotInterface(robot_name)

    print("goto p0")
    robot_interface.getMotionControl().moveJoint(
        joints[0],
        10.0 * (math.pi / 180.0),
        5.0 * (math.pi / 180.0),
        0, 0
    )
    wait_arrival(robot_interface)
    time.sleep(1.0)

    q1 = robot_interface.getRobotState().getJointPositions()
    tcp_force1 = robot_interface.getRobotState().getTcpForceSensors()

    print("goto p1")
    robot_interface.getMotionControl().moveJoint(
        joints[1],
        20.0 * (math.pi / 180.0),
        10.0 * (math.pi / 180.0),
        0, 0
    )
    wait_arrival(robot_interface)
    time.sleep(1.0)

    q2 = robot_interface.getRobotState().getJointPositions()
    tcp_force2 = robot_interface.getRobotState().getTcpForceSensors()

    print("goto p2")
    robot_interface.getMotionControl().moveJoint(
        joints[2],
        20.0 * (math.pi / 180.0),
        10.0 * (math.pi / 180.0),
        0, 0
    )
    wait_arrival(robot_interface)
    time.sleep(1.0)

    q3 = robot_interface.getRobotState().getJointPositions()
    tcp_force3 = robot_interface.getRobotState().getTcpForceSensors()

    pose1 = robot_interface.getRobotAlgorithm().forwardKinematics(q1)
    pose2 = robot_interface.getRobotAlgorithm().forwardKinematics(q2)
    pose3 = robot_interface.getRobotAlgorithm().forwardKinematics(q3)

    calib_forces = [list(tcp_force1), list(tcp_force2), list(tcp_force3)]
    calib_poses = [list(pose1[0]), list(pose2[0]), list(pose3[0])]

    # 三点标定接口
    result = robot_interface.getRobotAlgorithm().calibrateTcpForceSensor(calib_forces, calib_poses)

    return result

def exampleServoj1(cli: Any):
    robot_name = cli.getRobotNames()[0]  # 接口调用: 获取机器人的名字
    robot = cli.getRobotInterface(robot_name)


    # 选择力传感器类型
    if EMBEDDED:
        sensor_pose = [0, 0, 0, 0, 0, 0]
        robot.getRobotConfig().selectTcpForceSensor("embedded")
    else:
        sensor_pose = [0, 0, 0.047, 0, 0, 0]
        robot.getRobotConfig().selectTcpForceSensor(SENSOR_VENDOR)

    # 设置传感器安装位姿
    robot.getRobotConfig().setTcpForceSensorPose(sensor_pose)

    # 设置工具坐标系偏移
    robot.getRobotConfig().setTcpOffset(tcp_offset)

    # 设置负载
    robot.getRobotConfig().setPayload(mass, com, [0.], [0.])

    # 读取并设置力传感器偏移
    force_offset = robot.getRobotState().getTcpForceSensors()
    robot.getRobotConfig().setTcpForceOffset(force_offset)

    # 导纳模型
    admittance_m = [25.0, 25.0, 25.0, 2.0, 2.0, 2.0]
    admittance_d = [300.0, 300.0, 300.0, 12.0, 12.0, 12.0]
    admittance_k = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    robot.getForceControl().setDynamicModel(admittance_m, admittance_d, admittance_k)

    # 力控目标
    compliance = [False, False, True, False, False, False]
    target_wrench = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    speed_limits = [2.0] * 6
    feature = [0, 0, 0, 0, 0, 0]
    task_frame = pyaubo_sdk.TaskFrameType.TOOL_FORCE
    robot.getForceControl().setTargetForce(feature, compliance, target_wrench, speed_limits, task_frame)

    # 读取轨迹文件并加载轨迹点
    file = open('../trajs/record6.offt')
    traj = []
    for line in file:
        str_list = line.split(",")
        float_list = []
        for strs in str_list:
            float_list.append(float(strs))
        traj.append(float_list)

    traj_sz = len(traj)
    if traj_sz == 0:
        print("没有轨迹点")
    else:
        print("加载的轨迹点数量为: ", traj_sz)

    # 关节运动到第一个点
    # 当前位置要与轨迹中的第一个点一致，否则容易引起较大超调
    print("goto p1")
    mc = robot.getMotionControl()
    mc.moveJoint(traj[0], M_PI, M_PI, 0., 0.)
    wait_arrival(robot)

    # 开启力控
    fc_enable(cli)

    # 开启 servo 模式
    switch_servoj_mode(cli, 7)

    traj.remove(traj[0])
    for q in traj:
        ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        while ret == 2:
            # print("queue full: ", ret)
            # time.sleep(0.005)
            ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        if ret != 0:
            print("servoj error: ", ret)
            robot.getForceControl().fcDisable()
            return -1

        # 5ms 下发节拍
        # time.sleep(0.005)

    # 等待 servo 运动完成
    if (waitServoJointComplete(cli)== 0):
        print("Servoj motion complete")

    # 关闭 servo 模式
    switch_servoj_mode(cli, 0)
    # 关闭力控
    fc_disable(cli)

    print("Servoj end")
    return 0

if __name__ == '__main__':
    robot_rpc_client = pyaubo_sdk.RpcClient()
    robot_rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 超时
    robot_rpc_client.connect(robot_ip, robot_port)  # 接口调用: 连接 RPC 服务
    if robot_rpc_client.hasConnected():
        print("RPC客户端连接成功!")
        robot_rpc_client.login("aubo", "123456")  # 接口调用: 登录
        if robot_rpc_client.hasLogined():
            print("RPC客户端登录成功!")
            exampleServoj1(robot_rpc_client)  # ServoJ示例
            robot_rpc_client.logout()  # 退出登录
            robot_rpc_client.disconnect()  # 断开连接
