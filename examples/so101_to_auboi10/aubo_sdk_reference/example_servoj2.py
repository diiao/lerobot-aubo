#! /usr/bin/env python
# coding=utf-8

"""
servoj 运动

示例说明:
1. exampleServoj1: 采用旧接口 setServoMode 开启 servo 模式, 跟踪一个轨迹,目标点下发时间间隔5ms,截断式
2. exampleServoj2: 采用新接口 mode 1 开启 servo 模式, 跟踪一个轨迹,目标点下发时间间隔5ms,截断式
3. exampleServoj3: 采用新接口 mode 2 开启 servo 模式, 跟踪一个轨迹,目标点下发时间间隔5ms
4. exampleServoj4: 采用新接口 mode 5 开启 servo 模式, 跟踪一个轨迹,目标点下发时间间隔5ms, 路点必达式
"""
import pyaubo_sdk
import time
from typing import Any

robot_ip = "127.0.0.1"  # 服务器 IP 地址
robot_port = 30004  # 端口号
M_PI = 3.14159265358979323846
robot_rpc_client = pyaubo_sdk.RpcClient()


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


# 测试1:(旧接口 setServoMode)
# 采用servoj跟踪一个轨迹,目标点下发时间间隔5ms,截断式,每次更新路点后不保证经过上个路点
def exampleServoj1(cli: Any):
    robot_name = cli.getRobotNames()[0]  # 接口调用: 获取机器人的名字
    robot = cli.getRobotInterface(robot_name)

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
    # robot_rpc_client.getRuntimeMachine().start()
    mc = robot.getMotionControl()
    mc.moveJoint(traj[0], M_PI, M_PI, 0., 0.)
    wait_arrival(robot)

    # 开启 servo 模式
    robot.getMotionControl().setServoMode(True)
    i = 0
    while not mc.isServoModeEnabled():
        i = i + 1
        if i > 5:
            print("Servo Mode is ", mc.isServoModeEnabled())
            return -1
        time.sleep(0.005)

    traj.remove(traj[0])
    t = 0.02
    for q in traj:
        ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        while ret == 2:
            # print("queue full: ", ret)
            time.sleep(0.005)
            ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        if ret != 0:
            print("servoj error: ", ret)
            return -1

    # 等待 servo 运动完成
    if (waitServoJointComplete(cli)== 0):
        print("Servoj motion complete")

    # 关闭 servo 模式
    mc.setServoMode(False)
    print("Servoj end")
    return 0

# 测试2:(新接口 mode 1 等价 setServoMode)
# 采用servoj跟踪一个轨迹,目标点下发时间间隔5ms,截断式,每次更新路点后不保证经过上个路点
def exampleServoj2(cli: Any):
    robot_name = cli.getRobotNames()[0]  # 接口调用: 获取机器人的名字
    robot = cli.getRobotInterface(robot_name)
    mc = robot.getMotionControl()

    # 设置速度比例
    mc.setSpeedFraction(1)

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
    mc.moveJoint(traj[0], M_PI, M_PI, 0., 0.)
    wait_arrival(robot)


    # 开启 servo 模式
    switch_servoj_mode(cli, 1)

    t = 0.02
    traj.remove(traj[0])
    for q in traj:
        ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        while ret == 2:
            # print("queue full: ", ret)
            # time.sleep(0.005)
            ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        if ret != 0:
            print("servoj error: ", ret)
            return -1

        # 5ms 下发节拍
        # time.sleep(0.005)

    # 等待 servo 运动完成
    if (waitServoJointComplete(cli)== 0):
        print("Servoj motion complete")

    # 关闭 servo 模式
    switch_servoj_mode(cli, 0)

    print("Servoj end")
    return 0

# 测试3: 采用servoj mode 2 跟踪一个轨迹,目标点下发时间间隔5ms
def exampleServoj3(cli: Any):
    robot_name = cli.getRobotNames()[0]  # 接口调用: 获取机器人的名字
    robot = cli.getRobotInterface(robot_name)
    mc = robot.getMotionControl()

    # 设置速度比例
    mc.setSpeedFraction(1)

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
    mc.moveJoint(traj[0], M_PI, M_PI, 0., 0.)
    wait_arrival(robot)


    # 开启 servo 模式
    switch_servoj_mode(cli, 2)

    t = 0.1
    traj.remove(traj[0])
    for q in traj:
        ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        while ret == 2:
            # print("queue full: ", ret)
            # time.sleep(0.005)
            ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        if ret != 0:
            print("servoj error: ", ret)
            return -1

        # 5ms 下发节拍
        time.sleep(0.005)

    # 等待 servo 运动完成
    if (waitServoJointComplete(cli)== 0):
        print("Servoj motion complete")

    # 关闭 servo 模式
    switch_servoj_mode(cli, 0)

    print("Servoj end")
    return 0

# 测试4: 采用servoj跟踪一个轨迹,目标点下发时间间隔5ms, 路点必达式
def exampleServoj4(cli: Any):
    robot_name = cli.getRobotNames()[0]  # 接口调用: 获取机器人的名字
    robot = cli.getRobotInterface(robot_name)
    mc = robot.getMotionControl()

    # 设置速度比例
    mc.setSpeedFraction(1)

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
    mc.moveJoint(traj[0], M_PI, M_PI, 0., 0.)
    wait_arrival(robot)


    # 开启 servo 模式
    switch_servoj_mode(cli, 5)

    t = 0.01
    traj.remove(traj[0])
    for q in traj:
        ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        while ret == 2:
            # print("queue full: ", ret)
            # time.sleep(0.005)
            ret = mc.servoJoint(q, 0.1, 0.2, t, 0.1, 200)
        if ret != 0:
            print("servoj error: ", ret)
            return -1

        # 5ms 下发节拍
        # time.sleep(0.005)

    # 等待 servo 运动完成
    if (waitServoJointComplete(cli)== 0):
        print("Servoj motion complete")

    # 关闭 servo 模式
    switch_servoj_mode(cli, 0)

    print("Servoj end")
    return 0

if __name__ == '__main__':
    robot_rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 请求超时时间
    robot_rpc_client.connect(robot_ip, robot_port)  # 接口调用: 连接到 RPC 服务
    if robot_rpc_client.hasConnected():
        print("Robot rcp_client connected successfully!")
        robot_rpc_client.login("aubo", "123456")  # 接口调用: 机械臂登录
        if robot_rpc_client.hasLogined():
            print("Robot rcp_client logined successfully!")
            exampleServoj1(robot_rpc_client)
            # exampleServoj2(robot_rpc_client)
            # exampleServoj3(robot_rpc_client)
            # exampleServoj4(robot_rpc_client)
            robot_rpc_client.disconnect()  # 接口调用: 断开RPC连接

